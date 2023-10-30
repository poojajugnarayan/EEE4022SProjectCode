# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/make_fragments.py

import math
import os, sys
import numpy as np
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimize_posegraph import optimize_posegraph_for_fragment

# check opencv python package
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
    # source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
    #                                     config)
    # target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
    #                                     config)
    source_rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_files[s], depth_files[s],convert_rgb_to_intensity=False)
    target_rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_files[t], depth_files[t],convert_rgb_to_intensity=False)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["depth_diff_max"]
    if abs(s - t) != 1:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info
                ] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                             t - sid,
                                                             trans,
                                                             info,
                                                             uncertain=False))
            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - sid, t - sid, trans, info, uncertain=True))
                
                if not success:
                    with open('Test Images/skips.txt','a') as file:
                        file.write('i skipped')
                    # print("works")
    fid=str(fragment_id)
    temp=path_dataset+config["template_fragment_posegraph"]+"frag"+fid+".json"
    o3d.io.write_pose_graph(temp,pose_graph)


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print(
            "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
            (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_files[i_abs], depth_files[i_abs],convert_rgb_to_intensity=False)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    fid=str(fragment_id)

    temp=path_dataset+config["template_fragment_posegraph_optimized"]+"frag"+fid+".json"
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        temp, intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    
    pcd_name = path_dataset+config["template_fragment_pointcloud"]+"frag"+fid+".ply"
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)
    
    # mesh_name = path_dataset+ config["template_global_mesh"]+"mesh1.ply"
    # o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)
        


def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments, config):
#   Define your custom camera intrinsic parameters
    width = 1280  # Image width
    height = 720  # Image height
    fx = 531.0  # Focal length in pixels (x-axis)
    fy = 531.0  # Focal length in pixels (y-axis)
    cx = 637  # Principal point (x-axis)
    cy = 331  # Principal point (y-axis)

    # Create a PinholeCameraIntrinsic object
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                 depth_files, fragment_id, n_fragments,
                                 intrinsic, config)


def run(config):

    print("making fragments from RGBD sequence.")
    # make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))

    # [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    
    color_files=[]
    depth_files=[]
    for i in range(0,0+config['n_frames']):
        num= str(i)
        col= config["path_dataset"]+config["path_color"]+num+'.png' 
        dep= config["path_dataset"]+config["path_depth"]+num+'.png' 
        if(i<10):
            col= config["path_dataset"]+'00000'+num+'_color.png' 
            dep= config["path_dataset"]+'00000'+num+'_aligned_depth.png' 
        elif(i<100):
            col= config["path_dataset"]+'0000'+num+'_color.png' 
            dep= config["path_dataset"]+'0000'+num+'_aligned_depth.png'
        elif(i<1000):
            col= config["path_dataset"]+'000'+num+'_color.png' 
            dep= config["path_dataset"]+'000'+num+'_aligned_depth.png'  
        else:
            col= config["path_dataset"]+'00'+num+'_color.png' 
            dep= config["path_dataset"]+'00'+num+'_aligned_depth.png' 

        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
        color_files.append(o3d.io.read_image(col))
        depth_files.append(o3d.io.read_image(dep))
        
    # color=[]
    # depth=[]
    # color_files=[]
    # depth_files=[]
    # for i in range(0,10):
    #     num= str(i)
    #     # col= input_dir+segmented_dir+'segmented_'+num+'.png' 
    #     col= config["path_dataset"]+config["path_color"]+num+'.png' 
    #     dep= config["path_dataset"]+config["path_depth"]+num+'.png' 
    #     color_raw = o3d.io.read_image(col)
    #     depth_raw = o3d.io.read_image(dep)
    #     # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    #     color.append(color_raw)
    #     depth.append(depth_raw)
        
    # for i in range(6):
    #     for j in range(i, i+5):
    #         # print(j)
    #         color_files.append(color[j])
    #         depth_files.append(depth[j])
            
    n_files = config['n_frames']
    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))

    if config["python_multi_threading"] is True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
        Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
            fragment_id, color_files, depth_files, n_files, n_fragments, config)
                                    for fragment_id in range(n_fragments))
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, config)
    
