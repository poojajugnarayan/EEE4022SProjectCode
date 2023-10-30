# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import os, sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import *


def scalable_integrate_rgb_frames(path_dataset, intrinsic, config):
    poses = []
    # [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    
    color_files=[]
    depth_files=[]
    for i in range(0,0+config['n_frames']):
        num= str(i)
        # col= config["path_dataset"]+config["path_color"]+num+'.png' 
        # dep= config["path_dataset"]+config["path_depth"]+num+'.png' 
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
            
        color_raw = o3d.io.read_image(col)
        depth_raw = o3d.io.read_image(dep)
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
        color_files.append(color_raw)
        depth_files.append(depth_raw)
    # color=[]
    # depth=[]
    # color_files=[]
    # depth_files=[]
    # for i in range(0,config['n_frames']):
    #         num= str(i)
    #         # col= input_dir+segmented_dir+'segmented_'+num+'.png' 
    #         # col= config["path_dataset"]+config["path_color"]+num+'.png' 
    #         # dep= config["path_dataset"]+config["path_depth"]+num+'.png' 
      
    #         color_raw = o3d.io.read_image(col)
    #         depth_raw = o3d.io.read_image(dep)
    #         # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    #         color.append(color_raw)
    #         depth.append(depth_raw)
            
    # for i in range(6):
    #     for j in range(i, i+5):
    #         # print(j)
    #         color_files.append(color[j])
    #         depth_files.append(depth[j])
    
    
    n_fragments = int(config['n_frames']/config['n_frames_per_fragment'])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    temp=path_dataset+config["template_refined_posegraph_optimized"]+"pose.json"
    pose_graph_fragment = o3d.io.read_pose_graph(temp)

    for fragment_id in range(len(pose_graph_fragment.nodes)):
        temp=path_dataset+config["template_fragment_posegraph_optimized"]+"frag"+str(fragment_id)+".json"
        pose_graph_rgbd = o3d.io.read_pose_graph(temp)

        for frame_id in range(len(pose_graph_rgbd.nodes)):
            frame_id_abs = fragment_id * config['n_frames_per_fragment'] + frame_id
            print(
                "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
                (fragment_id, n_fragments - 1, frame_id_abs, frame_id + 1,
                 len(pose_graph_rgbd.nodes)))
            # rgbd = read_rgbd_image(color_files[frame_id_abs],depth_files[frame_id_abs], False, config)
            
            rgbd= o3d.geometry.RGBDImage.create_from_color_and_depth(color_files[frame_id_abs], depth_files[frame_id_abs],convert_rgb_to_intensity=False)
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            poses.append(pose)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])     

    mesh_name = path_dataset+ config["template_global_mesh"]+"mesh.ply"
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

    traj_name = path_dataset+config["template_global_traj"]+"traj.txt"
    write_poses_to_log(traj_name, poses)


def run(config):
    # print("integrate the whole RGBD sequence using estimated camera pose.")
    #   Define your custom camera intrinsic parameters
    width = 1280  # Image width
    height = 720  # Image height
    fx = 640.0  # Focal length in pixels (x-axis)
    fy = 640.0  # Focal length in pixels (y-axis)
    cx = 640  # Principal point (x-axis)
    cy = 360  # Principal point (y-axis)

    # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    scalable_integrate_rgb_frames(config["path_dataset"], intrinsic, config)


    

