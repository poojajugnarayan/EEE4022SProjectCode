import math
import os, sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt



def process_single_fragment(fragment_id, color_files, depth_files, n_files,
                            n_fragments):
#   Define your custom camera intrinsic parameters
    width = 1280  # Image width
    height = 720  # Image height
    fx = 640.0  # Focal length in pixels (x-axis)
    fy = 640.0  # Focal length in pixels (y-axis)
    cx = 640  # Principal point (x-axis)
    cy = 360  # Principal point (y-axis)

    # Create a PinholeCameraIntrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    sid = fragment_id * 9
    eid = min(sid + 9, n_files)

    make_posegraph_for_fragment( sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic)
    optimize_posegraph_for_fragment( fragment_id, )
    make_pointcloud_for_fragment(color_files,
                                 depth_files, fragment_id, n_fragments,
                                 intrinsic,)

def optimize_posegraph_for_fragment(fragment_id):
    pose_graph_name = "Test Images/pose graphs/fragment"+fragment_id+".json" #Check suffix
    pose_graph_optimized_name = "Test Images/pose graphs optimized/fragment"+fragment_id+".json" #Check suffix
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
            max_correspondence_distance = config["depth_diff_max"],
            preference_loop_closure = \
            config["preference_loop_closure_odometry"])



def make_posegraph_for_fragment(sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic):
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
                                                intrinsic)
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
            if s %9 == 0 \
                    and t % 9 == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            s - sid, t - sid, trans, info, uncertain=True))
    temp="Test Images/pose graphs/fragment"+fragment_id+".json" #Check suffix
    o3d.io.write_pose_graph(temp,pose_graph)



def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic):
    # source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
    #                                     config)
    # target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
    #                                     config)

    source_rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_files[s], depth_files[s],convert_rgb_to_intensity=False)
    target_rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_files[t], depth_files[t],convert_rgb_to_intensity=False)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = 10 #Used random number lol
    if abs(s - t) != 1:
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]



def make_pointcloud_for_fragment( color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic):
    temp="Test Images/pose graphs optimized/fragment"+fragment_id+".json" #Check suffix
    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        temp,  intrinsic)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name="Test Images/point cloud/fragment"+fragment_id+".ply" #Check suffix

    o3d.io.write_point_cloud(pcd_name, pcd, False, True)


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
















if __name__ == "__main__":
    print("hi")
    input_dir= 'Test Images/'
    colour_dir= 'images/'
    depth_dir= 'depth/'
    segmented_dir= 'segmented/'
    output_dir='Mesh Tests/'
    # rgbd_images=[]
    colour=[]
    depth=[]
    total=27
    for i in range(0,total):
        num= str(i)
        # col= input_dir+segmented_dir+'segmented_'+num+'.png' 
        col= input_dir+colour_dir+'image_'+num+'.png' 
        dep= input_dir+depth_dir+'depth_'+num+'.png' 
        color_raw = o3d.io.read_image(col)
        depth_raw = o3d.io.read_image(dep)
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
        colour.append(color_raw)
        depth.append(depth_raw)
        # rgbd_images(rgbd_image)
    n_files= total
    n_fragments=3
    for i in range(n_fragments):
        process_single_fragment(i, colour, depth,
                                    n_files, n_fragments)
    # end for