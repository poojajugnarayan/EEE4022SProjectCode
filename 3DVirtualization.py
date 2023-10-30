# import open3d as o3d
# import numpy as np
# import os
# import sys
# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     pcds = []
#     merged_pcd = o3d.geometry.PointCloud()
#     input_dir= 'Test Images/'
#     colour_dir= 'images/'
#     depth_dir= 'depth/'
#     segmented_dir= 'segmented/'
#     output_dir='Mesh Tests/'
#     for i in range(0,1):
#         num1= str(i)
#         num2= str(i+1)
#         # col= input_dir+segmented_dir+'segmented_'+num+'.png' 
        
#         col1= input_dir+colour_dir+'image_'+num1+'.png' 
#         dep1= input_dir+depth_dir+'depth_'+num1+'.png' 
#         col2= input_dir+colour_dir+'image_'+num2+'.png' 
#         dep2= input_dir+depth_dir+'depth_'+num2+'.png' 

#         color_raw1 = o3d.io.read_image(col1)
#         depth_raw1 = o3d.io.read_image(dep1)
#         color_raw2 = o3d.io.read_image(col2)
#         depth_raw2 = o3d.io.read_image(dep2)

#         rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw1, depth_raw1,convert_rgb_to_intensity=False)
#         rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw2, depth_raw2,convert_rgb_to_intensity=False)

#         # Define your custom camera intrinsic parameters
#         width = 1280  # Image width
#         height = 720  # Image height
#         fx = 640.0  # Focal length in pixels (x-axis)
#         fy = 640.0  # Focal length in pixels (y-axis)
#         cx = 640  # Principal point (x-axis)
#         cy = 360  # Principal point (y-axis)

#         # Create a PinholeCameraIntrinsic object
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)        

#         pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image2,intrinsic)
        
#         option = o3d.pipelines.odometry.OdometryOption()
#         odo_init = np.identity(4)

#         [success_color_term, trans_color_term,
#         info] = o3d.pipelines.odometry.compute_rgbd_odometry(
#             rgbd_image1, rgbd_image2, intrinsic, odo_init,
#             o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
#         [success_hybrid_term, trans_hybrid_term,
#         info] = o3d.pipelines.odometry.compute_rgbd_odometry(
#             rgbd_image1, rgbd_image2, intrinsic, odo_init,
#             o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        
#         if success_color_term:
#             print("Using RGB-D Odometry")
#             print(trans_color_term)
#             source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
#                 rgbd_image1, intrinsic)
#             source_pcd_color_term.transform(trans_color_term)
#             o3d.visualization.draw_geometries([pcd2, source_pcd_color_term],
#                                             zoom=0.48,
#                                             front=[0.0999, -0.1787, -0.9788],
#                                             lookat=[0.0345, -0.0937, 1.8033],
#                                             up=[-0.0067, -0.9838, 0.1790])
#         if success_hybrid_term:
#             print("Using Hybrid RGB-D Odometry")
#             print(trans_hybrid_term)
#             source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
#                 rgbd_image1, intrinsic)
#             source_pcd_hybrid_term.transform(trans_hybrid_term)
#             o3d.visualization.draw_geometries([pcd2, source_pcd_hybrid_term],
#                                             zoom=0.48,
#                                             front=[0.0999, -0.1787, -0.9788],
#                                             lookat=[0.0345, -0.0937, 1.8033],
#                                             up=[-0.0067, -0.9838, 0.1790])
                
        
#     #     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])     
#     #     pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.02) 
#     #     pcds.append(pcd_downsampled)
        
#     #     merged_pcd += pcd
#     #     # pcds.append(pcd)
#     #     print("run",num)
    
#     # # pcd_downsampled = merged_pcd.voxel_down_sample(voxel_size=0.02) 
#     # o3d.visualization.draw_geometries(pcds)
    
#     # merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=9)
#     # o3d.visualization.draw_geometries([mesh])
    
#     # output_filename = output_dir+"mesh_3.ply"
#     # o3d.io.write_triangle_mesh(output_filename,mesh)
#     # print("image mesh 3 created")




# import open3d as o3d
# import numpy as np
# import os
# import sys
# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     pcds = []
#     merged_pcd = o3d.geometry.PointCloud()
#     input_dir= 'Test Images/'
#     colour_dir= 'images/'
#     depth_dir= 'depth/'
#     segmented_dir= 'segmented/'
#     output_dir='Mesh Tests/'
#     for i in range(0,1):
#         num1= str(i)
#         num2= str(i+1)
#         # col= input_dir+segmented_dir+'segmented_'+num+'.png' 
        
#         col1= input_dir+colour_dir+'image_'+num1+'.png' 
#         dep1= input_dir+depth_dir+'depth_'+num1+'.png' 
#         col2= input_dir+colour_dir+'image_'+num2+'.png' 
#         dep2= input_dir+depth_dir+'depth_'+num2+'.png' 

#         color_raw1 = o3d.io.read_image(col1)
#         depth_raw1 = o3d.io.read_image(dep1)
#         color_raw2 = o3d.io.read_image(col2)
#         depth_raw2 = o3d.io.read_image(dep2)

#         rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw1, depth_raw1,convert_rgb_to_intensity=False)
#         rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw2, depth_raw2,convert_rgb_to_intensity=False)

#         # Define your custom camera intrinsic parameters
#         width = 1280  # Image width
#         height = 720  # Image height
#         fx = 640.0  # Focal length in pixels (x-axis)
#         fy = 640.0  # Focal length in pixels (y-axis)
#         cx = 640  # Principal point (x-axis)
#         cy = 360  # Principal point (y-axis)

#         # Create a PinholeCameraIntrinsic object
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)        

#         pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image2,intrinsic)
        
#         option = o3d.pipelines.odometry.OdometryOption()
#         odo_init = np.identity(4)

#         [success_color_term, trans_color_term,
#         info] = o3d.pipelines.odometry.compute_rgbd_odometry(
#             rgbd_image1, rgbd_image2, intrinsic, odo_init,
#             o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
#         [success_hybrid_term, trans_hybrid_term,
#         info] = o3d.pipelines.odometry.compute_rgbd_odometry(
#             rgbd_image1, rgbd_image2, intrinsic, odo_init,
#             o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        
#         if success_color_term:
#             print("Using RGB-D Odometry")
#             print(trans_color_term)
#             source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
#                 rgbd_image1, intrinsic)
#             source_pcd_color_term.transform(trans_color_term)
#             o3d.visualization.draw_geometries([pcd2, source_pcd_color_term],
#                                             zoom=0.48,
#                                             front=[0.0999, -0.1787, -0.9788],
#                                             lookat=[0.0345, -0.0937, 1.8033],
#                                             up=[-0.0067, -0.9838, 0.1790])
#         if success_hybrid_term:
#             print("Using Hybrid RGB-D Odometry")
#             print(trans_hybrid_term)
#             source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
#                 rgbd_image1, intrinsic)
#             source_pcd_hybrid_term.transform(trans_hybrid_term)
#             o3d.visualization.draw_geometries([pcd2, source_pcd_hybrid_term],
#                                             zoom=0.48,
#                                             front=[0.0999, -0.1787, -0.9788],
#                                             lookat=[0.0345, -0.0937, 1.8033],
#                                             up=[-0.0067, -0.9838, 0.1790])
                
        
#     #     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])     
#     #     pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.02) 
#     #     pcds.append(pcd_downsampled)
        
#     #     merged_pcd += pcd
#     #     # pcds.append(pcd)
#     #     print("run",num)
    
#     # # pcd_downsampled = merged_pcd.voxel_down_sample(voxel_size=0.02) 
#     # o3d.visualization.draw_geometries(pcds)
    
#     # merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=9)
#     # o3d.visualization.draw_geometries([mesh])
    
#     # output_filename = output_dir+"mesh_3.ply"
#     # o3d.io.write_triangle_mesh(output_filename,mesh)
#     # print("image mesh 3 created")



        

 
    
    


# import open3d as o3d
# import numpy as np
# import os
# import sys
# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     pcds = []
#     merged_pcd = o3d.geometry.PointCloud()
#     input_dir= 'Test Images/'
#     colour_dir= 'images/'
#     depth_dir= 'depth/'
#     segmented_dir= 'segmented/'
#     output_dir='Mesh Tests/'
#     for i in range(0,2):
#         num= str(i)
#         # col= input_dir+segmented_dir+'segmented_'+num+'.png' 
#         col= input_dir+colour_dir+'image_'+num+'.png' 
#         dep= input_dir+depth_dir+'depth_'+num+'.png' 
        
#         # col= input_dir+'000000_color.png' 
#         # dep= input_dir+'000000_aligned_depth.png' 
#         color_raw = o3d.io.read_image(col)
#         depth_raw = o3d.io.read_image(dep)

#         # # Convert the depth image to a numpy array
#         # depth_data = np.asarray(depth_raw)
#         # # Multiply each value in the depth image by 1000
#         # print(depth_data)
#         # # Create a new Open3D Image from the modified depth data
#         # modified_depth_raw = o3d.geometry.Image(depth_data)

#         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
#         # plt.subplot(1, 2, 1)
#         # plt.title('Outdoor image')
#         # plt.imshow(rgbd_image.color)
#         # plt.subplot(1, 2, 2)
#         # plt.title('Outdoor depth image')
#         # plt.imshow(rgbd_image.depth)
#         # plt.show()
#         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        
#         # # Define your custom camera intrinsic parameters
#         # width = 1280  # Image width
#         # height = 720  # Image height
#         # fx = 640.0  # Focal length in pixels (x-axis)
#         # fy = 640.0  # Focal length in pixels (y-axis)
#         # cx = 640  # Principal point (x-axis)
#         # cy = 360  # Principal point (y-axis)

#         # # Create a PinholeCameraIntrinsic object
#         # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
#         # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,intrinsic)
#         pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])     
#         pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.02) 
#         pcds.append(pcd_downsampled)
        
#         # o3d.visualization.draw_geometries([pcd_downsampled])
#         # o3d.visualization.draw_geometries([pcd], zoom=0.5) or 0.35
        
#         merged_pcd += pcd
#         # pcds.append(pcd)
#         print("run",num)
    
#     # pcd_downsampled = merged_pcd.voxel_down_sample(voxel_size=0.02) 
#     # o3d.visualization.draw_geometries(pcds,mesh_show_back_face=True)
#     o3d.visualization.draw_geometries(pcds)
#     # merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=9)
#     # o3d.visualization.draw_geometries([mesh])
    
#     # output_filename = output_dir+"mesh_3.ply"
#     # o3d.io.write_triangle_mesh(output_filename,mesh)
#     # print("image mesh 3 created")



        

 
    
    
