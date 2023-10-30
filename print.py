import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time
import datetime

if __name__ == "__main__":
    times = [0, 0, 0, 0, 0, 0]
    start_time = time.time()
    name = "Test Images/global mesh/mesh30ns.ply"



    print("Testing IO for meshes ...")
    mesh = o3d.io.read_triangle_mesh(name)
    mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])     
    print(mesh)
    
    # edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    # edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)    # vertex_manifold = mesh.is_vertex_manifold()
    # self_intersecting = mesh.is_self_intersecting()
    
    # print('filter with average with 1 iteration')
    # mesh = meshin.filter_smooth_simple(number_of_iterations=1)
    # mesh.compute_vertex_normals()
    
    # print('filter with average with 5 iterations')
    # mesh = meshin.filter_smooth_simple(number_of_iterations=5)
    # mesh.compute_vertex_normals()
    
    # print('filter with Laplacian with 10 iterations')
    # mesh = meshin.filter_smooth_laplacian(number_of_iterations=10)
    # mesh.compute_vertex_normals()

    # print('filter with Laplacian with 50 iterations')
    # mesh = meshin.filter_smooth_laplacian(number_of_iterations=50)
    # mesh.compute_vertex_normals()
    
    # print('filter with Taubin with 10 iterations')
    # mesh = meshin.filter_smooth_taubin(number_of_iterations=10)
    # mesh.compute_vertex_normals()

    # print('filter with Taubin with 100 iterations')
    # mesh = meshin.filter_smooth_taubin(number_of_iterations=100)
    # mesh.compute_vertex_normals()

    # print(mesh)
    # mesh_name = "Test Images/global mesh/mesh.ply"
    # o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)


    # times[0] = time.time() - start_time
    # # print("Testing IO for point cloud ...")
    # # pcd = o3d.io.read_point_cloud(name)
    # # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])     
    # # print(pcd)
    # # o3d.visualization.draw_geometries([pcd])

    # print("- Time    %s" % datetime.timedelta(seconds=times[0]))
    o3d.visualization.draw_geometries([mesh])


   