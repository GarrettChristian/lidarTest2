
from pickletools import float8
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys

import math




labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000000.label"
binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000000.bin"

pcd_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(pcd_arr))

print(np.shape(pcd_arr)[0] // 4)

label_arr = np.fromfile(labelsFileName, dtype=np.int32)

print(np.shape(label_arr))



# ------

pcd_arr = pcd_arr.reshape((int(np.shape(pcd_arr)[0]) // 4, 4))
print(np.shape(pcd_arr))
pcd_arr = np.delete(pcd_arr, 3, 1)
print(np.shape(pcd_arr))

# ------

toSave = pcd_arr

print(np.shape(toSave))

iVals = np.ones((np.shape(toSave)[0],), dtype=np.float32)
print(np.shape(toSave))
toSave = np.c_[toSave, iVals]
print(np.shape(toSave))
print(toSave) 
pcd_arr1 = toSave.flatten()
print(np.shape(pcd_arr1))
pcd_arr1.tofile("testInitinsityAll1.bin")



# Tried to use the points to crop didn't work
# vol = o3d.visualization.SelectionPolygonVolume()

# vol.orthogonal_axis = "Z"
# vol.axis_max = np.max(pointsLineSet[:, 2])
# vol.axis_min = np.min(pointsLineSet[:, 2])
# # Convert the np.array to a Vector3dVector
# vol.bounding_polygon = o3d.utility.Vector3dVector(pointsLineSet)

# cropped_pcd = vol.crop_point_cloud(pcdNoCar)
# bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
# bounding_box.color = (0, 1, 0)


# o3d.visualization.draw_geometries([cropped_pcd, pcdCar, bounding_box])








# cutPoints = o3d.geometry.PointCloud()
# cutPoints.points = o3d.utility.Vector3dVector(pointsLineSet)
# hull2, _ = cutPoints.compute_convex_hull()
# hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
# hull_ls2.paint_uniform_color((0, 1, 1)) # The color of the convex hull

# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(pointsLineSet)
# line_set.lines = o3d.utility.Vector2iVector(lines)


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pcd_arr)

# o3d.visualization.draw_geometries([pcdNoCar, hull_ls, hull_ls2])

# o3d.visualization.draw_geometries([pcd])


# o3d.visualization.draw_geometries([cropped_pcd, bounding_box])




