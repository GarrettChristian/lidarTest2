
from cProfile import label
from pickletools import float8
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys

import math


import matplotlib.pyplot as plt




labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000001.label"
binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000001.bin"

pcd_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(pcd_arr))

print(np.shape(pcd_arr)[0] // 4)

label_arr = np.fromfile(labelsFileName, dtype=np.int32)

print(np.shape(label_arr))



# ------

pcd_arr = pcd_arr.reshape((int(np.shape(pcd_arr)[0]) // 4, 4))
print(np.shape(pcd_arr))

intensityExtract = pcd_arr[:, 3]

pcd_arr = np.delete(pcd_arr, 3, 1)
print(np.shape(pcd_arr))

# ------

# lowerAnd = np.full(np.shape(label_arr), 65535)
# (semantics = np.bitwise_and(label_arr, 65535)
semantics = label_arr & 0xFFFF
labelInstance = label_arr >> 16 


# Specific car
basePcd = pcd_arr
baseIntCar = intensityExtract



maskRoad = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
maskNotRoad = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)


pcdArrOnlyRoad = pcd_arr[maskRoad, :]

pcdArrExceptRoad = pcd_arr[maskNotRoad, :]


pcdRoad = o3d.geometry.PointCloud()
pcdRoad.points = o3d.utility.Vector3dVector(pcdArrOnlyRoad)
pcdNonRoad = o3d.geometry.PointCloud()
pcdNonRoad.points = o3d.utility.Vector3dVector(pcdArrExceptRoad)
# o3d.visualization.draw_geometries([pcdRoad])


# voxelGridRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdRoad, voxel_size=0.1)
# voxelGridNonRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdNonRoad, voxel_size=0.1)


# included = voxelGridNonRoad.check_if_included(o3d.utility.Vector3dVector(pcdArrOnlyCar))

# inAWall = np.logical_or.reduce(included, axis=0)
# print(inAWall)




maskSign = (semantics == 80) | (semantics == 81)

pcdArrOnlySigns = pcd_arr[maskSign, :]

pcdSigns = o3d.geometry.PointCloud()
pcdSigns.points = o3d.utility.Vector3dVector(pcdArrOnlySigns)
# o3d.visualization.draw_geometries([pcdSigns])


labels = np.array(pcdSigns.cluster_dbscan(eps=2, min_points=10, print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcdSigns.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcdSigns])

oneSign = pcdArrOnlySigns[labels == 2, :]

pcdSign = o3d.geometry.PointCloud()
pcdSign.points = o3d.utility.Vector3dVector(oneSign)


obb = pcdSign.get_oriented_bounding_box()
obb.color = (0, 1, 0)

o3d.visualization.draw_geometries([pcdSign, obb, pcdRoad])



