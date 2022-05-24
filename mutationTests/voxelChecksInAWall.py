
from pickletools import float8
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys

import math




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

maskCar = (labelInstance == 212)
pcdArrOnlyCar = pcd_arr[maskCar, :]
inOnlyCar = intensityExtract[maskCar]

# Flip Y
# pcdArrOnlyCar[:, 1] = pcdArrOnlyCar[:, 1] * -1
# Flip X
pcdArrOnlyCar[:, 0] = pcdArrOnlyCar[:, 0] * -1

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(basePcd)

pcdCar = o3d.geometry.PointCloud()
pcdCar.points = o3d.utility.Vector3dVector(pcdArrOnlyCar)


voxelGridScene = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
voxelGridCar = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdCar, voxel_size=0.1)

included = voxelGridScene.check_if_included(o3d.utility.Vector3dVector(pcdArrOnlyCar))
# print(included)

modCar = pcdArrOnlyCar[included]

pcdCar2 = o3d.geometry.PointCloud()
pcdCar2.points = o3d.utility.Vector3dVector(modCar)






view = [voxelGridCar, voxelGridScene]
# o3d.visualization.draw_geometries(view)

# o3d.visualization.draw_geometries([pcdCar])

# o3d.visualization.draw_geometries([pcdCar2])




maskRoad = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
maskNotRoad = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)


pcdArrOnlyRoad = pcd_arr[maskRoad, :]

pcdArrExceptRoad = pcd_arr[maskNotRoad, :]


pcdRoad = o3d.geometry.PointCloud()
pcdRoad.points = o3d.utility.Vector3dVector(pcdArrOnlyRoad)
pcdNonRoad = o3d.geometry.PointCloud()
pcdNonRoad.points = o3d.utility.Vector3dVector(pcdArrExceptRoad)
# o3d.visualization.draw_geometries([pcdRoad])


voxelGridRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdRoad, voxel_size=0.1)
voxelGridNonRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdNonRoad, voxel_size=0.1)


included = voxelGridNonRoad.check_if_included(o3d.utility.Vector3dVector(pcdArrOnlyCar))


inAWall = np.logical_or.reduce(included, axis=0)
print(inAWall)




