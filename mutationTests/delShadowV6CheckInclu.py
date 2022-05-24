
"""
Note this was to see if I could use voxel inclusion 
rather than the scene ray casting 

Issue is a voxel grid from a mesh is hollow
"""

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

intensityExtract = pcd_arr[:, 3]

pcd_arr = np.delete(pcd_arr, 3, 1)
print(np.shape(pcd_arr))

# ------

# lowerAnd = np.full(np.shape(label_arr), 65535)
# semantics = np.bitwise_and(label_arr, 65535)
semantics = label_arr & 0xFFFF
labelInstance = label_arr >> 16 
# print(semantics)
# print(labelInstance)


# Specific car
# maskCar = (labelInstance != 212)
# pcdArrNoCar = pcd_arr[maskCar, :]
# inNoCar = intensityExtract[maskCar]
pcdArrNoCar = pcd_arr
inNoCar = intensityExtract

maskCar = (labelInstance == 212)
pcdArrOnlyCar = pcd_arr[maskCar, :]
inOnlyCar = intensityExtract[maskCar]

# print(pcdArrOnlyCar)

# Flip Y
# pcdArrOnlyCar[:, 1] = pcdArrOnlyCar[:, 1] * -1
pcdArrOnlyCar[:, 0] = pcdArrOnlyCar[:, 0] * -1

# print(pcdArrOnlyCar)

# pcd_arr = np.append(pcdArrNoCar, pcdArrOnlyCar, axis=0)

# print(label_arr)


pcdNoCar = o3d.geometry.PointCloud()
pcdNoCar.points = o3d.utility.Vector3dVector(pcdArrNoCar)


pcdCar = o3d.geometry.PointCloud()
pcdCar.points = o3d.utility.Vector3dVector(pcdArrOnlyCar)

#  Computing of convex point cloud 
hull, _ = pcdCar.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((0, 0, 1)) # The color of the convex hull 
# o3d.visualization.draw_geometries([pcdCar, hull_ls])

center = hull.get_center()


velCam = np.array([0, 0, 0.2])
pointsLineSet = np.array([velCam])

ba = velCam - center
baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
ba2 = ba / baLen

pt2 = velCam + ((-50) * ba2)

# pointsLineSet.append(velCam)
# print(pointsLineSet)

# hull.scale(1.1, hull.get_center())
hv = np.asarray(hull.vertices)
hv = np.copy(hv)
print(hv[0])
# hull2 = hull.translate([pt2[0], pt2[1], 0])
# print(hv[0])

# hull2.scale(3, hull2.get_center())
# hull2 = hull2.translate([pt2[0], pt2[1], 0])
# hv2 = np.asarray(hull2.vertices)
# combinedVertices = np.vstack((hv, hv2))


# https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
velCam = np.array([0, 0, -0.5])
# pointsLineSet = np.array([velCam])
pointsLineSet = np.array([])
for pt1 in hv:

    ba = velCam - pt1
    baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
    ba2 = ba / baLen

    pt2 = velCam + ((-70) * ba2)

    if (np.size(pointsLineSet)):
        pointsLineSet = np.vstack((pointsLineSet, [pt2]))
    else:
        pointsLineSet = np.array([pt2])


# print(pointsLineSet)


cutPoints = o3d.geometry.PointCloud()
cutPoints.points = o3d.utility.Vector3dVector(pointsLineSet)
hull2, _ = cutPoints.compute_convex_hull()

hull2.scale(0.5, hull2.get_center())
hv2 = np.asarray(hull2.vertices)

combinedVertices = np.vstack((hv, hv2))

# print(combinedVertices)
print(np.shape(combinedVertices))


# cutPoints = o3d.geometry.PointCloud()
cutPoints.points = o3d.utility.Vector3dVector(combinedVertices)
cutPointsHull, _ = cutPoints.compute_convex_hull()
hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(cutPointsHull)
hull_ls2.paint_uniform_color((0, 1, 1))

# o3d.visualization.draw_geometries([pcdNoCar, pcdCar, hull_ls2])

# https://stackoverflow.com/questions/35179000/filling-enclosed-space-air-inside-voxel-models-any-fast-algorithms
# fix for hollow voxel
print("make grid")
voxelGridShadow = o3d.geometry.VoxelGrid.create_from_triangle_mesh(cutPointsHull, voxel_size=0.3)
print("check inclusion")
included = voxelGridShadow.check_if_included(o3d.utility.Vector3dVector(pcdArrNoCar))
print("done with vox")

print(np.shape(pcdArrNoCar))
print(np.shape(included))
# print(included)

included = np.logical_not(included)

# print(mask2)
# print(np.shape(pcdArrNoCar))
pcdArrNoCar2 = pcdArrNoCar[included]
inNoCar2 = inNoCar[included]
print(np.shape(pcdArrNoCar2))



pcdNoCar2 = o3d.geometry.PointCloud()
pcdNoCar2.points = o3d.utility.Vector3dVector(pcdArrNoCar2)

o3d.visualization.draw_geometries([pcdNoCar2, pcdCar, hull_ls2])



# toSave = np.vstack((pcdArrNoCar2, pcdArrOnlyCar))
# inToSave = np.append(inNoCar2, inOnlyCar)

# print("here")
# print(np.shape(toSave))
# print(np.shape(inToSave))

# toSave = np.c_[toSave, inToSave]
# print(np.shape(toSave))
# print(toSave) 
# pcd_arr1 = toSave.flatten()
# print(np.shape(pcd_arr1))
# pcd_arr1.tofile("test2.bin")

