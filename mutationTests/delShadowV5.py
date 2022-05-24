
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
print(semantics)
print(labelInstance)


# Specific car
# maskCar = (labelInstance != 212)
# pcd_arrNoCar = pcd_arr[maskCar, :]
# inNoCar = intensityExtract[maskCar]
pcd_arrNoCar = pcd_arr
inNoCar = intensityExtract

maskCar = (labelInstance == 212)
pcd_arrOnlyCar = pcd_arr[maskCar, :]
inOnlyCar = intensityExtract[maskCar]

print(pcd_arrOnlyCar)

# Flip Y
# pcd_arrOnlyCar[:, 1] = pcd_arrOnlyCar[:, 1] * -1
pcd_arrOnlyCar[:, 0] = pcd_arrOnlyCar[:, 0] * -1

print(pcd_arrOnlyCar)

# pcd_arr = np.append(pcd_arrNoCar, pcd_arrOnlyCar, axis=0)

print(label_arr)


pcdNoCar = o3d.geometry.PointCloud()
pcdNoCar.points = o3d.utility.Vector3dVector(pcd_arrNoCar)


pcdCar = o3d.geometry.PointCloud()
pcdCar.points = o3d.utility.Vector3dVector(pcd_arrOnlyCar)

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


print(pointsLineSet)


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


legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(cutPointsHull)


mask2 = np.ones((np.shape(pcd_arrNoCar)[0],), dtype=int)

scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(legacyMesh)
for idx in range(np.shape(pcd_arrNoCar)[0]):
    pt = pcd_arrNoCar[idx]
    query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

    occupancy = scene.compute_occupancy(query_point)
    if occupancy == 1:
        mask2[idx] = 0


# print(mask2)
# print(np.shape(pcd_arrNoCar))
pcd_arrNoCar2 = pcd_arrNoCar[mask2 == 1, :]
inNoCar2 = inNoCar[mask2 == 1]
# print(np.shape(pcd_arrNoCar2))



pcdNoCar2 = o3d.geometry.PointCloud()
pcdNoCar2.points = o3d.utility.Vector3dVector(pcd_arrNoCar2)

# hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
# hull_ls2.paint_uniform_color((0, 1, 1))

o3d.visualization.draw_geometries([pcdNoCar2, pcdCar, hull_ls2])



toSave = np.vstack((pcd_arrNoCar2, pcd_arrOnlyCar))
inToSave = np.append(inNoCar2, inOnlyCar)

print("here")
print(np.shape(toSave))
print(np.shape(inToSave))

toSave = np.c_[toSave, inToSave]
print(np.shape(toSave))
print(toSave) 
pcd_arr1 = toSave.flatten()
print(np.shape(pcd_arr1))
pcd_arr1.tofile("test2.bin")

