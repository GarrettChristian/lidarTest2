
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

ininsityExtract = pcd_arr[:, 3]

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
# inNoCar = ininsityExtract[maskCar]
pcd_arrNoCar = pcd_arr
inNoCar = ininsityExtract

maskCar = (labelInstance == 212)
pcd_arrOnlyCar = pcd_arr[maskCar, :]
inOnlyCar = ininsityExtract[maskCar]

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


print(np.shape(pcd_arrOnlyCar))

print(hull_ls)
# print(np.asarray(hull_ls.points))

hull_pts = np.asarray(hull_ls.points)



# https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
velCam = np.array([0, 0, 0.2])
pointsLineSet = np.array([velCam])
for pt1 in hull_pts:

    ba = velCam - pt1
    baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
    ba2 = ba / baLen

    pt2 = velCam + ((-50) * ba2)

    pointsLineSet = np.vstack((pointsLineSet, [pt2]))

# pointsLineSet.append(velCam)
# print(pointsLineSet)


cutPoints = o3d.geometry.PointCloud()
cutPoints.points = o3d.utility.Vector3dVector(pointsLineSet)
hull2, _ = cutPoints.compute_convex_hull()
legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(hull2)

# print(hull2.has_vertices())
# print(hull2.has_triangles())

mask2 = np.ones((np.shape(pcd_arrNoCar)[0],), dtype=int)

scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(legacyMesh)
for idx in range(np.shape(pcd_arrNoCar)[0]):
    pt = pcd_arrNoCar[idx]
    query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

    occupancy = scene.compute_occupancy(query_point)
    if occupancy == 1:
        mask2[idx] = 0


print(mask2)
print(np.shape(pcd_arrNoCar))
pcd_arrNoCar2 = pcd_arrNoCar[mask2 == 1, :]
inNoCar2 = inNoCar[mask2 == 1]
print(np.shape(pcd_arrNoCar2))



# pcdNoCar2 = o3d.geometry.PointCloud()
# pcdNoCar2.points = o3d.utility.Vector3dVector(pcd_arrNoCar2)


# hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
# hull_ls2.paint_uniform_color((0, 1, 1))

# o3d.visualization.draw_geometries([pcdNoCar2, pcdCar, hull_ls2])



toSave = np.vstack((pcd_arrNoCar2, pcd_arrOnlyCar))
inToSave = np.append(inNoCar2, inOnlyCar)

print("here")
print(np.shape(toSave))
print(np.shape(inToSave))

# iVals = np.ones((np.shape(toSave)[0],), dtype=np.float32)
# print(np.shape(toSave))
# toSave = np.c_[toSave, iVals]
toSave = np.c_[toSave, inToSave]
print(np.shape(toSave))
print(toSave) 
pcd_arr1 = toSave.flatten()
print(np.shape(pcd_arr1))
pcd_arr1.tofile("test2.bin")



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




