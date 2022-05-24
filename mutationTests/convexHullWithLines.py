
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

# lowerAnd = np.full(np.shape(label_arr), 65535)
# semantics = np.bitwise_and(label_arr, 65535)
semantics = label_arr & 0xFFFF
labelInstance = label_arr >> 16 
print(semantics)
print(labelInstance)


# Specific car
maskCar = (labelInstance != 212)
pcd_arrNoCar = pcd_arr[maskCar, :]

maskCar = (labelInstance == 212)
pcd_arrOnlyCar = pcd_arr[maskCar, :]

print(pcd_arrOnlyCar)

# Flip Y
# pcd_arrOnlyCar[:, 1] = pcd_arrOnlyCar[:, 1] * -1
pcd_arrOnlyCar[:, 0] = pcd_arrOnlyCar[:, 0] * -1

print(pcd_arrOnlyCar)

pcd_arr = np.append(pcd_arrNoCar, pcd_arrOnlyCar, axis=0)

print(label_arr)


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

pointsLineSet = []

# https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
velCam = np.array([0, 0, 0.2])
for pt1 in hull_pts:

    print(pt1)
    ba = velCam - pt1
    baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
    ba2 = ba / baLen

    pt2 = velCam + ((-50) * ba2)

    pointsLineSet.append(pt2)

pointsLineSet.append(velCam)

cutPoints = o3d.geometry.PointCloud()
cutPoints.points = o3d.utility.Vector3dVector(pointsLineSet)
hull2, _ = cutPoints.compute_convex_hull()
hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
hull_ls2.paint_uniform_color((0, 1, 1)) # The color of the convex hull

# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(pointsLineSet)
# line_set.lines = o3d.utility.Vector2iVector(lines)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_arr)

o3d.visualization.draw_geometries([pcd, hull_ls, hull_ls2])

# o3d.visualization.draw_geometries([pcd])




# o3d.visualization.draw_geometries([cropped_pcd, bounding_box])




