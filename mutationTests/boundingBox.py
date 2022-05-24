
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys




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
aabb = pcdCar.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)

obb = pcdCar.get_oriented_bounding_box()
obb.color = (0, 1, 0)

print(aabb)
print(obb)

print(np.shape(pcd_arrOnlyCar))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_arr)

o3d.visualization.draw_geometries([pcd, obb, aabb])

# o3d.visualization.draw_geometries([pcd])


# o3d.visualization.draw_geometries([cropped_pcd, bounding_box])




