"""
"""

import numpy as np

import glob, os
import struct
import open3d as o3d


labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000000.label"
binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000000.bin"

pcd_arr = np.fromfile(binFileName, dtype=np.float32)
print(np.shape(pcd_arr))

print(np.shape(pcd_arr)[0] // 4)

pcd_arr = pcd_arr.reshape((int(np.shape(pcd_arr)[0]) // 4, 4))

# Sort on y
pcd_arr = pcd_arr[pcd_arr[:, 0].argsort()]


pcd_arr1 = pcd_arr.flatten()
print(np.shape(pcd_arr1))
pcd_arr1.tofile("test2.bin")