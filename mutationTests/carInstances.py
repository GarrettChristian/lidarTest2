
import numpy as np

import glob, os
import struct
import open3d as o3d


"""
print("Load a polygon volume and use it to crop the original point cloud")
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

https://stackoverflow.com/questions/57155972/how-to-create-a-open3d-visualization-selectionpolygonvolume-object-without-loadi
bounding_polygon = np.array([ 
            [ 2.6509309513852526, 0.0, 1.6834473132326844 ],
                                ...
            [ 2.6579576128816544, 0.0, 1.6819127849749496 ]]).astype("float64")

vol = o3d.visualization.SelectionPolygonVolume()

vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

"""


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

# Only cars
mask = (semantics == 10)
car_intances = labelInstance[mask]

carz = set()
for lbl in car_intances:
    if (lbl not in carz and lbl != 0):
        print(lbl)
        carz.add(lbl)

# Specific car
maskCar = (labelInstance != 212)
pcd_arr2 = pcd_arr[maskCar, :]


print(label_arr)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_arr2)
o3d.visualization.draw_geometries([pcd])




