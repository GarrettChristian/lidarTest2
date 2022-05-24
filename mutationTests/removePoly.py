
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
            [ 2.6509309513852526, -250, 1.6834473132326844 ],
                                ...
            [ 2.6579576128816544, -250, 1.6819127849749496 ]]).astype("float64")

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


# br = [0, 0, 0]
# bl = [0, 0, 0]
# tr = [0, 0, 0]
# tl = [0, 0, 0]
# for xyz in pcd_arrOnlyCar:
#     if xyz[0] > br

print(np.shape(pcd_arrOnlyCar))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_arr)

# o3d.visualization.draw_geometries([pcd])

bounding_polygon = np.array([ 
            [0, 0, 20],
            [ -50000, -50000, -50000], # bottom right
            [ -50000, -50000, 50000], 
            [ -50000, 50000, -50000], # top right
            [ -50000, 50000, 50000],
            [ 50000, 50000, -50000], # top left
            [ 50000, 50000, 50000],
            [ 50000, -50000, -50000], # bottom left
            [ 50000, -50000, 50000], 
            ]).astype("float64")


vol = o3d.visualization.SelectionPolygonVolume()

vol.orthogonal_axis = "Z"
vol.axis_max = np.max(bounding_polygon[:, 2])
vol.axis_min = np.min(bounding_polygon[:, 2])
# Convert the np.array to a Vector3dVector
vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

cropped_pcd = vol.crop_point_cloud(pcd)
bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
bounding_box.color = (0, 1, 0)


# o3d.visualization.draw_geometries([cropped_pcd, bounding_box])




