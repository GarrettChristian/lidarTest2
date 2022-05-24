
from cProfile import label
from pickletools import float8
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys

import math


import matplotlib.pyplot as plt


"""
Rotate a point counterclockwise by a given angle around a given origin.
https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python

The angle should be given in radians.
"""
def rotate(origin, point, angle):    
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rotatePoints(points, angle):
    if (angle == 0 or angle == 360):
        return points
    elif (angle < 0 or angle > 360):
        print("Only angles between 0 and 360 are accepable")
        exit()
    elif (np.shape(points)[0] < 1):
        print("Points are empty")
        exit()

    # Flip points if within diff quadrant
    if (angle > 45 and angle <= 135):
        points[:, 1] = points[:, 1] * -1
    elif (angle > 135 and angle <= 225):
        points[:, 0] = points[:, 0] * -1
    elif (angle > 135 and angle <= 225):
        points[:, 0] = points[:, 0] * -1
        points[:, 1] = points[:, 1] * -1

    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(points)
    obb = pcdSign.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    translate = rotate(point, (0, 0), angle)

    addX = translate[0] - point[0]
    addY = translate[1] - point[1]

    points[:, 0] = points[:, 0] + addX
    points[:, 1] = points[:, 1] + addY

    return points

def main():
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
    centerOfSign = obb.get_center()

    print(centerOfSign)
    x1 = centerOfSign[0]
    y1 = centerOfSign[0]

    rotateRes = rotate((x1, y1), (0, 0), 90)
    print(rotateRes)

    dis = [obb, pcdSign]

    for angle in range(1,2):
        points = rotatePoints(oneSign, angle * 30)

        pcdSign2 = o3d.geometry.PointCloud()
        pcdSign2.points = o3d.utility.Vector3dVector(points)

        dis.append(pcdSign2)

    # o3d.visualization.draw_geometries([pcdSign, obb, pcdRoad])
    o3d.visualization.draw_geometries(dis)



if __name__ == '__main__':
    main()



