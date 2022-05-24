
from cProfile import label
from pickletools import float8
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys

import math

import matplotlib.pyplot as plt


# ------------------------------

# Global Variables


X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2
I_AXIS = 3


# ------------------------------


"""
Flip over axis then move back to original position
"""
def invertPointsKeepLoc(points, axis):
    if (axis != X_AXIS and axis != Y_AXIS):
        print("Axis must be 0 (X) or 1 (Y)")
        exit()

    # Get center of points and vel loc
    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(points)
    obb = pcdSign.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    cX = centerOfPoints[X_AXIS]
    cY = centerOfPoints[Y_AXIS]

    center = np.array((cX, cY))
    
    # Flip
    points[:, axis] = points[:, axis] * -1

    points = translatePointsXY(points, center)

    return points


"""
Rotate a point counterclockwise by a given angle around a given origin.
https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
"""
def rotate(origin, point, angle):    

    radians = (angle * math.pi) / 180

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(radians) * (px - ox) - math.sin(radians) * (py - oy)
    qy = oy + math.sin(radians) * (px - ox) + math.cos(radians) * (py - oy)
    return qx, qy



def translatePointsXY(points, destination):
    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    addX = destination[X_AXIS] - point[X_AXIS]
    addY = destination[Y_AXIS] - point[Y_AXIS]

    points[:, X_AXIS] = points[:, X_AXIS] + addX
    points[:, Y_AXIS] = points[:, Y_AXIS] + addY

    return points


def rotatePoints(points, angle):
    if (angle < 0 or angle > 360):
        print("Only angles between 0 and 360 are accepable")
        exit()
    elif (not np.size(points)):
        print("Points are empty")
        exit()

    # Do nothing if asked to rotate to the same place
    if (angle == 0 or angle == 360):
        return points

    # Flip points if within diff quadrant
    if (angle > 225 and angle <= 315):
        points = invertPointsKeepLoc(points, Y_AXIS)       
    elif (angle > 135 and angle <= 225):
        points = invertPointsKeepLoc(points, X_AXIS)
        points = invertPointsKeepLoc(points, Y_AXIS)  
    elif (angle > 45 and angle <= 135):
        points = invertPointsKeepLoc(points, X_AXIS)

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    newLocation = rotate((0, 0), point, angle)

    points = translatePointsXY(points, newLocation)

    return points


def saveToBin(xyz, intensity, file):
    xyzi = np.c_[xyz, intensity]
    xyziFlat = xyzi.flatten()
    xyziFlat.tofile(file)

"""
assetIntersectsWalls
Checks if a given asset intersects anything that isnt the ground

@param asset to check intersection
@param scene, full scene will remove the road
@param semantics for the given scene
"""
def assetIntersectsWalls(asset, scene, semantics):

    # Get everything, but the ground
    maskNotGround = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    pcdArrExceptGround = scene[maskNotGround, :]

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArrExceptGround)

    voxelGridNonRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdScene, voxel_size=0.1)

    included = voxelGridNonRoad.check_if_included(o3d.utility.Vector3dVector(asset))
    return np.logical_or.reduce(included, axis=0)


def main():
    labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000001.label"
    binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000001.bin"

    pcdArr = np.fromfile(binFileName, dtype=np.float32)
    print(np.shape(pcdArr))

    print(np.shape(pcdArr)[0] // 4)

    label_arr = np.fromfile(labelsFileName, dtype=np.int32)

    print(np.shape(label_arr))

    # ------

    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    print(np.shape(pcdArr))

    intensityExtract = pcdArr[:, 3]

    pcdArr = np.delete(pcdArr, 3, 1)
    print(np.shape(pcdArr))

    # ------

    # lowerAnd = np.full(np.shape(label_arr), 65535)
    # (semantics = np.bitwise_and(label_arr, 65535)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 


    # Specific car
    maskCar = (labelInstance != 212)
    pcdArrNoCar = pcdArr[maskCar, :]
    inNoCar = intensityExtract[maskCar]
    
    maskCar = (labelInstance == 212)
    pcdArrOnlyCar = pcdArr[maskCar, :]
    inOnlyCar = intensityExtract[maskCar]


    maskRoad = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
    maskNotRoad = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)


    pcdArrOnlyRoad = pcdArr[maskRoad, :]

    pcdArrExceptRoad = pcdArr[maskNotRoad, :]


    pcdRoad = o3d.geometry.PointCloud()
    pcdRoad.points = o3d.utility.Vector3dVector(pcdArrOnlyRoad)
    pcdNonRoad = o3d.geometry.PointCloud()
    pcdNonRoad.points = o3d.utility.Vector3dVector(pcdArrExceptRoad)

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

    pcdCar = o3d.geometry.PointCloud()
    pcdCar.points = o3d.utility.Vector3dVector(pcdArrOnlyCar)

    obb = pcdCar.get_oriented_bounding_box()
    obb.color = (0, 1, 0)

    # dis = [obb, pcdCar, pcdRoad]
    dis = [obb, pcdScene]
    # dis = []
    xyzs = np.array([[0, 0, 0]])

    for angle in range(1, 12):
        points = rotatePoints(np.copy(pcdArrOnlyCar), angle * 30)

        if (not assetIntersectsWalls(points, pcdArr, semantics)):

            pcdSign2 = o3d.geometry.PointCloud()
            pcdSign2.points = o3d.utility.Vector3dVector(points)

            dis.append(pcdSign2)

        pnt = rotate((0, 0), (4.763, 7.04), angle * 30)

        xyz = np.array([pnt[0], pnt[1], 0])
        print(angle * 30)

        xyzs = np.vstack((xyzs, [xyz]))

    
    # o3d.visualization.draw_geometries([pcdSign, obb, pcdRoad])
    # o3d.visualization.draw_geometries(dis)

    pcdXyzs = o3d.geometry.PointCloud()
    pcdXyzs.points = o3d.utility.Vector3dVector(xyzs)
    dis.append(pcdXyzs)
    # o3d.visualization.draw_geometries([pcdNonRoad])
    o3d.visualization.draw_geometries(dis)

if __name__ == '__main__':
    main()



