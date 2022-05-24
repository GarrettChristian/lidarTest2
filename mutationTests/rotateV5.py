
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
Flip over axis
"""
def mirrorPoints(points, axis):
    if (axis != X_AXIS and axis != Y_AXIS):
        print("Axis must be 0 (X) or 1 (Y)")
        exit()
    
    # Flip
    points[:, axis] = points[:, axis] * -1

    return points



"""
Translate a group of points to a new location based on the center
"""
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


"""
Rotate a point counterclockwise by a given angle around a given origin.
https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
"""
def rotateOnePoint(origin, point, angle):    

    radians = (angle * math.pi) / 180

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(radians) * (px - ox) - math.sin(radians) * (py - oy)
    qy = oy + math.sin(radians) * (px - ox) + math.cos(radians) * (py - oy)
    return qx, qy



def rotatePointsLocalized(points, angle):
    # Preconditions for asset rotation
    if (angle < 0 or angle > 360):
        print("Only angles between 0 and 360 are accepable")
        exit()
    elif (not np.size(points)):
        print("Points are empty")
        exit()

    # Do nothing if asked to rotate to the same place
    if (angle == 0 or angle == 360):
        return points

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    center = (centerOfPoints[X_AXIS], centerOfPoints[Y_AXIS])

    for point in points:
        pointXY = (point[X_AXIS], point[Y_AXIS])
        newX, newY = rotateOnePoint(center, pointXY, angle)
        point[X_AXIS] = newX
        point[Y_AXIS] = newY
    
    return points



def rotatePoints(points, angle):
    # Preconditions for asset rotation
    if (angle < 0 or angle > 360):
        print("Only angles between 0 and 360 are accepable")
        exit()
    elif (not np.size(points)):
        print("Points are empty")
        exit()

    # Do nothing if asked to rotate to the same place
    if (angle == 0 or angle == 360):
        return points

    # Rotate the points relative to their center 
    points = rotatePointsLocalized(points, angle)

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    newLocation = rotateOnePoint((0, 0), point, angle)

    points = translatePointsXY(points, newLocation)

    return points



def saveToBin(xyz, intensity, file):
    xyzi = np.c_[xyz, intensity]
    xyziFlat = xyzi.flatten()
    xyziFlat.tofile(file)


"""
Creates a mask the size of the points array
True is included in the mesh
False is not included in the mesh
"""
def checkInclusionBasedOnTriangleMesh(points, mesh):

    obb = mesh.get_oriented_bounding_box()
    print(obb)
    print(obb.get_max_bound())
    print(obb.get_min_bound())

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    mask = np.zeros((np.shape(points)[0],), dtype=bool)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacyMesh)

    # pcdAsset = o3d.geometry.PointCloud()
    pointsVector = o3d.utility.Vector3dVector(points)

    indexesWithinBox = obb.get_point_indices_within_bounding_box(pointsVector)

    for idx in indexesWithinBox:
        pt = points[idx]
        query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

        occupancy = scene.compute_occupancy(query_point)
        mask[idx] = (occupancy == 1)

    return mask


"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon

https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
"""
def removeLidarShadow(asset, scene, intensity, semantics):

    # Prepare asset and scene point clouds
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(scene)

    #  Get the asset's hull mesh
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    
    centerCamPoint = np.array([0, 0, -0.5])
    castHullPoints = np.array([])
    for point1 in hullVertices:

        ba = centerCamPoint - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen

        pt2 = centerCamPoint + ((-100) * ba2)

        if (np.size(castHullPoints)):
            castHullPoints = np.vstack((castHullPoints, [pt2]))
        else:
            castHullPoints = np.array([pt2])

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(castHullPoints)
    hull2, _ = pcdCastHull.compute_convex_hull()

    hull2.scale(0.5, hull2.get_center())
    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdCut = o3d.geometry.PointCloud()
    pcdCut.points = o3d.utility.Vector3dVector(combinedVertices)
    cutPointsHull, _ = pcdCut.compute_convex_hull()

    mask = checkInclusionBasedOnTriangleMesh(scene, cutPointsHull)
    mask = np.logical_not(mask)

    scene = scene[mask, :]
    intensity = intensity[mask]
    semantics = semantics[mask]

    return (scene, intensity, semantics)


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



"""
Will take in the intensity for a set of points 
and alter the intensity by a factor
"""
def alterAssetIntensity(intensity, change):
    
    intensity = intensity * change

    for idx in range(np.shape(intensity)[0]):
        if (intensity[idx] > 1):
            intensity[idx] = 1
        elif (intensity[idx] < 0):
            intensity[idx] = 0

    return intensity


"""
Checks that all points exist above the ground
"""
def pointsAboveGround(points, scene, semantics):
    
    maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
    pcdArrGround = scene[maskGround, :]

    # Remove Z dim
    pcdArrGround[:, Z_AXIS] = 0
    pointsCopy = np.copy(points)
    pointsCopy[:, Z_AXIS] = 0

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArrGround)

    voxelGridGround = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdScene, voxel_size=1)

    included = voxelGridGround.check_if_included(o3d.utility.Vector3dVector(pointsCopy))
    return np.logical_and.reduce(included, axis=0)


"""
Main Method
"""
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

    intensity = pcdArr[:, 3]

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
    inNoCar = intensity[maskCar]
    
    maskCar = (labelInstance == 212)
    pcdArrOnlyCar = pcdArr[maskCar, :]
    inOnlyCar = intensity[maskCar]


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
    display = [obb]

    for angle in range(1, 12):
        points = rotatePoints(np.copy(pcdArrOnlyCar), angle * 30)

        if (not assetIntersectsWalls(points, pcdArr, semantics)):
            
            if (pointsAboveGround(points, pcdArr, semantics)):

                pcdAsset = o3d.geometry.PointCloud()
                pcdAsset.points = o3d.utility.Vector3dVector(points)

                display.append(pcdAsset)

                print("here")
                print(np.shape(pcdArr))
                print(np.shape(intensity))

                pcdArr, intensity, semantics = removeLidarShadow(points, pcdArr, intensity, semantics)
    


    # o3d.visualization.draw_geometries([pcdSign, obb, pcdRoad])
    # o3d.visualization.draw_geometries(dis)

    # pcdXyzs = o3d.geometry.PointCloud()
    # pcdXyzs.points = o3d.utility.Vector3dVector(xyzs)
    # dis.append(pcdXyzs)
    # o3d.visualization.draw_geometries([pcdNonRoad])

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

    display.append(pcdScene)

    o3d.visualization.draw_geometries(display)


    # saveToBin()



if __name__ == '__main__':
    main()



