
from yaml import scan
import numpy as np

import glob, os
import struct
import open3d as o3d
import sys

import math


import matplotlib.pyplot as plt
import copy




# http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html
def movePointsToMesh(pointsInShadow, mesh):

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    sceneRays = o3d.t.geometry.RaycastingScene()
    sceneRays.add_triangles(legacyMesh)

    raysVectors = []
    for point in pointsInShadow:
        raysVectors.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectors, dtype=o3d.core.Dtype.Float32)

    ans = sceneRays.cast_rays(rays)

    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    print(np.shape(pointsOnMesh))

    newPosition = []
    for vector in pointsOnMesh:
        newPosition.append(vector.numpy())

    return newPosition

"""
Creates a mask the size of the points array
True is included in the mesh
False is not included in the mesh
"""
def checkInclusionBasedOnTriangleMesh(points, mesh):

    obb = mesh.get_oriented_bounding_box()

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
https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
"""
def getLidarShadowMesh(asset):

    # Prepare asset and scene point clouds
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    #  Get the asset's hull mesh
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    
    castHullPoints = np.array([])
    for point1 in hullVertices:

        ba = 0 - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen

        pt2 = 0 + ((-100) * ba2)

        if (np.size(castHullPoints)):
            castHullPoints = np.vstack((castHullPoints, [pt2]))
        else:
            castHullPoints = np.array([pt2])

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(castHullPoints)
    hull2, _ = pcdCastHull.compute_convex_hull()

    # hull2.scale(0.5, hull2.get_center())
    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdShadow = o3d.geometry.PointCloud()
    pcdShadow.points = o3d.utility.Vector3dVector(combinedVertices)
    shadowMesh, _ = pcdShadow.compute_convex_hull()

    return shadowMesh


# https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
def getAngleRadians(p0, p1, p2):
    p0x = p0[0] 
    p0y = p0[1]
    p1x = p1[0] 
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]

    result = math.atan2(p2y - p0y, p2x - p0x) - math.atan2(p1y - p0y, p1x - p0x)


    # degreesResult = result * (180 / math.pi)

    # return result
    return result



# Get closest to the two bounding points selected
def closestBoundingTwo(min, max, asset):

    # Remove Z dim
    assetCopy = np.copy(asset)
    assetCopy[:, 2] = 0

    # Find closest
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(assetCopy)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(min, 1)
    [k, idx2, _] = pcd_tree.search_knn_vector_3d(max, 1)

    return asset[idx][0], asset[idx2][0]


def createStopSign(center):
    

    box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.75, depth=0.30)
    box2 = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.30, depth=0.75)

    box.translate(center, relative=False)
    box2.translate(center, relative=False)

    signVertices = np.vstack((np.array(box.vertices), np.array(box2.vertices)))

    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(signVertices)

    #  Get the asset's hull mesh
    sign, _ = pcdSign.compute_convex_hull()

    return sign



def createCrossbuck(center):
    
    box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.22, depth=1.22)
    box2 = o3d.geometry.TriangleMesh.create_box(width=0.05, height=1.22, depth=0.22)

    box.translate(center, relative=False)
    box2.translate(center, relative=False)

    box += box2

    rotation2 = box.get_rotation_matrix_from_xyz((0.785398, 0, 0))
    box.rotate(rotation2, center=box.get_center())

    return box


def createYeild():
    
    yeildPoints = np.array([[0, -0.455, 0.91], 
                            [0, 0, 0.91], 
                            [0, 0.455, 0.91], 
                            [0, 0, 0],
                            [0.05, -0.455, 0.91], 
                            [0.05, 0, 0.91], 
                            [0.05, 0.455, 0.91], 
                            [0.05, 0, 0]])

    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(yeildPoints)

    #  Get the asset's hull mesh
    sign, _ = pcdSign.compute_convex_hull()

    return sign



"""
Main Method
"""
def main():
    labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000030.label"
    binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000030.bin"

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
    # o3d.visualization.draw_geometries([pcdRoad])


    # voxelGridRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdRoad, voxel_size=0.1)
    # voxelGridNonRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdNonRoad, voxel_size=0.1)


    # included = voxelGridNonRoad.check_if_included(o3d.utility.Vector3dVector(pcdArrOnlyCar))

    # inAWall = np.logical_or.reduce(included, axis=0)
    # print(inAWall)

    maskSign = (semantics == 81)
    maskNotSign = (semantics != 81)

    pcdArrOnlySigns = pcd_arr[maskSign, :]
    pcdArrNotSigns = pcd_arr[maskNotSign, :]

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

    oneSign = pcdArrOnlySigns[labels == 1, :]

    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(oneSign)
    pcdNotSign = o3d.geometry.PointCloud()
    pcdNotSign.points = o3d.utility.Vector3dVector(pcdArrNotSigns)
    

    abb = pcdSign.get_axis_aligned_bounding_box()
    obb = pcdSign.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    signCenter = obb.get_center()


    print(obb.extent)
    print(obb.get_min_bound())
    print(obb.get_max_bound())

    minB, maxB = closestBoundingTwo(obb.get_min_bound(), obb.get_max_bound(), oneSign)
    print("minB", minB)
    print("maxB", maxB)

    # Create shape mesh
    # box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=1, depth=1)
    # box = createStopSign(signCenter)
    box = createCrossbuck(signCenter)
    # box = createYeild() 
    boxCenter = box.get_center()
    x = signCenter[0] - boxCenter[0]
    y = signCenter[1] - boxCenter[1]
    z = signCenter[2] - boxCenter[2]
    box.translate(signCenter, relative=False)
    axisAlignBox = box.get_axis_aligned_bounding_box()
    axisAlignBox.color = (1, 0, 0)


    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    qw = math.sqrt(1 + obb.R[0][0] + obb.R[1][1] + obb.R[2][2]) /2
    qx = (obb.R[2][1] - obb.R[1][2])/ ( 4 *qw)
    qy = (obb.R[0][2] - obb.R[2][0]) / ( 4 *qw)
    qz = (obb.R[1][0] - obb.R[0][1]) / ( 4 *qw)

    roll = math.atan2(obb.R[2][1], obb.R[2][2])

    print(obb.R)
    print("box min", box.get_min_bound())
    print("box max", box.get_max_bound())

    # angleSign = getAngleRadians((0, 0), minB, maxB)
    # angleBox = getAngleRadians((0, 0), box.get_min_bound(), box.get_max_bound())
    angleSign = getAngleRadians(obb.get_center(), minB, box.get_min_bound())

    print(angleSign)
    print(angleSign * (180 / math.pi))

    pcdminmaxBoxOG = o3d.geometry.PointCloud()
    pcdminmaxBoxOG.points = o3d.utility.Vector3dVector([box.get_min_bound(), box.get_max_bound()])
    pcdminmaxBoxOG.paint_uniform_color((1, 0.1, .5))
    boxCopy = copy.deepcopy(box)
    boxCopy.paint_uniform_color((0, 0, 1))

    # rotation = box.get_rotation_matrix_from_xyz((0, 0, np.pi / 7))
    # rotation = box.get_rotation_matrix_from_xyz((0, 0, 0))
    rotation = box.get_rotation_matrix_from_xyz((0, 0, angleSign * -1))
    rotation2 = box.get_rotation_matrix_from_xyz((0, 0, angleSign))
    # rotation = box.get_rotation_matrix_from_quaternion((0, 0, 0, qz))
    print(np.shape(rotation))
    print(rotation)
    # rotation = obb.R
    # rotation = newRot

    boxRotate1 = copy.deepcopy(box)
    boxRotate2 = copy.deepcopy(box)

    boxRotate1.rotate(rotation, center=boxRotate1.get_center())
    boxRotate2.rotate(rotation2, center=boxRotate2.get_center())

    dist1 = np.linalg.norm(minB - boxRotate1.get_min_bound())
    dist2 = np.linalg.norm(minB - boxRotate2.get_min_bound())

    if (dist1 < dist2):
        box = boxRotate1
    else:
        box = boxRotate2

    shadowMesh = getLidarShadowMesh(np.array(box.vertices))
    hull_ls44 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowMesh)
    hull_ls44.paint_uniform_color((0, 0.2, 1))

    mask = checkInclusionBasedOnTriangleMesh(pcdArrNotSigns, shadowMesh)

    # pointsNearSign = np.vstack((pcdArrNotSigns[mask], oneSign))
    moved = movePointsToMesh(pcd_arr, box)
    pcdmoved = o3d.geometry.PointCloud()
    pcdmoved.points = o3d.utility.Vector3dVector(moved)

    pcdminmax = o3d.geometry.PointCloud()
    pcdminmax.points = o3d.utility.Vector3dVector([minB, maxB])
    pcdminmax.paint_uniform_color((0.75, 0.2, 0.75))
    pcdminmaxBox = o3d.geometry.PointCloud()
    pcdminmaxBox.points = o3d.utility.Vector3dVector([box.get_min_bound(), box.get_max_bound()])
    pcdminmaxBox.paint_uniform_color((0.50, 0.1, 1))

    o3d.visualization.draw_geometries([pcdSign, obb, abb, pcdRoad, pcdminmax, pcdminmaxBox, box, pcdminmaxBoxOG, boxCopy, axisAlignBox])
    # o3d.visualization.draw_geometries([obb, abb, pcdminmax, pcdminmaxBox, pcdminmaxBoxOG, pcdmoved, pcdNotSign])
    # o3d.visualization.draw_geometries([pcdNotSign, pcdSign])
    # o3d.visualization.draw_geometries([pcdmoved, pcdNotSign, obb, box, hull_ls44, pcdSign])
    # o3d.visualization.draw_geometries([pcdmoved, pcdNotSign, obb, box, hull_ls44])
    # o3d.visualization.draw_geometries([pcdmoved, pcdNotSign, obb, hull_ls44])




if __name__ == '__main__':
    main()

