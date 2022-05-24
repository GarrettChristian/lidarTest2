

from pymongo import MongoClient
import glob, os
import numpy as np
import open3d as o3d
import math
import random
import argparse


name_label_mapping = {
        0: 'unlabeled',
        1: 'outlier',
        10: 'car',
        11: 'bicycle',
        13: 'bus',
        15: 'motorcycle',
        16: 'on-rails',
        18: 'truck',
        20: 'other-vehicle',
        20: 'person',
        31: 'bicyclist',
        32: 'motorcyclist',
        40: 'road',
        44: 'parking',
        48: 'sidewalk',
        49: 'other-ground',
        50: 'building',
        51: 'fence',
        52: 'other-structure',
        60: 'lane-marking',
        70: 'vegetation',
        71: 'trunk',
        72: 'terrain',
        80: 'pole',
        81: 'traffic-sign',
        99: 'other-object',
        252: 'moving-car',
        253: 'moving-bicyclist',
        254: 'moving-person',
        255: 'moving-motorcyclist',
        256: 'moving-on-rails',
        257: 'moving-bus',
        258: 'moving-truck',
        259: 'moving-other-vehicle'
}

centerCamPoint = np.array([0, 0, 0.3])

# ------------------------------------------------------------------

def checkInclusionBasedOnTriangleMeshAsset(points, mesh):

    obb = mesh.get_oriented_bounding_box()

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacyMesh)

    pointsVector = o3d.utility.Vector3dVector(points)

    indexesWithinBox = obb.get_point_indices_within_bounding_box(pointsVector)

    foundNum = 0
    acceptableNum = 10

    centerArea = np.array([
            [ -2.5, -2.5, -3], # bottom right
            [ -2.5, -2.5, 3], 
            [ -2.5, 2.5, -3], # top right
            [ -2.5, 2.5, 3],
            [ 2.5, 2.5, -3], # top left
            [ 2.5, 2.5, 3],
            [ 2.5, -2.5, -3], # bottom left
            [ 2.5, -2.5, 3], 
            ]).astype("float64")
    pcdCenter = o3d.geometry.PointCloud()
    pcdCenter.points = o3d.utility.Vector3dVector(centerArea)

    #  Get the asset's bounding box
    centerBox = pcdCenter.get_oriented_bounding_box()
    ignoreIndex = centerBox.get_point_indices_within_bounding_box(pointsVector)
    ignoreIndex = set(ignoreIndex)
    
    for idx in indexesWithinBox:
        if (idx not in ignoreIndex):
            pt = points[idx]
            query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

            occupancy = scene.compute_occupancy(query_point)
            if (occupancy == 1): 
                foundNum += 1
                if (foundNum >= acceptableNum):
                    return True

    return False


def assetIsValid(asset, sceneWithoutInstance):

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    #  Get the asset's bounding box
    obb = pcdAsset.get_oriented_bounding_box()
    boxPoints = np.asarray(obb.get_box_points())
    

    boxVertices = np.vstack((boxPoints, centerCamPoint))

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(boxVertices)
    hull2, _ = pcdCastHull.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
    

    assetCenter = obb.get_center()

    # Dist is acceptable
    dist = np.linalg.norm(centerCamPoint - assetCenter)
    if dist > 50:
        return hull_ls, False
    
    incuded = checkInclusionBasedOnTriangleMeshAsset(sceneWithoutInstance, hull2)
    if (incuded):
        hull_ls.paint_uniform_color((1, 0.2, 0.2))
    else:
        hull_ls.paint_uniform_color((0.2, 1, 0.2))

    return hull_ls, not incuded


def removeLidarShadowLines(asset):

    # Prepare asset and scene point clouds
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    #  Get the asset's hull mesh
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    
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

    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdCut = o3d.geometry.PointCloud()
    pcdCut.points = o3d.utility.Vector3dVector(combinedVertices)
    cutPointsHull, _ = pcdCut.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cutPointsHull)
    hull_ls.paint_uniform_color((0, 1, 1))

    return hull_ls


def viewOne(binFileName, labelsFileName):
    print(binFileName)
    print(labelsFileName)

    # Label
    label_arr = np.fromfile(labelsFileName, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 

    # Bin File
    pcdArr = np.fromfile(binFileName, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    pcdArr = np.delete(pcdArr, 3, 1)

    seenInst = set()
    for instance in labelInstance:
        seenInst.add(instance)
    
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArr)
        

    # mask1 = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    # tmp = pcdArr[mask1, :]
    # pcdScene = o3d.geometry.PointCloud()
    # pcdScene.points = o3d.utility.Vector3dVector(tmp)

    display = [pcdScene]

    # Box for center points
    centerArea = np.array([
            [ -2.5, -2.5, -3], # bottom right
            [ -2.5, -2.5, 3], 
            [ -2.5, 2.5, -3], # top right
            [ -2.5, 2.5, 3],
            [ 2.5, 2.5, -3], # top left
            [ 2.5, 2.5, 3],
            [ 2.5, -2.5, -3], # bottom left
            [ 2.5, -2.5, 3], 
            ]).astype("float64")
    
    pcdCenter = o3d.geometry.PointCloud()
    pcdCenter.points = o3d.utility.Vector3dVector(centerArea)

    #  Get the asset's bounding box
    centerBox = pcdCenter.get_oriented_bounding_box()
    centerBox.color = (0.1, 0.2, 0.2)
    display.append(centerBox)

    for instance in seenInst:
        if instance != 0:
            instancePoints = pcdArr[labelInstance == instance]

            if (np.shape(instancePoints)[0] > 20):
                pcdItem = o3d.geometry.PointCloud()
                pcdItem.points = o3d.utility.Vector3dVector(instancePoints)
                hull, _ = pcdItem.compute_convex_hull()
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
                hull_ls.paint_uniform_color((0, 0, 1))


                maskInst = (labelInstance != instance) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
                pcdWithoutInstance = pcdArr[maskInst, :]

                boxToVel, valid = assetIsValid(instancePoints, pcdWithoutInstance)
                

                display.append(hull_ls)
                if (valid):
                    display.append(removeLidarShadowLines(instancePoints))
                    display.append(boxToVel)
                    # display.append(hullToVelLines(instancePoints, pcdWithoutInstance))

                # get_oriented_bounding_box
                # get_axis_aligned_bounding_box
                obb = pcdItem.get_oriented_bounding_box()
                obb.color = (0.7, 0, 1)
                display.append(obb)



    o3d.visualization.draw_geometries(display)
    
def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    # p.add_argument(
    #     'binLocation', help='Path to the dir to test')
    p.add_argument(
        "-num", help="specific scenario number provide full ie 002732")
    p.add_argument(
        "-scene", help="Scene number provide as 1 rather than 01 (default all labeled)", 
        nargs='?', const=0, default=range(0, 11))

    return p.parse_args()
    

def main():

    print("\n\n------------------------------")
    print("\n\nStarting open3D viewer\n\n")

    path = "/home/garrett/Documents/data/dataset/sequences/"

    print("Parsing {} :".format(path))

    num = 0

    args = parse_args()

    # Get scenes

    print("Collecting Labels and Bins for scenes {}".format(args.scene))

    binFiles = []
    labelFiles = []

    
            
    print("Starting Visualization")


    if (args.num):
        currPath = path + str(args.scene).rjust(2, '0')

        labelFiles = [currPath + "/labels/" + args.num + ".label"]
        binFiles = [currPath + "/velodyne/" + args.num + ".bin"]
    
    else:
        for sceneNum in args.scene:
        
            folderNum = str(sceneNum).rjust(2, '0')
            currPath = path + folderNum

            labelFilesScene = np.array(glob.glob(currPath + "/labels/*.label", recursive = True))
            binFilesScene = np.array(glob.glob(currPath + "/velodyne/*.bin", recursive = True))
            print("Parsing Scene {}".format(folderNum))

            # Sort
            labelFilesScene = sorted(labelFilesScene)
            binFilesScene = sorted(binFilesScene)
            
            for labelFile in labelFilesScene:
                labelFiles.append(labelFile)
            
            for binFile in binFilesScene:
                binFiles.append(binFile)

    try:
        idx = random.choice(range(len(labelFiles)))
        # for idx in range(len(labelFiles)):
        print(num, binFiles[idx])
        num += 1
        viewOne(binFiles[idx], labelFiles[idx])
        

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")



if __name__ == '__main__':
    main()



