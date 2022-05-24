
from pymongo import MongoClient
import glob, os
import numpy as np
import open3d as o3d
import math


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
        30: 'person',
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

def hullToVelLines(asset):

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    centerCamPoint = np.array([0, 0, -0.5])

    #  Get the asset's hull mesh
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)

    hullVertices = np.vstack((hullVertices, centerCamPoint))

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(hullVertices)
    hull2, _ = pcdCastHull.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
    hull_ls.paint_uniform_color((0, 0.5, 1))

    return hull_ls


def removeLidarShadowLines(asset):

    # Prepare asset and scene point clouds
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

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
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cutPointsHull)
    hull_ls.paint_uniform_color((0, 0, 1))

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
        
    display = [pcdScene]

    for instance in seenInst:
        if instance != 0:
            instancePoints = pcdArr[labelInstance == instance]

            pcdItem = o3d.geometry.PointCloud()
            pcdItem.points = o3d.utility.Vector3dVector(instancePoints)
            hull, _ = pcdItem.compute_convex_hull()
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color((0, 0, 1))

            display.append(hull_ls)
            display.append(removeLidarShadowLines(instancePoints))
            display.append(hullToVelLines(instancePoints))

    o3d.visualization.draw_geometries(display)
    

def main():

    print("\n\n------------------------------")
    print("\n\nStarting open3D viewer\n\n")

    path = "/home/garrett/Documents/data/dataset/sequences/"

    print("Parsing {} :".format(path))

    num = 0

    # Get scenes
    scenes = range(0, 11)

    print("Collecting Labels and Bins for scenes {}".format(scenes))

    binFiles = []
    labelFiles = []

    for sceneNum in scenes:
        
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
            
    print("Starting Visualization")

    try:
        for idx in range(len(labelFiles)):
            viewOne(binFiles[idx], labelFiles[idx])
            print(num, binFiles[idx])
            num += 1

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")



if __name__ == '__main__':
    main()



