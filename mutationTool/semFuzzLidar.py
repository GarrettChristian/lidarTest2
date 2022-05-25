


import numpy as np
import open3d as o3d
import math
import random
import argparse
from enum import Enum

from globals import Mutation
from globals import Transformations
import globals

from mongoUtil import mongoConnect, saveMutation
from mongoUtil import getRandomAssetOfType
from mongoUtil import getRandomAssetScene
from fileIoUtil import openLabelBinFiles, saveToBin
from open3dUtil import assetIntersectsWalls, pointsWithinDist, rotatePoints, pointsAboveGround, mirrorPoints, getValidRotations
from open3dUtil import assetIsNotObsured
from open3dUtil import removeLidarShadow
import open3dUtil

# -------------------------------------------------------------




def intensityChange(pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, type):

    mask = np.ones(np.shape(intensityAsset), dtype=bool)

    if (type in globals.vehicles):    
        dists = open3dUtil.nearestNeighbors(intensityAsset, 2)
        class0 = intensityAsset[dists[:, 1] == 0]
        class1 = intensityAsset[dists[:, 1] == 1]

        mask = dists[:, 1] == 0
        if np.shape(class0)[0] < np.shape(class1)[0]:
            mask = dists[:, 1] == 1

    mod = random.uniform(.1, .3)

    
    print("Intensity {}".format(mod))
    
    print(mask)
    print(intensityAsset)
    intensityAsset = np.where(mask, intensityAsset + mod, intensityAsset)
    print(intensityAsset)

    return pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset


def remove(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, instanceAsset):

    pcdArr = pcdArr[labelInstance != instanceAsset]
    intensity = intensity[labelInstance != instanceAsset]
    semantics = semantics[labelInstance != instanceAsset]
    labelInstance = labelInstance[labelInstance != instanceAsset]

    return open3dUtil.replaceBasedOnShadow(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)


def rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, transformation):
    attempts = 0
    success = False
    degrees = []
    

    if (Transformations.ROTATE_MIRROR == transformation):
            axis = globals.mirrorAxis
            if (not axis):
                axis = random.randint(0, 1)
            print("Mirror Axis: {}".format(axis))
            pcdArrAsset = mirrorPoints(pcdArrAsset, axis)

    if (Transformations.ROTATE_MIRROR == transformation or Transformations.ROTATE == transformation):
        degrees = getValidRotations(pcdArrAsset, pcdArr, semantics)

        print(degrees)
    
    while (attempts < 10 and not success):
        pcdArrAssetNew = np.copy(pcdArrAsset)
        
        if (Transformations.ROTATE == transformation 
            or Transformations.ROTATE_MIRROR == transformation):
            
            rotateDeg = globals.rotation
            if (not rotateDeg):
                rotateDeg = random.choice(degrees)
            print(rotateDeg)
            pcdArrAssetNew = rotatePoints(pcdArrAssetNew, rotateDeg)

        # Get asset box
        pcdAsset = o3d.geometry.PointCloud()
        pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAssetNew)
        hull, _ = pcdAsset.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((0, 0, 1))

        # Get scene
        pcdScene = o3d.geometry.PointCloud()
        pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        # Color as intensity
        colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
        colors[:, 0] = intensity
        pcdScene.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([hull_ls, pcdScene])

        print("Check not in walls")
        success = not assetIntersectsWalls(pcdArrAssetNew, pcdArr, semantics)
        if (success):
            print("Not in walls")

            print("Check on Road")
            success = pointsAboveGround(pcdArrAssetNew, pcdArr, semantics) or pointsWithinDist(pcdArrAssetNew) 
            if (success):
                print("On Road")

                print("Check unobsuccured")
                success = assetIsNotObsured(pcdArrAssetNew, pcdArr, semantics)
                if (success):
                    print("Asset Unobscured")
                    pcdArrAsset = pcdArrAssetNew
                    print("Removing shadow")
                    pcdArr, intensity, semantics, labelInstance = removeLidarShadow(pcdArrAssetNew, pcdArr, intensity, semantics, labelInstance)
                    # Combine
                    pcdArr = np.vstack((pcdArr, pcdArrAsset))
                    intensity = np.hstack((intensity, intensityAsset))
                    semantics = np.hstack((semantics, semanticsAsset))
                    labelInstance = np.hstack((labelInstance, labelInstanceAsset))

        attempts += 1


     # Visualize the mutation if enabled
    if success and globals.visualize:

        # Get asset box
        pcdAsset = o3d.geometry.PointCloud()
        pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
        hull, _ = pcdAsset.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((0, 0, 1))

        # Get scene
        pcdScene = o3d.geometry.PointCloud()
        pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        # Color as intensity
        colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
        colors[:, 0] = intensity
        pcdScene.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([hull_ls, pcdScene])
    
    return success, pcdArr, intensity, semantics, labelInstance

# ----------------------------------------------------------


def performMutation():

    # Select Seed
    idx = random.choice(range(len(globals.labelFiles)))
    print(globals.binFiles[idx])

    # Select Mutation
    mutation = random.choice(globals.mutationsEnabled)

    transformation = random.choice(globals.tranformationsEnabled)

    print(mutation, transformation)

    # Base:
    pcdArr, intensity, semantics, labelInstance = openLabelBinFiles(globals.binFiles[idx], globals.labelFiles[idx])
    pcdArrAsset = None
    intensityAsset = None
    semanticsAsset = None
    labelInstanceAsset = None
    assetRecord = None

    success = False
    if (Mutation.ADD == mutation):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = getRandomAssetOfType(10)
        print(assetRecord)

    elif (Mutation.SCENE == mutation):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = getRandomAssetScene(globals.binFiles[idx])
        print(assetRecord)
        
    else:
        print("ERROR: {} NOT SUPPORTED".format())

    if (Transformations.INTENSITY == transformation):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset  = intensityChange(pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord["typeNum"])
        pcdArr, intensity, semantics, labelInstance = open3dUtil.combineRemoveInstance(pcdArr, intensity, semantics, labelInstance,
                                                                            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord["instance"])
        success = True

        # Visualize the mutation if enabled
        if success and globals.visualize:

            # Get asset box
            pcdAsset = o3d.geometry.PointCloud()
            pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
            hull, _ = pcdAsset.compute_convex_hull()
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color((0, 0, 1))

            # Get scene
            pcdScene = o3d.geometry.PointCloud()
            pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

            # Color as intensity
            colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
            colors[:, 0] = intensity
            pcdScene.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([hull_ls, pcdScene])

    elif (Transformations.REMOVE == transformation):
        pcdArr, intensity, semantics, labelInstance = remove(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, assetRecord["instance"])
        success = True

        # Visualize the mutation if enabled
        if success and globals.visualize:

            # Get asset box
            pcdAsset = o3d.geometry.PointCloud()
            pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
            hull, _ = pcdAsset.compute_convex_hull()
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            hull_ls.paint_uniform_color((0, 0, 1))

            # Get scene
            pcdScene = o3d.geometry.PointCloud()
            pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

            # Color as intensity
            colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
            colors[:, 0] = intensity
            pcdScene.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([hull_ls, pcdScene])
    elif (Transformations.ROTATE_MIRROR == transformation or Transformations.ROTATE == transformation):
        success, pcdArr, intensity, semantics, labelInstance = rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, 
                                                                intensityAsset, semanticsAsset, labelInstanceAsset, transformation)

       

    if (success):
        saveMutation("tmp")
        saveToBin(pcdArr, intensity, labelInstance, "test.bin")
    

# -------------------------------------------------------------


# Boot Util


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    p.add_argument("-seq", 
        help="Sequences number provide as 1 rather than 01 (default all labeled 0-10)", 
        nargs='?', const=0, default=range(0, 11))
    p.add_argument( "-scene", 
        help="specific scenario number provide full ie 002732")
    p.add_argument("-path", 
        help="Path to the sequences", 
        nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
        default="/home/garrett/Documents/data/dataset/sequences/")
    p.add_argument('-m', required=False,
        help='Mutations to perform comma seperated example: ADD,REMOVE')
    p.add_argument('-t', required=False,
        help='Transformations to perform comma seperated example: ROTATE,ROTATE_MIRROR')

    p.add_argument('-vis', help='Visualize with Open3D',
        action='store_true', default=False)
    
    p.add_argument('-rotate', help='Value to rotate', required=False)
    p.add_argument('-mirror', help='Value to mirror', required=False)
    p.add_argument('-intensity', help='Value to change intensity', required=False)


    p.add_argument('-asset', help='Specific assetId to load', required=False)


    return p.parse_args()

    


# ----------------------------------------------------------

def main():

    print("\n\n------------------------------")
    print("\n\nStarting Fuzzer\n\n")
    

    args = parse_args()
    globals.init(args)
    
    mongoConnect()


    # Mutate
    print("Starting Mutation")

    num = 0

    try:
        performMutation()        

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")



if __name__ == '__main__':
    main()



