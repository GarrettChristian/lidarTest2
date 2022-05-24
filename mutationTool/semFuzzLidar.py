


import numpy as np
import open3d as o3d
import math
import random
import argparse
from enum import Enum

from globals import Mutation
from globals import Transformations
import globals

from mongoUtil import mongoConnect
from mongoUtil import getRandomAssetOfType
from mongoUtil import getRandomAssetWithinScene
from fileIoUtil import openLabelBinFiles
from open3dUtil import assetIntersectsWalls, rotatePoints, pointsAboveGround, mirrorPoints
from open3dUtil import assetIsNotObsured
from open3dUtil import removeLidarShadow

# -------------------------------------------------------------



# ----------------------------------------------------------


def performMutation():

    # Select Seed
    idx = random.choice(range(len(globals.labelFiles)))
    print(globals.binFiles[idx])

    # Select Mutation
    mutation = random.choice(globals.mutationsEnabled)
    print(mutation)

    # Base:
    pcdArr, intensity, semantics, labelInstance = openLabelBinFiles(globals.binFiles[idx], globals.labelFiles[idx])
    pcdArrAsset = None
    intensityAsset = None
    semanticsAsset = None
    labelInstanceAsset = None

    if (Mutation.ADD == mutation):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset = getRandomAssetOfType(10)
        

    elif (Mutation.COPY == mutation):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset = getRandomAssetWithinScene(10)

    elif (Mutation.REMOVE == mutation):
        print("TODO")

    else:
        print("ERROR: {} NOT SUPPORTED".format())


    attempts = 0
    success = False
    while (attempts < 10 and not success):
        pcdArrAssetNew = np.copy(pcdArrAsset)
                
        transformation = random.choice(globals.tranformationsEnabled)
        
        if (Transformations.ROTATE == transformation or Transformations.ROTATE_MIRROR == transformation):
            rotateDeg = globals.rotation
            if (not rotateDeg):
                rotateDeg = random.randint(0, 360)
            print(rotateDeg)
            pcdArrAssetNew = rotatePoints(pcdArrAssetNew, rotateDeg)
            
        # elif (Transformations.MIRROR == transformation or Transformations.ROTATE_MIRROR == transformation):
        if (Transformations.ROTATE_MIRROR == transformation):
            axis = globals.mirrorAxis
            if (not axis):
                axis = random.randint(0, 1)
            print("Mirror Axis: {}".format(axis))
            pcdArrAssetNew = mirrorPoints(pcdArrAssetNew, axis)
        

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
            success = success and pointsAboveGround(pcdArrAssetNew, pcdArr, semantics)
            if (success):
                print("On Road")

                print("Check unobsuccured")
                success = success and assetIsNotObsured(pcdArrAssetNew, pcdArr, semantics)
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



