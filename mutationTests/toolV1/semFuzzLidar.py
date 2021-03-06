


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

saveNum = 0

def scale(pcdArrAsset):

    posX = random.randint(0, 1)
    percentScale = random.uniform(0.6, 0.9)
    if (posX == 1):
        percentScale = random.uniform(1.1, 1.4)

    print("scale {}".format(percentScale))

    return open3dUtil.scale(pcdArrAsset, percentScale)


def densify(pcdArrAsset, intensity, semantics, labelInstance):

    percentRemove = random.uniform(0.1, 0.4)
    amountToRemove = int(np.floor(np.shape(pcdArrAsset)[0] * percentRemove))
    print("densify {} {}".format(amountToRemove, percentRemove))

    maskRemove = np.full(np.shape(pcdArrAsset)[0], False)
    maskRemove[:amountToRemove] = True
    np.random.shuffle(maskRemove)

    pcdArrAssetSparse = pcdArrAsset[maskRemove]
    intensitySparse = intensity[maskRemove]
    semanticsSparse = semantics[maskRemove]
    labelInstanceSparse = labelInstance[maskRemove]

    pcdArrAssetSparse = noise(pcdArrAssetSparse)

    return open3dUtil.combine(pcdArrAsset, intensity, semantics, labelInstance, 
                        pcdArrAssetSparse, intensitySparse, semanticsSparse, labelInstanceSparse)


def sparsify(pcdArrAsset, intensity, semantics, labelInstance):

    percentRemove = random.uniform(0.1, 0.4)
    amountToRemove = int(np.floor(np.shape(pcdArrAsset)[0] * percentRemove))

    print("sparsify {} {}".format(amountToRemove, percentRemove))
    maskRemove = np.full(np.shape(pcdArrAsset)[0], True)
    maskRemove[:amountToRemove] = False
    np.random.shuffle(maskRemove)

    pcdArrAssetSparse = pcdArrAsset[maskRemove]
    intensitySparse = intensity[maskRemove]
    semanticsSparse = semantics[maskRemove]
    labelInstanceSparse = labelInstance[maskRemove]

    return pcdArrAssetSparse, intensitySparse, semanticsSparse, labelInstanceSparse


def noise(pcdArrAsset):

    asset = np.copy(pcdArrAsset)

    mu, sigma = 0, 0.1 
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma, np.shape(pcdArrAsset)) 

    asset = asset + noise

    return asset



def translateFinal(pcdArrAsset, pcdArr, intensity, semantics, labelInstance):

    
    success, pcdArrAssetTranslated = translateV2(pcdArrAsset, pcdArr, semantics)

    if success:
        pcdArr, intensity, semantics, labelInstance = removeLidarShadow(pcdArrAssetTranslated, pcdArr, intensity, semantics, labelInstance)
    
    return success, pcdArrAssetTranslated, pcdArr, intensity, semantics, labelInstance


# def translate(pcdArrAsset, pcdArr, semantics):

#     success = False
#     attempts = 0
#     while attempts < 10 and not success:

#         posX = random.randint(0, 1)
#         xTranslate = random.uniform(-0.3, -2)
#         if (posX == 1):
#             xTranslate = random.uniform(0.3, 2)

#         posY = random.randint(0, 1)
#         yTranslate = random.uniform(-0.3, -2)
#         if (posY == 1):
#             yTranslate = random.uniform(0.3, 2)

#         print(xTranslate, yTranslate)

#         pcdAssetTranslated = open3dUtil.translatePointsXY(pcdArrAsset, xTranslate, yTranslate)


#         print("Check not in walls")
#         success = not assetIntersectsWalls(pcdAssetTranslated, pcdArr, semantics)
#         if (success):
#             print("Not in walls")

#             print("Check on Road")
#             success = pointsAboveGround(pcdAssetTranslated, pcdArr, semantics) or pointsWithinDist(pcdAssetTranslated) 
#             if (success):
#                 print("On Road")

#                 print("Check unobsuccured")
#                 success = assetIsNotObsured(pcdAssetTranslated, pcdArr, semantics)
#                 if (success):
#                     print("Asset Unobscured")
#                     pcdArrAsset = pcdAssetTranslated
#                     print("Removing shadow")

#         attempts += 1


#     return success, pcdAssetTranslated


def translateV2(pcdArrAsset, pcdArr, semantics):

    success = False
    attempts = 0
    while attempts < 10 and not success:

        posX = random.randint(0, 1)
        translateAmount = random.uniform(-0.3, -2)
        if (posX == 1):
            translateAmount = random.uniform(0.3, 2)

        pcdAssetTranslated = open3dUtil.translatePointsFromCenter(pcdArrAsset, translateAmount)


        print("Translate", translateAmount)

        # Get asset box
        pcdAsset = o3d.geometry.PointCloud()
        pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
        hull, _ = pcdAsset.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((0, 0, 1))

        pcdAsset = o3d.geometry.PointCloud()
        pcdAsset.points = o3d.utility.Vector3dVector(pcdAssetTranslated)
        hull, _ = pcdAsset.compute_convex_hull()
        hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls2.paint_uniform_color((0, 1, 0))

        # Get scene
        pcdScene = o3d.geometry.PointCloud()
        pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        o3d.visualization.draw_geometries([hull_ls, hull_ls2, pcdScene])

        print("Check not in walls")
        success = not assetIntersectsWalls(pcdAssetTranslated, pcdArr, semantics)
        if (success):
            print("Not in walls")

            print("Check on Road")
            success = pointsAboveGround(pcdAssetTranslated, pcdArr, semantics) or pointsWithinDist(pcdAssetTranslated) 
            if (success):
                print("On Road")

                print("Check unobsuccured")
                success = assetIsNotObsured(pcdAssetTranslated, pcdArr, semantics)
                if (success):
                    print("Asset Unobscured")
                    pcdArrAsset = pcdAssetTranslated
                    print("Removing shadow")

        attempts += 1


    return success, pcdAssetTranslated



def intensityChange(intensityAsset, type):

    mask = np.ones(np.shape(intensityAsset), dtype=bool)

    if (type in globals.vehicles):    
        dists = open3dUtil.nearestNeighbors(intensityAsset, 2)
        class0 = intensityAsset[dists[:, 1] == 0]
        class1 = intensityAsset[dists[:, 1] == 1]

        mask = dists[:, 1] == 0
        if np.shape(class0)[0] < np.shape(class1)[0]:
            mask = dists[:, 1] == 1

    average = np.average(intensityAsset[mask])
    
    mod = random.uniform(.1, .3)
    if average > .1:
        mod = random.uniform(-.1, -.3)

    
    print("Intensity {}".format(mod))
    
    print(mask)
    print(intensityAsset)
    intensityAsset = np.where(mask, intensityAsset + mod, intensityAsset)
    intensityAsset = np.where(intensityAsset < 0, 0, intensityAsset)
    print(intensityAsset)
    print(average)

    return intensityAsset


def rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset):
    attempts = 0
    success = False
    degrees = []
    
    degrees = getValidRotations(pcdArrAsset, pcdArr, semantics)

    print(degrees)
    
    while (len(degrees) > 0 and attempts < 10 and not success):
            
        rotateDeg = globals.rotation
        if (not rotateDeg):
            modifier = random.randint(-4, 4)
            rotateDeg = random.choice(degrees) + modifier
            if rotateDeg < 0:
                rotateDeg = 0
            elif rotateDeg > 360:
                rotateDeg = 360
        print(rotateDeg)
        pcdArrAssetNew = rotatePoints(pcdArrAsset, rotateDeg)


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
        

        print("Check on Road")
        success = pointsAboveGround(pcdArrAssetNew, pcdArr, semantics) or pointsWithinDist(pcdArrAssetNew) 
        if (success):
            print("On Road")

            print("align to Z dim")
            pcdArrAssetNew = open3dUtil.alignZdim(pcdArrAssetNew, pcdArr, semantics)

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

                print("Check unobsuccured")
                success = assetIsNotObsured(pcdArrAssetNew, pcdArr, semantics)
                if (success):
                    print("Asset Unobscured")
                    pcdArrAsset = pcdArrAssetNew
                    print("Removing shadow")
                    pcdArr, intensity, semantics, labelInstance = removeLidarShadow(pcdArrAssetNew, pcdArr, intensity, semantics, labelInstance)

        attempts += 1
    
    return success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance

# ----------------------------------------------------------


def performMutation():
    global saveNum

    print(saveNum)

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

    success = True

    tranformationSplitString = str(transformation).replace("Transformations.", "")
    tranformationSplit = tranformationSplitString.split('_')
    assetLocation = tranformationSplit[0]

    # Get the asset
    if (assetLocation == "ADD"):
        instanceType = random.choice(list(globals.instances.keys()))
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = getRandomAssetOfType(instanceType)
        print(assetRecord)

    elif (assetLocation == "SCENE"):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = getRandomAssetScene(globals.binFiles[idx])
        pcdArr, intensity, semantics, labelInstance = open3dUtil.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)
        print(assetRecord)
        
    else:
        print("ERROR: {} NOT SUPPORTED".format(assetLocation))
        exit()

    
    if globals.visualize:

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


    for transformIndex in range (1, len(tranformationSplit)):
    # if (Transformations.INTENSITY == transformation):
        if (tranformationSplit[transformIndex] == "INTENSITY"):
            intensityAsset = intensityChange(intensityAsset, assetRecord["typeNum"])
            
        elif (tranformationSplit[transformIndex] == "NOISE"):
            pcdArrAsset = noise(pcdArrAsset)  
        
        elif (tranformationSplit[transformIndex] == "SCALE"):
            pcdArrAsset = scale(pcdArrAsset)       

        elif (tranformationSplit[transformIndex] == "REMOVE"):
            pcdArr, intensity, semantics, labelInstance = open3dUtil.replaceBasedOnShadow(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)

        elif (tranformationSplit[transformIndex] == "DENSIFY"):
            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset = densify(pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset)

        elif (tranformationSplit[transformIndex] == "SPARSIFY"):
            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset = sparsify(pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset)

        elif (tranformationSplit[transformIndex] == "MIRROR"):
            axis = globals.mirrorAxis
            if (not axis):
                axis = random.randint(0, 1)
            print("Mirror Axis: {}".format(axis))
            pcdArrAsset = mirrorPoints(pcdArrAsset, axis)

        elif (tranformationSplit[transformIndex] == "TRANSLATE"):
            if transformIndex + 1 == len(tranformationSplit):
                success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance = translateFinal(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)
            else:
                pcdArrAsset = translateV2(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)

        elif (tranformationSplit[transformIndex] == "ROTATE"):
            success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance = rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset)

        else:
            print("NOT SUPPORTED {}".format(tranformationSplit[transformIndex]))

    if success:
        pcdArr, intensity, semantics, labelInstance = open3dUtil.combine(pcdArr, intensity, semantics, labelInstance, 
                                                                        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset)
       

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


    if (success):
        saveMutation("tmp")
        saveToBin(pcdArr, intensity, semantics, labelInstance, str(saveNum))
        saveNum += 1
    

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
        for x in range(0, 1):
            performMutation()        

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")



if __name__ == '__main__':
    main()



