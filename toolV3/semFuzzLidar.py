


import numpy as np
import open3d as o3d
import math
import random
import argparse
from enum import Enum
import shortuuid
import os
import sys
import time
import json


import globals

from mongoUtil import mongoConnect, saveMutation
from mongoUtil import getRandomAssetOfType
from mongoUtil import getRandomAssetWithinScene
import mongoUtil
from pcdUtil import assetIntersectsWalls, pointsWithinDist, rotatePoints, pointsAboveGround, mirrorPoints, getValidRotations
from pcdUtil import assetIsNotObsured
from pcdUtil import removeLidarShadow
import pcdUtil as pcdUtil
import fileIoUtil
import eval




# -------------------------------------------------------------



"""
formatSecondsToHhmmss
Helper to convert seconds to hours minutes and seconds
@param seconds
@return formatted string of hhmmss
"""
def formatSecondsToHhmmss(seconds):
    hours = seconds / (60*60)
    seconds %= (60*60)
    minutes = seconds / 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)




# http://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
def deform(asset, details):

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    # Select random point
    pointIndex = np.random.choice(asset.shape[0], 1, replace=False)
    print(pointIndex)

    # Nearest k points
    assetNumPoints = np.shape(asset)[0]
    percentDeform = random.uniform(0.05, 0.12)
    k = int(assetNumPoints * percentDeform)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAsset)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcdAsset.points[pointIndex], k)
    # np.asarray(pcdAsset.colors)[idx[1:], :] = [0, 0, 1]

    mu, sigma = 0.05, 0.04
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma, (k))
    # noise = np.lexsort((noise[:,0], noise[:,1]))
    noise = np.sort(noise)[::-1]
    # print(np.shape(noise))
    # print(noise)
    print("total points {}".format(assetNumPoints))
    print("deformPoints {}".format(k))
    print("deformPercent {}".format(percentDeform))
    print("deformMu {}".format(mu))
    print("deformSigma {}".format(sigma))
    details["deformPercent"] = percentDeform
    details["deformPoints"] = k
    details["deformMu"] = mu
    details["deformSigma"] = sigma

    for index in range(0, len(idx)):
        asset[idx[index]] = pcdUtil.translatePointFromCenter(asset[idx[index]], noise[index])
        # if index != 0:
        #     pcdAsset.colors[idx[index]] = [0, 0, 1 - (index * .002)]

    # pcdAsset.points = o3d.utility.Vector3dVector(asset)

    # print(idx)

    # o3d.visualization.draw_geometries([pcdAsset])
    # o3d.visualization.draw_geometries([pcdAsset, pcd])

    return asset, details


def scale(pcdArrAsset, details):

    posX = random.randint(0, 1)
    percentScale = random.uniform(0.6, 0.9)
    if (posX == 1):
        percentScale = random.uniform(1.1, 1.4)

    details["scale"] = percentScale

    print("scale {}".format(percentScale))

    pcdScaled =  pcdUtil.scale(pcdArrAsset, percentScale)

    return pcdScaled, details


def densify(pcdArrAsset, intensity, semantics, labelInstance, details):

    percentRemove = random.uniform(0.1, 0.4)
    amountToRemove = int(np.floor(np.shape(pcdArrAsset)[0] * percentRemove))
    print("densify {} {}".format(amountToRemove, percentRemove))

    details["densify"] = percentRemove

    maskRemove = np.full(np.shape(pcdArrAsset)[0], False)
    maskRemove[:amountToRemove] = True
    np.random.shuffle(maskRemove)

    pcdArrAssetSparse = pcdArrAsset[maskRemove]
    intensitySparse = intensity[maskRemove]
    semanticsSparse = semantics[maskRemove]
    labelInstanceSparse = labelInstance[maskRemove]

    pcdArrAssetSparse, details = noise(pcdArrAssetSparse, details)

    pcdArrDense, intensityDense, semanticsDense, labelInstanceDense = pcdUtil.combine(pcdArrAsset, intensity, semantics, labelInstance, 
                        pcdArrAssetSparse, intensitySparse, semanticsSparse, labelInstanceSparse)


    return pcdArrDense, intensityDense, semanticsDense, labelInstanceDense, details


def sparsify(pcdArrAsset, intensity, semantics, labelInstance, details):

    percentRemove = random.uniform(0.1, 0.4)
    amountToRemove = int(np.floor(np.shape(pcdArrAsset)[0] * percentRemove))

    details["sparcify"] = percentRemove

    print("sparsify {} {}".format(amountToRemove, percentRemove))
    maskRemove = np.full(np.shape(pcdArrAsset)[0], True)
    maskRemove[:amountToRemove] = False
    np.random.shuffle(maskRemove)

    pcdArrAssetSparse = pcdArrAsset[maskRemove]
    intensitySparse = intensity[maskRemove]
    semanticsSparse = semantics[maskRemove]
    labelInstanceSparse = labelInstance[maskRemove]

    return pcdArrAssetSparse, intensitySparse, semanticsSparse, labelInstanceSparse, details


def noise(pcdArrAsset, details):

    asset = np.copy(pcdArrAsset)

    mu, sigma = 0, 0.1 
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma, np.shape(pcdArrAsset))

    details["noise"] = sigma

    asset = asset + noise

    return asset, details



def translateFinal(pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details):

    
    success, pcdArrAssetTranslated, details = translateV2(pcdArrAsset, pcdArr, semantics, details)

    if success:
        pcdArr, intensity, semantics, labelInstance = removeLidarShadow(pcdArrAssetTranslated, pcdArr, intensity, semantics, labelInstance)
    
    return success, pcdArrAssetTranslated, pcdArr, intensity, semantics, labelInstance, details


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


def translateV2(pcdArrAsset, pcdArr, semantics, details):

    success = False
    attempts = 0
    while attempts < 10 and not success:

        posX = random.randint(0, 1)
        translateAmount = random.uniform(-0.3, -2)
        if (posX == 1):
            translateAmount = random.uniform(0.3, 2)

        details["translate"] = translateAmount

        pcdAssetTranslated = pcdUtil.translatePointsFromCenter(pcdArrAsset, translateAmount)

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
        # pcdScene = o3d.geometry.PointCloud()
        # pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        # o3d.visualization.draw_geometries([hull_ls, hull_ls2, pcdScene])

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


    return success, pcdAssetTranslated, details



def intensityChange(intensityAsset, type, details):

    mask = np.ones(np.shape(intensityAsset), dtype=bool)

    if (type in globals.vehicles):    
        dists = pcdUtil.nearestNeighbors(intensityAsset, 2)
        class0 = intensityAsset[dists[:, 1] == 0]
        class1 = intensityAsset[dists[:, 1] == 1]

        mask = dists[:, 1] == 0
        if np.shape(class0)[0] < np.shape(class1)[0]:
            mask = dists[:, 1] == 1

    average = np.average(intensityAsset[mask])
    
    mod = random.uniform(.1, .3)
    if average > .1:
        mod = random.uniform(-.1, -.3)

    details["intensity"] = mod
    
    print("Intensity {}".format(mod))
    
    print(mask)
    print(intensityAsset)
    intensityAsset = np.where(mask, intensityAsset + mod, intensityAsset)
    intensityAsset = np.where(intensityAsset < 0, 0, intensityAsset)
    print(intensityAsset)
    print(average)

    return intensityAsset, details


def rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, details):
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
        details['rotate'] = rotateDeg

        pcdArrAssetNew = rotatePoints(pcdArrAsset, rotateDeg)


        # # Get asset box
        # pcdAsset = o3d.geometry.PointCloud()
        # pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAssetNew)
        # hull, _ = pcdAsset.compute_convex_hull()
        # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        # hull_ls.paint_uniform_color((0, 0, 1))

        # # Get scene
        # pcdScene = o3d.geometry.PointCloud()
        # pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        # # Color as intensity
        # colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
        # colors[:, 0] = intensity
        # pcdScene.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([hull_ls, pcdScene])
        

        print("Check on Road")
        success = pointsAboveGround(pcdArrAssetNew, pcdArr, semantics) or pointsWithinDist(pcdArrAssetNew) 
        if (success):
            print("On Road")

            print("align to Z dim")
            pcdArrAssetNew = pcdUtil.alignZdim(pcdArrAssetNew, pcdArr, semantics)

            # # Get asset box
            # pcdAsset = o3d.geometry.PointCloud()
            # pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAssetNew)
            # hull, _ = pcdAsset.compute_convex_hull()
            # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            # hull_ls.paint_uniform_color((0, 0, 1))

            # # Get scene
            # pcdScene = o3d.geometry.PointCloud()
            # pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

            # # Color as intensity
            # colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
            # colors[:, 0] = intensity
            # pcdScene.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([hull_ls, pcdScene])
            

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
    
    return success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details

# ----------------------------------------------------------


def performMutation():

    # Select Mutation
    # mutation = random.choice(globals.mutationsEnabled)
    mutation = random.choice(globals.mutationsEnabled)
    mutationSplitString = str(mutation).replace("Mutation.", "")
    
    print(mutationSplitString)

    # Create mutation details
    mutationId = str(shortuuid.uuid())
    details = {}
    details["_id"] = mutationId + "-" + mutationSplitString
    details["mutationId"] = mutationId
    details["time"] = int(time.time())
    details["dateTime"] = time.ctime(time.time())
    details["batchId"] = globals.batchId
    details["mutation"] = mutationSplitString

    # Select Seed
    idx = random.choice(range(len(globals.labelFiles)))
    print(globals.binFiles[idx])
    head_tail = os.path.split(globals.binFiles[idx])
    scene = head_tail[1]
    scene = scene.replace('.bin', '')
  
    head_tail = os.path.split(head_tail[0])
    head_tail = os.path.split(head_tail[0])
    sequence = head_tail[1]
    details["baseSequence"] = sequence
    details["baseScene"] = scene

    # Base:
    pcdArr, intensity, semantics, labelInstance = fileIoUtil.openLabelBinFiles(globals.binFiles[idx], globals.labelFiles[idx])
    pcdArrAsset = None
    intensityAsset = None
    semanticsAsset = None
    labelInstanceAsset = None
    assetRecord = None
    pcdArrOriginal = np.copy(pcdArr)

    success = True
    combine = True

    mutationSplit = mutationSplitString.split('_')
    assetLocation = mutationSplit[0]
    
    mutationSet = set()
    for mutation in mutationSplit:
        mutationSet.add(mutation)

    # Get the asset
    if (globals.assetId != None):
        pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = mongoUtil.getAssetById(globals.assetId)

        if (assetRecord != None and "REMOVE" in mutationSet
            and assetRecord["sequence"] == sequence and assetRecord["scene"] == scene):
            pcdArr, intensity, semantics, labelInstance = pcdUtil.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)
        
    elif (assetLocation == "ADD"):
        if ("DEFORM" in mutationSet):
            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = mongoUtil.getRandomAssetOfTypes(globals.vehicles)
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = mongoUtil.getRandomAsset()

    elif (assetLocation == "SCENE"):
        if ("DEFORM" in mutationSet):
            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = mongoUtil.getRandomAssetOfTypesWithinScene(globals.vehicles, sequence, scene)
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetRecord = getRandomAssetWithinScene(sequence, scene)

        
        if (assetRecord != None):
            pcdArr, intensity, semantics, labelInstance = pcdUtil.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)
        
    else:
        print("ERROR: {} NOT SUPPORTED".format(assetLocation))
        exit()

    # Validate the asset was found
    if assetRecord == None:
        print("Invalid Asset / No asset found")
        success = False
    else:
        print(assetRecord)
        details["asset"] = assetRecord["_id"]
        details["assetType"] = assetRecord["type"]
    

    # if globals.visualize:

    #     # Get asset box
    #     pcdAsset = o3d.geometry.PointCloud()
    #     pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
    #     hull, _ = pcdAsset.compute_convex_hull()
    #     hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    #     hull_ls.paint_uniform_color((0, 0, 1))

    #     # Get scene
    #     pcdScene = o3d.geometry.PointCloud()
    #     pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

    #     # Color as intensity
    #     colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
    #     colors[:, 0] = intensity
    #     pcdScene.colors = o3d.utility.Vector3dVector(colors)

    #     o3d.visualization.draw_geometries([hull_ls, pcdScene])


    for mutationIndex in range (1, len(mutationSplit)):
        if success:
        # if (Transformations.INTENSITY == transformation):
            if (mutationSplit[mutationIndex] == "INTENSITY"):
                intensityAsset, details = intensityChange(intensityAsset, assetRecord["typeNum"], details)
                
            elif (mutationSplit[mutationIndex] == "DEFORM"):
                pcdArrAsset, details = deform(pcdArrAsset, details)

            elif (mutationSplit[mutationIndex] == "NOISE"):
                pcdArrAsset, details = noise(pcdArrAsset, details)  
            
            elif (mutationSplit[mutationIndex] == "SCALE"):
                pcdArrAsset, details = scale(pcdArrAsset, details)       

            elif (mutationSplit[mutationIndex] == "REMOVE"):
                pcdArr, intensity, semantics, labelInstance = pcdUtil.replaceBasedOnShadow(pcdArrAsset, pcdArr, intensity, semantics, labelInstance)
                if mutationIndex + 1 == len(mutationSplit):
                    combine = False

            elif (mutationSplit[mutationIndex] == "DENSIFY"):
                pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, details = densify(pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, details)

            elif (mutationSplit[mutationIndex] == "SPARSIFY"):
                pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, details = sparsify(pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, details)

            elif (mutationSplit[mutationIndex] == "MIRROR"):
                axis = globals.mirrorAxis
                if (not axis):
                    axis = random.randint(0, 1)
                print("Mirror Axis: {}".format(axis))
                details["mirror"] = 1
                pcdArrAsset = mirrorPoints(pcdArrAsset, axis)

            elif (mutationSplit[mutationIndex] == "TRANSLATE"):
                if mutationIndex + 1 == len(mutationSplit):
                    success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details = translateFinal(pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details)
                else:
                    pcdArrAsset, details = translateV2(pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details)

            elif (mutationSplit[mutationIndex] == "ROTATE"):
                success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details = rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, details)

            else:
                print("NOT SUPPORTED {}".format(mutationSplit[mutationIndex]))


    if success and combine:
        pcdArr, intensity, semantics, labelInstance = pcdUtil.combine(pcdArr, intensity, semantics, labelInstance, 
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
        if ("INTENSITY" in mutationSet):
            colors[:, 0] = intensity
        else:
            for semIdx in range(0, len(semantics)):
                colors[semIdx][0] = (globals.color_map_alt[semantics[semIdx]][0] / 255)
                colors[semIdx][1] = (globals.color_map_alt[semantics[semIdx]][1] / 255)
                colors[semIdx][2] = (globals.color_map_alt[semantics[semIdx]][2] / 255)
        pcdScene.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([hull_ls, pcdScene])


    xyziFinal = None
    labelFinal = None

    

    if (success):
        xyziFinal, labelFinal = fileIoUtil.prepareToSave(pcdArr, intensity, semantics, labelInstance)

    
    return success, details, xyziFinal, labelFinal
    




def prepFinalDetails():
    finalData = {}

    finalData["batchId"] = globals.batchId

    models = ["cyl", "spv", "sal"]

    for model in models:
        finalData[model] = {}

        for mutation in globals.mutationsEnabled:
            mutationString = str(mutation).replace("Mutation.", "")

            finalData[model][mutationString] = {}
            finalData[model][mutationString]["five"] = []


    for mutation in globals.mutationsEnabled:
        mutationString = str(mutation).replace("Mutation.", "")
        finalData[mutationString] = 0


    return finalData



def finalDetails(details, finalData):

    models = ["cyl", "spv", "sal"]

    potentialRemove = set()
    deleteFiles = []
    
    for detail in details:
        # Add count for mutation
        finalData[detail["mutation"]] = finalData[detail["mutation"]] + 1

        # Check if we have a lower accuracy change for this mutation
        for model in models:

            # don't have five yet, add it 
            if (len(finalData[model][detail["mutation"]]["five"]) < 5):
                finalData[model][detail["mutation"]]["five"].append((detail["_id"], detail[model]["accuracyChange"]))
                finalData[model][detail["mutation"]]["five"].sort(key = lambda x: x[1])

            # Do have five check against current highest
            else:
                idRemove = detail["_id"]
                    
                # new lower change to acc
                if (finalData[model][detail["mutation"]]["five"][4][1] > detail[model]["accuracyChange"]):
                    finalData[model][detail["mutation"]]["five"].append((detail["_id"], detail[model]["accuracyChange"]))
                    finalData[model][detail["mutation"]]["five"].sort(key = lambda x: x[1])
                    idRemove = finalData[model][detail["mutation"]]["five"].pop()[0]
            
                potentialRemove.add(idRemove)
                

    idInUse = set()
    for mutation in globals.mutationsEnabled:
        mutationString = str(mutation).replace("Mutation.", "")
        for model in models:
            for detailRecord in finalData[model][mutationString]["five"]:
                idInUse.add(detailRecord[0])

    for idRemove in potentialRemove:
        if idRemove not in idInUse:
            labelRemove = globals.doneLabelActualDir + "/" + idRemove + ".label"
            binRemove = globals.doneVelDir + "/" + idRemove + ".bin"
            cylRemove = globals.doneLabelDir + "/cyl/" + idRemove + ".label"
            salRemove = globals.doneLabelDir + "/sal/" + idRemove + ".label"
            spvRemove = globals.doneLabelDir + "/spv/" + idRemove + ".label"
            deleteFiles.append(cylRemove)
            deleteFiles.append(salRemove)
            deleteFiles.append(spvRemove)
            deleteFiles.append(binRemove)
            deleteFiles.append(labelRemove)

    for file in deleteFiles:
        os.remove(file)

    return finalData



def runMutations(threadNum):
    
    finalData = prepFinalDetails()

    # Until signaled to end TODO
    for num in range (0, 5):

        mutationDetails = []
        bins = []
        labels = []

        # Mutate
        batchNum = 100
        for index in range(0, batchNum):
            print("\n\n{}".format((num * batchNum) + index))

            success, details, xyziFinal, labelFinal = performMutation()
            if success:
                mutationDetails.append(details)
                bins.append(xyziFinal)
                labels.append(labelFinal)

        # Save
        if (globals.saveMutationFlag):
            # Save folders
            saveVel = globals.stageDir + "/velodyne" + str(threadNum) + "/"
            saveLabel = globals.stageDir + "/labels" + str(threadNum) + "/"

            # Save bin and labels
            for index in range(0, len(mutationDetails)):
                fileIoUtil.saveToBin(bins[index], labels[index], saveVel, saveLabel, mutationDetails[index]["_id"])

            # Save mutation details
        
        # Evaluate
        if (globals.evalMutationFlag):
            details = eval.evalBatch(threadNum, mutationDetails)

            finalData = finalDetails(details, finalData)

        # Save details
        if (globals.saveMutationFlag):
            saveMutation(mutationDetails)


    print()
    print(json.dumps(finalData, indent=4))
    print()

    with open(globals.dataDir + '/finalData.json', 'w') as outfile:
        json.dump(finalData, outfile, indent=4)




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
    # p.add_argument("-save", 
    #     help="Where to save the sequences", 
    #     nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
    #     default="/home/garrett/Documents/data/dataset/sequences/")
    # p.add_argument('-m', required=False,
    #     help='Mutations to perform comma seperated example: ADD,REMOVE')

    p.add_argument('-m', required=False,
        help='Transformations to perform comma seperated example: ROTATE,ROTATE_MIRROR')

    p.add_argument("-t", 
        help="Thread number default to 1", 
        nargs='?', const=1, default=1)

    p.add_argument('-vis', help='Visualize with Open3D',
        action='store_true', default=False)

    p.add_argument('-ne', help='Disables Evaluation',
        action='store_false', default=True)
    p.add_argument('-ns', help='Disables Saving',
        action='store_false', default=True)

    p.add_argument("-assetId", 
        help="Asset Identifier", 
        nargs='?', const=None, default=None)
        
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


    # Start timer
    tic = time.perf_counter()

    try:
        runMutations(0)

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")


    # End timer
    toc = time.perf_counter()
    timeSeconds = toc - tic
    timeFormatted = formatSecondsToHhmmss(timeSeconds)

    print("Ran for {}".format(timeFormatted))


if __name__ == '__main__':
    main()



