


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

import mongoUtil
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
    pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBinFiles(globals.binFiles[idx], globals.labelFiles[idx])
    pcdArrAsset = None
    intensityAsset = None
    semanticsAsset = None
    instancesAsset = None
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
        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getAssetById(globals.assetId)

        if (assetRecord != None and "REMOVE" in mutationSet
            and assetRecord["sequence"] == sequence and assetRecord["scene"] == scene):
            pcdArr, intensity, semantics, instances = pcdUtil.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, instances)
        
    elif (assetLocation == "ADD"):
        if ("DEFORM" in mutationSet):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetOfTypes(globals.vehicles)
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAsset()
            # pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetOfTypes([30])

    elif (assetLocation == "SCENE"):
        if ("DEFORM" in mutationSet or "SCALE" in mutationSet):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetOfTypesWithinScene(globals.vehicles, sequence, scene)
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetWithinScene(sequence, scene)

        
        if (assetRecord != None):
            pcdArr, intensity, semantics, instances = pcdUtil.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, instances)
    
    elif (assetLocation == "SIGN"):
        success, pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset = pcdUtil.getSignAsset(pcdArr, intensity, semantics, instances)
        assetRecord = {}
        assetRecord["_id"] = "sign"
        assetRecord["type"] = "sign"
        assetRecord["typeNum"] = "sign"
        assetRecord["sequence"] = sequence
        assetRecord["scene"] = scene
        if (success):
            pcdArr, intensity, semantics, instances = pcdUtil.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, instances)

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
        details["assetSequence"] = assetRecord["sequence"]
        details["assetScene"] = assetRecord["scene"]
        details["assetType"] = assetRecord["type"]
        details["typeNum"] = assetRecord["typeNum"]
    

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
                intensityAsset, details = pcdUtil.intensityChange(intensityAsset, assetRecord["typeNum"], details)
                
            elif (mutationSplit[mutationIndex] == "DEFORM"):
                pcdArrAsset, details = pcdUtil.deform(pcdArrAsset, details)
            
            elif (mutationSplit[mutationIndex] == "SCALE"):
                success, pcdArr, intensity, semantics, instances, pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, details = pcdUtil.scaleVehicle(pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                                                                                                            pcdArr, intensity, semantics, instances, details) 

            elif (mutationSplit[mutationIndex] == "REMOVE"):
                success, pcdArr, intensity, semantics, instances = pcdUtil.replaceBasedOnShadow(pcdArrAsset, pcdArr, intensity, semantics, instances)
                if mutationIndex + 1 == len(mutationSplit):
                    combine = False

            elif (mutationSplit[mutationIndex] == "MIRROR"):
                axis = globals.mirrorAxis
                if (not axis):
                    axis = random.randint(0, 1)
                print("Mirror Axis: {}".format(axis))
                details["mirror"] = axis
                pcdArrAsset[:, axis] = pcdArrAsset[:, axis] * -1

            elif (mutationSplit[mutationIndex] == "ROTATE"):
                success, pcdArrAsset, pcdArr, intensity, semantics, instances, details = pcdUtil.rotate(pcdArr, intensity, semantics, instances, pcdArrAsset, details)

            elif (mutationSplit[mutationIndex] == "REPLACE"):
                success, pcdArr, intensity, semantics, instances, pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, details = pcdUtil.signReplace(pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                                                                                                            pcdArr, intensity, semantics, instances, details)

            else:
                print("NOT SUPPORTED {}".format(mutationSplit[mutationIndex]))


    if success and combine:
        pcdArr, intensity, semantics, instances = pcdUtil.combine(pcdArr, intensity, semantics, instances, 
                                                                        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
       

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

        # Color as intensity or label
        colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
        if ("INTENSITY" in mutationSet):
            colors[:, 2] = intensity
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
        xyziFinal, labelFinal = fileIoUtil.prepareToSave(pcdArr, intensity, semantics, instances)

    
    return success, details, xyziFinal, labelFinal
    









def runMutations(threadNum):
    
    finalData = eval.prepFinalDetails()

    errors = []

    # Start timer for tool
    ticAll = time.perf_counter()


    for num in range (0, globals.iterationNum):

        # Start timer for batch
        tic = time.perf_counter()

        mutationDetails = []
        bins = []
        labels = []

        # Mutate
        for index in range(0, globals.batchNum):
            print("\n\n{}".format((num * globals.batchNum) + index))

            success = False
            success, details, xyziFinal, labelFinal = performMutation()
            # try:
            #     success, details, xyziFinal, labelFinal = performMutation()
            # except Exception as e:
            #     print("\n\n\n ERROR IN PERFORM MUTATION \n\n\n")
            #     print(e)
            #     print("\n\n")
            #     errors.append(e)

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

            finalData = eval.finalDetails(details, finalData)

        # Save details
        if (globals.saveMutationFlag):
            mongoUtil.saveMutationDetails(mutationDetails)


        # End timer for batch
        toc = time.perf_counter()
        timeSeconds = toc - tic
        timeFormatted = formatSecondsToHhmmss(timeSeconds)
        print("Batch took {}".format(timeFormatted))


    print()
    print(json.dumps(finalData, indent=4))
    print()

    # End timer
    tocAll = time.perf_counter()
    timeSeconds = tocAll - ticAll
    timeFormatted = formatSecondsToHhmmss(timeSeconds)
    print("Ran for {}".format(timeFormatted))
    finalData["seconds"] = timeSeconds
    finalData["time"] = timeFormatted


    # Save final data
    if (globals.saveMutationFlag):
        with open(globals.dataDir + '/finalData.json', 'w') as outfile:
            json.dump(finalData, outfile, indent=4, sort_keys=True)


    for idx in range(0, len(errors)):
        print("\n")
        print("Error {}".format(idx))
        print(errors[idx])
        print("\n")



# -------------------------------------------------------------


# Boot Util


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    p.add_argument("-seq", 
        help="Sequences number, provide as 1 rather than 01 (default all labeled 0-10)", 
        nargs='?', const=0, default=range(0, 11))
    p.add_argument( "-scene", 
        help="specific scenario number provide full ie 002732")
    p.add_argument("-path", 
        help="Path to the sequences", 
        nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
        default="/home/garrett/Documents/data/dataset/sequences/")

    p.add_argument('-m', required=False,
        help='Transformations to perform comma seperated example: ROTATE,ROTATE_MIRROR')

    p.add_argument("-t", 
        help="Thread number default to 1", 
        nargs='?', const=1, default=1)
    p.add_argument("-b", 
        help="Batch to create before evaluating", 
        nargs='?', const=100, default=100)
    p.add_argument("-i", 
        help="Iteration of batches", 
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
    
    mongoUtil.mongoConnect()


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


   


if __name__ == '__main__':
    main()



