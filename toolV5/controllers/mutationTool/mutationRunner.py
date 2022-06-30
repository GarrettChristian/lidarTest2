"""
semFuzzLidar 
Main runner for the mutation tool

@Author Garrett Christian
@Date 6/23/22
"""


import numpy as np
import open3d as o3d
import random
import shortuuid
import os
import time
import json

import data.assetRepository as assetRepository
import data.mutationDetailsRepository as mutationDetailsRepository
import data.finalDataRepository as finalDataRepository

import data.fileIoUtil as fileIoUtil

import domain.semanticMapping as semanticMapping
import domain.mutationsEnum as mutationsEnum

import service.pcd.pcdCommon as pcdCommon
import service.pcd.pcdDeform as pcdDeform
import service.pcd.pcdIntensity as pcdIntensity
import service.pcd.pcdRemove as pcdRemove
import service.pcd.pcdRotate as pcdRotate
import service.pcd.pcdScale as pcdScale
import service.pcd.pcdSignReplace as pcdSignReplace

import service.eval.finalDetails as finalDetailsClass
import service.eval.eval as eval


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


def performMutation(mutation, assetRepo, sessionManager):

    # Start timer for the mutation
    tic = time.perf_counter()

    
    # Create mutation details
    mutationId = str(shortuuid.uuid())
    details = {}
    details["_id"] = mutationId + "-" + mutation
    details["mutationId"] = mutationId
    details["time"] = int(time.time())
    details["dateTime"] = time.ctime(time.time())
    details["batchId"] = sessionManager.batchId
    details["mutation"] = mutation

    # mutation
    print(mutation)
    mutationSplit = mutation.split('_')
    # Asset location
    assetLocation =  mutationSplit[0]
    # Transformations
    mutationSet = set()
    for mutationComponent in mutationSplit:
        mutationSet.add(mutationComponent)


    # Get the Asset and Scene

    # Base:
    pcdArr, intensity, semantics, instances = None, None, None, None
    pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset = None, None, None, None
    assetRecord = None
    success = True
    combine = True

    # Adding asset to scene pick random sequence and scene as base
    if (assetLocation == mutationsEnum.AssetLocation.ADD.name):

        # Select base scene, given
        if (sessionManager.sequence != "" and sessionManager.scene != ""):
            details["baseSequence"] = sessionManager.seq
            details["baseScene"] = sessionManager.scene
            pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(sessionManager.binPath,  sessionManager.labelPath, 
                                                                                sessionManager.seq, sessionManager.scene)

        # Select base scene, random
        else:
            idx = random.choice(range(len(sessionManager.labelFiles)))
            print(sessionManager.binFiles[idx])
            head_tail = os.path.split(sessionManager.binFiles[idx])
            scene = head_tail[1]
            scene = scene.replace('.bin', '')
        
            head_tail = os.path.split(head_tail[0])
            head_tail = os.path.split(head_tail[0])
            sequence = head_tail[1]
            details["baseSequence"] = sequence
            details["baseScene"] = scene
            pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBinFiles(sessionManager.binFiles[idx], sessionManager.labelFiles[idx])

        # Select Asset, Given
        if (sessionManager.assetId != None):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getAssetById(sessionManager.assetId)
        # Select Asset, Random
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAsset()
            # pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetOfTypes([81])

    # Specific scene get asset then get the scene that asset is from
    elif (assetLocation == mutationsEnum.AssetLocation.SCENE.name or 
        assetLocation == mutationsEnum.AssetLocation.SIGN.name or
        assetLocation == mutationsEnum.AssetLocation.VEHICLE.name):

        if (sessionManager.assetId != None):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getAssetById(sessionManager.assetId)
        elif (assetLocation == mutationsEnum.AssetLocation.VEHICLE.name):
            # pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetOfTypesWithinScene(globals.vehicles, sequence, scene)
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAssetOfTypes(semanticMapping.instancesVehicle.keys())
        elif (assetLocation == mutationsEnum.AssetLocation.SIGN.name):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAssetOfTypes([81])
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAsset()
            # pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetWithinScene(sequence, scene)

        
        if (assetRecord != None):
            pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(sessionManager.binPath, sessionManager.labelPath, assetRecord["sequence"], assetRecord["scene"])
            pcdArr, intensity, semantics, instances = pcdCommon.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, instances)
            details["baseSequence"] = assetRecord["sequence"]
            details["baseScene"] = assetRecord["scene"]


    else:
        print("ERROR asset location: {} NOT SUPPORTED".format(assetLocation))
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
    
    # Perform the mutation

    for mutationIndex in range (1, len(mutationSplit)):
        if success:
        # if (Transformations.INTENSITY == transformation):
            if (mutationSplit[mutationIndex] == mutationsEnum.Transformation.INTENSITY.name):
                intensityAsset, details = pcdIntensity.intensityChange(intensityAsset, assetRecord["typeNum"], details)
                
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.DEFORM.name):
                pcdArrAsset, details = pcdDeform.deform(pcdArrAsset, details,
                                                        sessionManager.deformPoint, sessionManager.deformPercent, sessionManager.deformMu, 
                                                        sessionManager.deformSigma, sessionManager.deformSeed)
            
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.SCALE.name):
                success, pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, pcdArr, intensity, semantics, instances, details = pcdScale.scaleVehicle(pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                                                                                                            pcdArr, intensity, semantics, instances, details,
                                                                                                                                                            sessionManager.scaleLimit, sessionManager.scaleAmount) 

            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.REMOVE.name):
                success, pcdArr, intensity, semantics, instances = pcdRemove.replaceBasedOnShadow(pcdArrAsset, pcdArr, intensity, semantics, instances, details)
                # Don't combine if remove is the last transformation
                if mutationIndex + 1 == len(mutationSplit):
                    combine = False

            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.MIRROR.name):
                pcdArrAsset, details = pcdRotate.mirrorAsset(pcdArrAsset, sessionManager.mirrorAxis, details)

            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.ROTATE.name):
                success, pcdArrAsset, pcdArr, intensity, semantics, instances, details = pcdRotate.rotate(pcdArr, intensity, semantics, instances, pcdArrAsset, details)

            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.REPLACE.name):
                success, pcdArr, intensity, semantics, instances, pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, details = pcdSignReplace.signReplace(pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                                                                                                            pcdArr, intensity, semantics, instances, details)
            else:
                print("UNSUPPORTED TRANSFORMATION: {}".format(mutationSplit[mutationIndex]))


    # Combine the final results
    if success and combine:
        pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances, 
                                                                        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
       

    # Visualize the mutation if enabled
    if success and sessionManager.visualize:
        visualize(pcdArrAsset, pcdArr, intensity, semantics, mutationSet)


    # End timer for mutation
    toc = time.perf_counter()
    timeSeconds = toc - tic
    timeFormatted = formatSecondsToHhmmss(timeSeconds)
    print("Mutation took {}".format(timeFormatted))

    # New bin and modified label to return 
    xyziFinal = None
    labelFinal = None

    # Combine the xyz, intensity and semantics, instance labels labels and bins
    if (success):
        details["seconds"] = timeSeconds
        details["time"] = timeFormatted
        xyziFinal, labelFinal = fileIoUtil.prepareToSave(pcdArr, intensity, semantics, instances)

    return success, details, xyziFinal, labelFinal
    

"""
visualize
Uses open3d to visualize the specific mutation

Colors based on semantics
Or by intensity if inensity was altered
"""
def visualize(pcdArrAsset, pcdArr, intensity, semantics, mutationSet):
    # Get asset box
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAsset)
    # hull, _ = pcdAsset.compute_convex_hull()
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    # hull_ls.paint_uniform_color((0, 0, 1))
    obb = pcdAsset.get_oriented_bounding_box()
    obb.color = (0.7, 0.7, 0.7)
    # Get scene
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

    # Color as intensity or label
    colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
    if (mutationsEnum.Transformation.INTENSITY.name in mutationSet):
        colors[:, 2] = intensity
    else:
        for semIdx in range(0, len(semantics)):
            colors[semIdx][0] = (semanticMapping.color_map_alt[semantics[semIdx]][0] / 255)
            colors[semIdx][1] = (semanticMapping.color_map_alt[semantics[semIdx]][1] / 255)
            colors[semIdx][2] = (semanticMapping.color_map_alt[semantics[semIdx]][2] / 255)
    pcdScene.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([obb, pcdScene])




def runMutations(sessionManager):

    assetRepo = assetRepository.AssetRepository(sessionManager.binPath, sessionManager.binPath, sessionManager.mongoConnect)
    detailsRepo = mutationDetailsRepository.DetailsRepository(sessionManager.mongoConnect)
    finalDataRepo = finalDataRepository.FinalDataRepository(sessionManager.mongoConnect)

    threadNum = 0
    
    finalData = finalDetailsClass.FinalDetails(sessionManager.batchId, 10, sessionManager.mutationsEnabled, 
                                                sessionManager.doneVelDir, sessionManager.doneLabelDir)

    errors = []

    # Start timer for tool
    ticAll = time.perf_counter() 

    # for num in range (0, globals.iterationNum):
    attemptedNum = 0
    successNum = 0
    while (successNum < sessionManager.expectedNum):

        # Start timer for batch
        tic = time.perf_counter()

        mutationDetails = []
        bins = []
        labels = []

        # Mutate
        # for index in range(0, globals.batchNum):
        batchCount = 0
        while(batchCount < sessionManager.batchNum and successNum < sessionManager.expectedNum):
            mutationEnum = random.choice(sessionManager.mutationsEnabled)
            mutation = mutationEnum.name
            attemptedNum += 1
            print("\n\nAttempt {}. [curr successful {}]".format(attemptedNum, successNum))

            success = False
            success, details, xyziFinal, labelFinal = performMutation(mutation, assetRepo, sessionManager)
            # try:
            #     success, details, xyziFinal, labelFinal = performMutation()
            # except Exception as e:
            #     print("\n\n\n ERROR IN PERFORM MUTATION \n\n\n")
            #     print(e)
            #     print("\n\n")
            #     errors.append(e)

            if success:
                batchCount += 1
                successNum += 1
                mutationDetails.append(details)
                bins.append(xyziFinal)
                labels.append(labelFinal)

        # Save
        if (sessionManager.saveMutationFlag):
            # Save folders
            saveVel = sessionManager.stageDir + "/velodyne" + str(threadNum) + "/"
            saveLabel = sessionManager.stageDir + "/labels" + str(threadNum) + "/"

            # Save bin and labels
            for index in range(0, len(mutationDetails)):
                fileIoUtil.saveToBin(bins[index], labels[index], saveVel, saveLabel, mutationDetails[index]["_id"])

            # Save mutation details
        
        # Evaluate
        if (sessionManager.evalMutationFlag):
            details = eval.evalBatch(threadNum, mutationDetails, sessionManager)
            deleteFiles = finalData.updateFinalDetails(details)
            fileIoUtil.removeFiles(deleteFiles)

        # Save details
        if (sessionManager.saveMutationFlag):
            # detailsToSave = []
            # for detail in mutationDetails:
            #     buckets = 0
            #     for model in globals.models:
            #         buckets += detail[model]["bucketA"]
            #         buckets += detail[model]["bucketJ"]
                        
            #     if (buckets > 0):
            #         detailsToSave.append(detail)

            # if (len(detailsToSave) > 0):
            detailsRepo.saveMutationDetails(mutationDetails)


        # End timer for batch
        toc = time.perf_counter()
        timeSeconds = toc - tic
        timeFormatted = formatSecondsToHhmmss(timeSeconds)
        print("Batch took {}".format(timeFormatted))


    # Final Items

    # End timer
    tocAll = time.perf_counter()
    timeSeconds = tocAll - ticAll
    timeFormatted = formatSecondsToHhmmss(timeSeconds)
    

    finalData.setTime(timeSeconds, timeFormatted)
    finalData.setAttempts(successNum, attemptedNum)
    if (sessionManager.evalMutationFlag):
        finalData.finalizeFinalDetails()
        finalDataRepo.saveFinalData(finalData.finalData)

    # Output final data
    print()
    print(json.dumps(finalData.finalData, indent=4))
    print()
    print("Ran for {}".format(timeFormatted))


    # Save final data
    if (sessionManager.saveMutationFlag):
        with open(sessionManager.dataDir + '/finalData.json', 'w') as outfile:
            json.dump(finalData.finalData, outfile, indent=4, sort_keys=True)

    # If caching errors
    for idx in range(0, len(errors)):
        print("\n")
        print("Error {}".format(idx))
        print(errors[idx])
        print("\n")






