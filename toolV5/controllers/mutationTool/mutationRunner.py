"""
semFuzzLidar 
Main runner for the mutation tool

@Author Garrett Christian
@Date 6/23/22
"""


from glob import glob
import numpy as np
import open3d as o3d
import random
import shortuuid
import os
import time
import json

# import threading
# from multiprocessing import Process
# from multiprocessing import Queue
import multiprocessing as mp

from data.assetRepository import AssetRepository
from data.mutationDetailsRepository import DetailsRepository
from data.finalDataRepository import FinalDataRepository

import data.fileIoUtil as fileIoUtil

import domain.semanticMapping as semanticMapping
import domain.mutationsEnum as mutationsEnum
from domain.modelConstants import models

import service.pcd.pcdCommon as pcdCommon
import service.pcd.pcdDeform as pcdDeform
import service.pcd.pcdIntensity as pcdIntensity
import service.pcd.pcdRemove as pcdRemove
import service.pcd.pcdRotate as pcdRotate
import service.pcd.pcdScale as pcdScale
import service.pcd.pcdSignReplace as pcdSignReplace

from service.eval.finalDetails import FinalDetails
import service.eval.eval as eval


# -------------------------------------------------------------


successNum = 0
attemptedNum = 0
batchCount = 0
potentialSuccess = 0
# finalData = None


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
    details["epochTime"] = int(time.time())
    details["dateTime"] = time.ctime(time.time())
    details["batchId"] = sessionManager.batchId
    details["mutation"] = mutation

    # mutation
    # print(mutation)
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
    modelPredictionsScene = {}
    modelPredictionsAsset = {}

    # NON SCENE SPECIFIC
    # Adding asset to scene pick random sequence and scene as base
    if (assetLocation == mutationsEnum.AssetLocation.ADD.name):

        # Select base scene, given
        if (sessionManager.scene != None):
            details["baseSequence"] = sessionManager.sequence
            details["baseScene"] = sessionManager.scene
            pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(sessionManager.binPath,  sessionManager.labelPath, 
                                                                                sessionManager.sequence, sessionManager.scene)

        # Select base scene, random
        else:
            idx = random.choice(range(len(sessionManager.labelFiles)))
            # print(sessionManager.binFiles[idx])
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

        if (assetRecord != None):
            # Get model prediction (asset & scene)
            _, instanceAssetScene = fileIoUtil.openLabel(sessionManager.labelPath, assetRecord["sequence"], assetRecord["scene"])
            for model in models:
                modelPredictionsScene[model] = fileIoUtil.openModelPredictions(sessionManager.basePredictionPath, model, details["baseSequence"], details["baseScene"])
                modelAssetScene = fileIoUtil.openModelPredictions(sessionManager.basePredictionPath, model, assetRecord["sequence"], assetRecord["scene"])
                modelPredictionsAsset[model] = modelAssetScene[instanceAssetScene == assetRecord["instance"]]

    # SCENE SPECIFIC
    # Specific scene get asset then get the scene that asset is from
    elif (assetLocation == mutationsEnum.AssetLocation.SCENE.name or 
        assetLocation == mutationsEnum.AssetLocation.SIGN.name or
        assetLocation == mutationsEnum.AssetLocation.VEHICLE.name):

        # GIVEN ASSET ID (OPTIONAL)
        if (sessionManager.assetId != None):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getAssetById(sessionManager.assetId)

        # REMOVE OPTION TO TRY ALL ASSETS
        elif(sessionManager.removeAll):
            sessionManager.removeAllNum += 1
            assetRecordResult = assetRepo.getAssetsPaged(sessionManager.removeAllNum, 1)

            assetRecordList = list(assetRecordResult)
            if (len(assetRecordList) > 0):
                assetRecord = assetRecordList[0]
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord =  assetRepo.getInstanceFromAssetRecord(assetRecord)
            # if there are no more assets to try, but haven't hit the expected number, short ciruit
            else:
                sessionManager.expectedNum = successNum


        # ONLY VEHICLE ASSETS
        elif (assetLocation == mutationsEnum.AssetLocation.VEHICLE.name):
            # pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetOfTypesWithinScene(globals.vehicles, sequence, scene)
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAssetOfTypes(semanticMapping.instancesVehicle.keys())
        

        # ONLY SIGN ASSETS
        elif (assetLocation == mutationsEnum.AssetLocation.SIGN.name):
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAssetOfType(81)
        
        # ANY ASSET
        else:
            pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = assetRepo.getRandomAsset()
            # pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, assetRecord = mongoUtil.getRandomAssetWithinScene(sequence, scene)

        
        if (assetRecord != None):
            pcdArr, intensity, semantics, instances = fileIoUtil.openLabelBin(sessionManager.binPath, sessionManager.labelPath, assetRecord["sequence"], assetRecord["scene"])
            instancesBeforeRemoval = np.copy(instances)
            pcdArr, intensity, semantics, instances, sceneMask = pcdCommon.removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, instances)
            details["baseSequence"] = assetRecord["sequence"]
            details["baseScene"] = assetRecord["scene"]

            # Get model prediction (asset & scene)
            for model in models:
                modelPredictionsScene[model] = fileIoUtil.openModelPredictions(sessionManager.basePredictionPath, model, details["baseSequence"], details["baseScene"])
                modelPredictionsAsset[model] = modelPredictionsScene[model][instancesBeforeRemoval == assetRecord["instance"]]
                # Remove asset from the scene 
                modelPredictionsScene[model] = modelPredictionsScene[model][sceneMask]
                # print("{} {} {} {}".format(np.shape(semantics), np.shape(modelPredictionsScene[model]), np.shape(semanticsAsset), np.shape(modelPredictionsAsset[model])))


    else:
        print("ERROR asset location: {} NOT SUPPORTED".format(assetLocation))


    
    # Validate the asset was found
    if assetRecord == None:
        print("Invalid Asset / No asset found")
        success = False
    else:
        # print(assetRecord)
        details["asset"] = assetRecord["_id"]
        details["assetSequence"] = assetRecord["sequence"]
        details["assetScene"] = assetRecord["scene"]
        details["assetType"] = assetRecord["type"]
        details["typeNum"] = assetRecord["typeNum"]
        details["assetPoints"] = assetRecord["points"]
        # Points are collected for weight average on add, if sign is taken only the sign point are averaged, not the pole
        if (assetRecord["typeNum"] == 81):
            details["assetPoints"] = int(np.shape(pcdArrAsset[semanticsAsset == 81])[0])





    # Perform the mutation

    for mutationIndex in range (1, len(mutationSplit)):
        if success:
            # INTENSITY
            if (mutationSplit[mutationIndex] == mutationsEnum.Transformation.INTENSITY.name):
                intensityAsset, details = pcdIntensity.intensityChange(intensityAsset, details, sessionManager.intensityChange)
                
            # DEFORM
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.DEFORM.name):
                pcdArrAsset, details = pcdDeform.deform(pcdArrAsset, details,
                                                        sessionManager.deformPoint, sessionManager.deformPercent, sessionManager.deformMu, 
                                                        sessionManager.deformSigma, sessionManager.deformSeed)
            
            
            # SCALE
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.SCALE.name):
                success, assetResult, sceneResult, details, modelPredictionsScene, modelPredictionsAsset = pcdScale.scaleVehicle(pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                                                                                pcdArr, intensity, semantics, instances, details,
                                                                                                                                sessionManager.scaleLimit, sessionManager.scaleAmount,
                                                                                                                                modelPredictionsScene, modelPredictionsAsset) 
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset = assetResult
                pcdArr, intensity, semantics, instances = sceneResult


            # REMOVE
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.REMOVE.name):
                success, pcdArr, intensity, semantics, instances, details, modelPredictionsScene = pcdRemove.replaceBasedOnShadow(pcdArrAsset, pcdArr, intensity, semantics, instances, 
                                                                                                                                details, modelPredictionsScene)
                # Don't combine if remove is the last transformation
                if mutationIndex + 1 == len(mutationSplit):
                    combine = False

            # MIRROR
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.MIRROR.name):
                pcdArrAsset, details = pcdRotate.mirrorAsset(pcdArrAsset, details, sessionManager.mirrorAxis)

            # ROTATE
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.ROTATE.name):
                success, pcdArrAsset, sceneResult, details, modelPredictionsScene = pcdRotate.rotate(pcdArr, intensity, semantics, instances, 
                                                                                                    pcdArrAsset, 
                                                                                                    details, sessionManager.rotation, 
                                                                                                    modelPredictionsScene)
                pcdArr, intensity, semantics, instances = sceneResult


            # REPLACE
            elif (mutationSplit[mutationIndex] == mutationsEnum.Transformation.REPLACE.name):
                success, sceneResult, assetResult, details, modelPredictionsScene, modelPredictionsAsset = pcdSignReplace.signReplace(pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                                                                            pcdArr, intensity, semantics, instances, 
                                                                                                                            details, sessionManager.signChange,
                                                                                                                            modelPredictionsScene, modelPredictionsAsset)
                pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset = assetResult
                pcdArr, intensity, semantics, instances = sceneResult    


            else:
                print("UNSUPPORTED TRANSFORMATION: {}".format(mutationSplit[mutationIndex]))
                success = False


    # Combine the final results
    if success and combine:
        pcdArr, intensity, semantics, instances = pcdCommon.combine(pcdArr, intensity, semantics, instances, 
                                                                        pcdArrAsset, intensityAsset, semanticsAsset, instancesAsset)
        # Combine model results
        for model in models:
            modelPredictionsScene[model] = np.hstack((modelPredictionsScene[model], modelPredictionsAsset[model]))
       

    # Visualize the mutation if enabled
    if success and sessionManager.visualize:
        visualize(pcdArrAsset, pcdArr, intensity, semantics, mutationSet)


    # End timer for mutation
    toc = time.perf_counter()
    timeSeconds = toc - tic
    timeFormatted = formatSecondsToHhmmss(timeSeconds)
    # print("Mutation took {}".format(timeFormatted))


    # Combine the xyz, intensity and semantics, instance labels labels and bins
    if (success):
        details["seconds"] = timeSeconds
        details["mutationTime"] = timeFormatted
        if (sessionManager.saveMutationFlag):
            # Save mutated bin and update label file
            saveVel = sessionManager.stageDir + "/"
            saveLabel = sessionManager.doneLabelDir + "/"
            fileIoUtil.saveBinLabelPair(pcdArr, intensity, semantics, instances, saveVel, saveLabel, details["_id"])
            # Save updated model predictions
            for model in models:
                saveAtModel = sessionManager.doneMutatedPredDir + "/" + model + "/"
                fileIoUtil.saveLabelSemantics(modelPredictionsScene[model], saveAtModel, details["_id"])
                

    return success, details
    

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
            colors[semIdx][0] = (semanticMapping.color_map_alt_rgb[semantics[semIdx]][0] / 255)
            colors[semIdx][1] = (semanticMapping.color_map_alt_rgb[semantics[semIdx]][1] / 255)
            colors[semIdx][2] = (semanticMapping.color_map_alt_rgb[semantics[semIdx]][2] / 255)
    pcdScene.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([obb, pcdScene])



# def batchMutation(threadNum, mutex, sessionManager, mutationDetails, bins, labels, assetRepo):
def batchMutation(threadNum, sessionManager, mutationDetails, assetRepo):
    global successNum
    global attemptedNum
    global batchCount
    global potentialSuccess

    # mutex.acquire()
    print("Starting {} Thread Num, saving at {}".format(threadNum, sessionManager.stageDir))

    while(batchCount + potentialSuccess < sessionManager.batchNum and successNum + potentialSuccess < sessionManager.expectedNum):
        potentialSuccess += 1
        # mutex.release()

        mutationEnum = random.choice(sessionManager.mutationsEnabled)
        mutation = mutationEnum.name
        
        success = False
        success, details = performMutation(mutation, assetRepo, sessionManager)

        # mutex.acquire()
        potentialSuccess -= 1
        attemptedNum += 1
        if success:
            print("\n\nThread {}, Attempt {} (successful) [curr successful {}]".format(threadNum, attemptedNum, successNum))
            print(details)
            batchCount += 1
            successNum += 1
            mutationDetails.append(details)
            # bins.append(xyziFinal)
            # labels.append(labelFinal)
        else:
            print("\n\nThread {}, Attempt {} (unsuccessful) [curr successful {}]".format(threadNum, attemptedNum, successNum))
            print(details)
    
    # mutex.release()



def evalBatch(sessionManager, mutationDetails, finalData, queue, complete, total):
    # Catch for 0 mutations
    if (len(mutationDetails) < 1):
        return

    # PyMongo not process safe
    # Need to recreate here
    detailsRepo = DetailsRepository(sessionManager.mongoConnect)

    # Evaluate
    if (sessionManager.evalMutationFlag):
        details = eval.evalBatch(mutationDetails, sessionManager,  complete, total)
        deleteFiles = finalData.updateFinalDetails(details)
        
        if (not sessionManager.saveAll):
            fileIoUtil.removeFiles(deleteFiles)

    # Save details
    if (sessionManager.saveMutationFlag):
        detailsRepo.saveMutationDetails(mutationDetails)
    
    print("Put Final Data")
    ret = {}
    ret['finalData'] = finalData
    queue.put(ret)
    print("Eval batch done")




def runMutations(sessionManager):
    global successNum
    global attemptedNum
    global batchCount
    global potentialSuccess
    # global finalData

    assetRepo = AssetRepository(sessionManager.binPath, sessionManager.labelPath, sessionManager.mongoConnect)
    detailsRepo = DetailsRepository(sessionManager.mongoConnect)
    finalDataRepo = FinalDataRepository(sessionManager.mongoConnect)

    threadNum = 0
    threadCount = 3
    
    finalData = FinalDetails(sessionManager.batchId, topNum=10, doneVelDir=sessionManager.doneVelDir, doneLabelDir=sessionManager.doneLabelDir)

    errors = []



    mp.set_start_method('forkserver')
    mpManager = mp.Manager()

   

    
    # muti processing 
    ret = {}
    ret["finalData"] = finalData
    queue = mpManager.Queue()
    queue.put(ret)


    # Start timer for tool
    ticAll = time.perf_counter() 

    attemptedNum = 0
    successNum = 0
    processList = []
    while (successNum < sessionManager.expectedNum):

        # Start timer for batch
        tic = time.perf_counter()

        mutationDetails = []
        bins = []
        labels = []

        # threadList = []
        batchCount = 0
        # potentialSuccess = 0
        # mutex = threading.Lock()
        # for n in range(threadCount):
        #     thread = threading.Thread(target=batchMutation, args=(n, mutex, sessionManager, labels, assetRepo,))
        #     threadList.append(thread)
        #     thread.start()

        # for thread in threadList:
        #     thread.join()

        threadNum = (threadNum + 1) % 2

        batchMutation(0, sessionManager, mutationDetails, assetRepo)

        # End timer for mutation batch generation
        toc = time.perf_counter()
        timeSeconds = toc - tic
        timeFormatted = formatSecondsToHhmmss(timeSeconds)
        print("Batch took {}".format(timeFormatted))

        # Mutate
        # for index in range(0, globals.batchNum):

        
        if (sessionManager.asyncEval):
            print("\n\nJoin Process {}", len(processList))
            for process in processList:
                process.join()
            print("Get queue\n\n")
            ret = queue.get()
            finalData = ret["finalData"]
            print("Starting evaluation Success: {} / {}".format(successNum, sessionManager.expectedNum))
            # thread = threading.Thread(target=evalBatch, args=(sessionManager, mutationDetails, bins, labels, evalNum, detailsRepo))
            process = mp.Process(target=evalBatch, args=(sessionManager, mutationDetails, finalData, queue, successNum, sessionManager.expectedNum))
            # processList = [process]
            process.start()
        else:
            evalBatch(sessionManager, mutationDetails, finalData, queue, successNum, sessionManager.expectedNum)

        # evalBatch(sessionManager, mutationDetails)

    if (sessionManager.asyncEval):
        # Wait on last Eval batch
        print("Join Process last")
        for process in processList:
            process.join()
        print("Get queue last")
        ret = queue.get()
        finalData = ret["finalData"]

        


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






