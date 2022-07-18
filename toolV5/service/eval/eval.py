"""
eval 
Handles evaluation of predictions files, runs models, updates the final analytics 

@Author Garrett Christian
@Date 6/23/22
"""


import glob, os
import numpy as np
from os.path import basename
import shutil
from operator import itemgetter
import time
import subprocess

import service.eval.ioueval as ioueval
import service.eval.modelRunner as modelRunner

import data.baseAccuracyRepository as baseAccuracyRepository
import data.fileIoUtil as fileIoUtil

from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import learning_map
from domain.semanticMapping import learning_ignore
from domain.semanticMapping import name_label_mapping
from domain.modelConstants import models

 
# --------------------------------------------------------------------------
# Constants for the ioueval evaluator

numClasses = len(learning_map_inv)

# make lookup table for mapping
maxkey = max(learning_map.keys())

# +100 hack making lut bigger just in case there are unknown labels
remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
remap_lut[list(learning_map.keys())] = list(learning_map.values())

# create evaluator
ignore = []
for cl, ign in learning_ignore.items():
    if ign:
        x_cl = int(cl)
        ignore.append(x_cl)

evaluator = ioueval.iouEval(numClasses, ignore)


# --------------------------------------------------------------------------
# Evaluators


def evalLabels(modifiedLabel, modifiedPrediction, newPrediction):

    # Set up results
    results = {}
    results["mod"] = {}
    results["mod"]["classes"] = {}
    results["new"] = {}
    results["new"]["classes"] = {}

    accMod, jaccMod, classJaccMod = evaluateLabelPred(modifiedLabel, modifiedPrediction)
    accNew, jaccNew, classJaccNew = evaluateLabelPred(modifiedLabel, newPrediction)

    # save classwise jaccard
    for i, jacc in enumerate(classJaccMod):
        if i not in ignore:
            results["mod"][name_label_mapping[learning_map_inv[i]]] = jacc
    for i, jacc in enumerate(classJaccNew):
        if i not in ignore:
            results["new"][name_label_mapping[learning_map_inv[i]]] = jacc
    # Save acc
    results["mod"]["accuracy"] = accMod
    results["new"]["accuracy"] = accNew
    # Save jacc
    results["mod"]["jaccard"] = jaccMod
    results["new"]["jaccard"] = jaccNew
    
    # Get percent loss
    results["accuracyChange"] = accNew - accMod
    results["jaccardChange"] = jaccNew - jaccMod
    results["percentLossAcc"] = results["accuracyChange"] * 100
    results["percentLossJac"] = results["jaccardChange"] * 100

    return results
    


def evaluateLabelPred(labelFile, predictionFile):
    global evaluator

    groundTruth, _ = fileIoUtil.openLabelFile(labelFile)
    prediction, _ = fileIoUtil.openLabelFile(predictionFile)

    # Map to correct classes for evaluation,
    # Example classification "moving-car" -> "car"
    groundTruthMapped = remap_lut[groundTruth] # remap to xentropy format
    predictionMapped = remap_lut[prediction] # remap to xentropy format

    # Prepare evaluator
    evaluator.reset()
    evaluator.addBatch(predictionMapped, groundTruthMapped)
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    return m_accuracy, m_jaccard, class_jaccard



"""
Evaluates a label file against a prediction file
Modified from SemanticKITTI Development Kit 
https://github.com/PRBonn/semantic-kitti-api
https://github.com/PRBonn/semantic-kitti-api/blob/master/evaluate_semantics.py

@param label_file to evaluate as ground truth
@param pred_file to evaluate against the ground truth
@param model string that created the predictions
@param details dictionary that enumerates what occured in this transformation
@return results with accuracy results for this model and pair of labels
"""
def evalLabelsOld(label_file, pred_file, baseAccuracy, baseAccuracyAsset, typeNum, mutation, assetPoints):

    # print()
    # print(label_file)
    # print(pred_file)



    numClasses = len(learning_map_inv)

    # make lookup table for mapping
    maxkey = max(learning_map.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(learning_map.keys())] = list(learning_map.values())

    # create evaluator
    ignore = []
    for cl, ign in learning_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            # print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    evaluator = ioueval.iouEval(numClasses, ignore)
    evaluator.reset()

    label = np.fromfile(label_file, dtype=np.int32)
    label = label.reshape((-1))  # reshape to vector
    label = label & 0xFFFF       # get lower half for semantics

    # open prediction
    pred = np.fromfile(pred_file, dtype=np.int32)
    pred = pred.reshape((-1))    # reshape to vector
    pred = pred & 0xFFFF         # get lower half for semantics

    label = remap_lut[label] # remap to xentropy format
    pred = remap_lut[pred] # remap to xentropy format

    # add single scan to evaluation
    evaluator.addBatch(pred, label)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, m_jaccard_modified, class_jaccard = evaluator.getIoU()

    results = {}

    results["jaccard"] = m_jaccard.item()
    results["jaccardAlt"] = m_jaccard_modified.item()
    results["accuracy"] = m_accuracy.item()

    # collect classwise jaccard
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            results[name_label_mapping[learning_map_inv[i]]] = jacc

    jacChange = results["jaccard"] - baseAccuracy["jaccard"]
    accChange = results["accuracy"] - baseAccuracy["accuracy"]
    
    results["jaccardChangeAlt"] = results["jaccardAlt"] - baseAccuracy["jaccardAlt"]
    results["jaccardChange"] = jacChange
    results["accuracyChange"] = accChange

    # Get set of classes found within the label
    classesInLabel = set(label)

    # Asset accuracy change (ADD ONLY)
    if "ADD" in mutation:

        typeNumReduced = learning_map_inv[learning_map[typeNum]]
        typeNameAsset = name_label_mapping[typeNumReduced]
        typeJacChange = results[typeNameAsset] - baseAccuracyAsset[typeNameAsset]
        results["jaccardChangeAsset"] = typeJacChange

        results["22typeNumReduced"] = typeNumReduced
        results["22typeNameAsset"] = typeNameAsset
        results["22baseAccuracyAsset"] = baseAccuracyAsset[typeNameAsset]

        totalClasses = 0
        jaccardAdd = 0
        for classNum in learning_map_inv.keys():
            if classNum not in ignore:
                totalClasses += 1
                classNumInv = learning_map_inv[classNum]
                className = name_label_mapping[classNumInv]
                baseClassJacc = baseAccuracy[className]
                # Do a weighted average between the two possible options for the class 
                if (classNum == learning_map[typeNum]):
                    # Replace the asset you added to the scene with the original guess for that class
                    # So if you added a person and originally in the base there was a person 
                    # this should scale if you guessed the class originally 
                    classPoints = int(np.sum(label == classNum))
                    classPointsWithoutAsset = classPoints - assetPoints
                    weightedAvgOfClass = (baseAccuracyAsset[typeNameAsset] * assetPoints + baseClassJacc * classPointsWithoutAsset) / (classPoints)
                    jaccardAdd += weightedAvgOfClass
                    results["2" + className] = weightedAvgOfClass
                elif (classNum in classesInLabel):
                    jaccardAdd += baseClassJacc
                    results["2" + className] = baseClassJacc
                else:
                # Add 0 if not found in new label class, this could happen if you cover the only person 
                    results["2" + className] = 0

                

        jaccardAddBase = jaccardAdd / totalClasses
        results["jaccardAddBase"] = jaccardAddBase
        results["jaccardChangeOrig"] = jacChange
        results["jaccardChange"] = results["jaccard"] - jaccardAddBase


    if "REMOVE" in mutation:

        totalClasses = 0
        jaccardRemove = 0
        
        for classNum in learning_map_inv.keys():
            if classNum not in ignore:
                totalClasses += 1
                classNumInv = learning_map_inv[classNum]
                className = name_label_mapping[classNumInv]
                print(className)
                baseClassJacc = baseAccuracy[className]
                if (classNum in classesInLabel):
                    print(className)
                    jaccardRemove += baseClassJacc
                # Add 0 if not found in new label class, this could happen if you remove the only person 
        
        jaccardRemoveBase = jaccardRemove / totalClasses
        results["jaccardRemoveBase"] = jaccardRemoveBase
        results["jaccardChangeOrig"] = jacChange
        results["jaccardChange"] = results["jaccard"] - jaccardRemoveBase


        
        
        


    # Bucketing
    percentLossAcc = results["accuracyChange"] * 100

    bucketA = 0 # percentLoss >= 0.1 %
    if (percentLossAcc < -5):
        bucketA = 5
    elif (percentLossAcc < -3):
        bucketA = 4
    elif (percentLossAcc < -2):
        bucketA = 3
    elif (percentLossAcc < -1):
        bucketA = 2
    elif (percentLossAcc < -0.1):
        bucketA = 1

    results["percentLossAcc"] = percentLossAcc
    results["bucketA"] = bucketA

    percentLossJac = results["jaccardChange"] * 100

    bucketJ = 0 # percentLoss >= 0.1 %
    if (percentLossJac < -5):
        bucketJ = 5
    elif (percentLossJac < -3):
        bucketJ = 4
    elif (percentLossJac < -2):
        bucketJ = 3
    elif (percentLossJac < -1):
        bucketJ = 2
    elif (percentLossJac < -0.1):
        bucketJ = 1

    results["percentLossJac"] = percentLossJac
    results["bucketJ"] = bucketJ


    return results
    

# --------------------------------------------------------------------------

"""
Evaluates a given set of mutations
Note all bins and labels must be in:
/data
    /staging
        /velodyne0
        /labels0

@param threadNum integer that shows where to get and put the results
@param details list of detail dictionarys that enumerates what occured in this transformation
@return details updated with the results from the models
"""
def evalBatch(details, sessionManager,  complete, total):

    print("\n\nBegin Evaluation:")

    # Get the new predictions

    # Move the bins that are in this collection of details to the current velodyne folder to run the models on them
    print("Move bins to evaluate to the current folder")
    for detail in details:
        shutil.move(sessionManager.stageDir + "/" + detail["_id"] + ".bin", sessionManager.currentVelDir + "/" + detail["_id"] + ".bin")

    # run all models on bin files
    print("Run models")
    modelRunner.runCyl(sessionManager)
    modelRunner.runSpv(sessionManager)
    modelRunner.runSal(sessionManager)
    modelRunner.runSq3(sessionManager.dataRoot, sessionManager.resultDir + "/" + "sq3")
    modelRunner.runDar(sessionManager.dataRoot, sessionManager.resultDir + "/" + "dar")


    # Move bins to done from the current folder
    print("Move bins to done")
    allfiles = os.listdir(sessionManager.currentVelDir + "/")
    for f in allfiles:
        shutil.move(sessionManager.currentVelDir + "/" + f, sessionManager.doneVelDir + "/" + f)



    # Evaluate 
    print("Eval")

    # Get the labels (modified ground truth)
    labelFiles = []
    for detail in details:
        labelFiles.append(sessionManager.doneLabelDir + "/" + detail["_id"] + ".label")


    # Get the modified predictions for the models
    # Set up modified pred dict
    modifiedPred = {}
    for model in models:
        modifiedPred[model] = []
    # Get the modified pred file paths
    for detail in details:
        for model in models:
            modifiedPred[model].append(sessionManager.doneMutatedPredDir + "/" + model + "/" + detail["_id"] + ".label")
    

    # Get the predictions made for these scenes
    predFilesCyl = glob.glob(sessionManager.resultCylDir + "/*.label")
    predFilesSpv = glob.glob(sessionManager.resultSpvDir + "/*.label")
    predFilesSal = glob.glob(sessionManager.resultSalDir + "/predictions/*.label")
    predFilesSq3 = glob.glob(sessionManager.resultSq3Dir + "/predictions/*.label")
    predFilesDar = glob.glob(sessionManager.resultDarDir + "/predictions/*.label")


    # Sort the labels, modified predictions, and new predictions
    labelFiles = sorted(labelFiles)
    predFiles = {}
    predFiles["cyl"] = sorted(predFilesCyl)    
    predFiles["spv"] = sorted(predFilesSpv)
    predFiles["sal"] = sorted(predFilesSal)
    predFiles["sq3"] = sorted(predFilesSq3)
    predFiles["dar"] = sorted(predFilesDar)
    for model in models:
        modifiedPred[model] = sorted(modifiedPred[model])
    details = sorted(details, key=itemgetter('_id')) 

    # Assert that we have predictions and labels
    totalFiles = len(labelFiles)
    for model in models:
        totalFiles += len(predFiles[model])
    if (totalFiles / (1 + len(models)) != len(details)):
        raise ValueError("ERROR: preds do not match labels, cyl {}, spv {}, sal {}, sq3 {}, dar {}, labels {}, details {}".format(
                                                                                                                        len(predFiles["cyl"]), 
                                                                                                                        len(predFiles["spv"]), 
                                                                                                                        len(predFiles["sal"]), 
                                                                                                                        len(predFiles["sq3"]), 
                                                                                                                        len(predFiles["dar"]), 
                                                                                                                        len(labelFiles), 
                                                                                                                        len(details)))

    # Evaluate the labels, modified predictions, and new predictions
    for index in range(0, len(labelFiles)):
        print("{}/{}, {} | {}/{}".format(index, len(labelFiles), details[index]["_id"],  complete, total))

        for model in models:
            # Get the accuracy differentials
            modelResults = evalLabels(labelFiles[index], modifiedPred[model][index], predFiles[model][index])
            # Update the details
            details[index][model] = modelResults

            # Move model prediction to the done folder
            shutil.move(predFiles[model][index], sessionManager.donePredDir + "/" + model + "/" + details[index]["_id"] + ".label")


    return details




