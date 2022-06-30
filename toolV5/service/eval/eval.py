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


import service.eval.ioueval as ioueval

import data.baseAccuracyRepository as baseAccuracyRepository

from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import learning_map_inv
from domain.semanticMapping import learning_map
from domain.semanticMapping import learning_ignore
from domain.semanticMapping import name_label_mapping

# --------------------------------------------------------------------------
# Constants


pathToModels = "/home/garrett/Documents"

modelCylinder3D = "Cylinder3D"
modelSpvnas = "spvnas"
modelSalsaNext = "SalsaNext"

modelCyl = "cyl"
modelSpv = "spv"
modelSal = "sal"


# --------------------------------------------------------------------------
# Runners

"""
Runner for the Cylinder3D model
"""
def runCyl(sessionManager):
    print("running {}".format(modelCylinder3D))

    runCommand = "python demo_folder.py "
    runCommand += "--demo-folder " + sessionManager.dataRoot + "/sequences/00/velodyne " 
    runCommand += "--save-folder " + sessionManager.resultCylDir

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelCylinder3D)

    # runCommand += "2> /dev/null"

    # Run Model
    os.system(runCommand)
    

"""
Runner for the SPVNAS model
"""
def runSpv(sessionManager):
    print("running {}".format(modelSpvnas))

    runCommand = "torchpack dist-run " 
    runCommand += "-np 1 python evaluate.py configs/semantic_kitti/default.yaml "
    runCommand += "--name SemanticKITTI_val_SPVNAS@65GMACs "
    runCommand += "--data-dir " + sessionManager.dataRoot + "/sequences "
    runCommand += "--save-dir " + sessionManager.resultSpvDir

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSpvnas)

    # runCommand += "2> /dev/null"

    # Run Model
    os.system(runCommand)


"""
Runner for the SalsaNext model
"""
def runSal(sessionManager):
    print("running {}".format(modelSalsaNext))

    runCommand = "python infer.py " 
    # Data to run on
    runCommand += "-d " + sessionManager.dataRoot
    # Results
    runCommand += " -l " + sessionManager.resultDir + "/" + modelSal
    # model
    runCommand += " -m /home/garrett/Documents/SalsaNext/pretrained "
    runCommand += "-s test -c 1"
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSalsaNext + "/train/tasks/semantic")

    # runCommand += "2> /dev/null"

    # Run Model
    os.system(runCommand)


# --------------------------------------------------------------------------
# Evaluators


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
def evalLabels(label_file, pred_file, baseAccuracy, baseAccuracyAsset, typeNum, mutation):

    # print()
    # print(label_file)
    # print(pred_file)


    fileName = basename(label_file)
    fileName = fileName.replace(".label", "")

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
                # Found some guesses in original of type average between the two 
                if (classNum == learning_map[typeNum] and baseClassJacc > 0):
                    # Replace the asset you added to the scene with the original guess for that class
                    # So if you added a person and originally in the base there was a person 
                    # this should scale if you guessed the class originally 
                    averageOfClass = (baseAccuracyAsset[typeNameAsset] + baseClassJacc) / 2
                    jaccardAdd += averageOfClass
                    results["2" + className] = averageOfClass
                # Was class 0 in base prediction, likely was not there add from asset prediction jaccard
                elif (classNum == learning_map[typeNum]): # baseClassJacc == 0
                    # Replace the asset you added to the scene with the original guess for that class
                    # So if you added a person and originally in the base there was no person 
                    # this should scale if you guessed the class originally 
                    jaccardAdd += baseAccuracyAsset[typeNameAsset]
                    results["2" + className] = baseAccuracyAsset[typeNameAsset]
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
                classInvName = name_label_mapping[classNum]
                baseClassJacc = baseAccuracy[classInvName]
                if (classNum in classesInLabel):
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
def evalBatch(threadNum, details, sessionManager):

    baseAccRepository = baseAccuracyRepository.BaseAccuracyRepository(sessionManager.mongoConnect)

    # Lock mutex
    print("\n\nBegin Evaluation:")


    # move the bins to the velodyne folder to run the models on them
    print("move to vel folder")
    stageVel = sessionManager.stageDir + "/velodyne" + str(threadNum) + "/"
    allfiles = os.listdir(stageVel)
    for f in allfiles:
        shutil.move(stageVel + f, sessionManager.currentVelDir + "/" + f)

    # run all models on bin files
    print("Run models")
    runCyl(sessionManager)
    runSpv(sessionManager)
    runSal(sessionManager)

    # Move the model label files to the evaluation folder   
    print("Move to eval folder")
    evalCylDir = sessionManager.evalDir + "/label" + str(threadNum) + "/" + modelCyl + "/"
    evalSpvDir = sessionManager.evalDir + "/label" + str(threadNum) + "/" + modelSpv + "/"
    evalSalDir = sessionManager.evalDir + "/label" + str(threadNum) + "/" + modelSal + "/"

    allfiles = os.listdir(sessionManager.resultCylDir + "/")
    for f in allfiles:
        shutil.move(sessionManager.resultCylDir + "/" + f, evalCylDir + f)

    allfiles = os.listdir(sessionManager.resultSpvDir + "/")
    for f in allfiles:
        shutil.move(sessionManager.resultSpvDir + "/" + f, evalSpvDir + f)

    allfiles = os.listdir(sessionManager.resultSalDir + "/predictions/")
    for f in allfiles:
        shutil.move(sessionManager.resultSalDir + "/predictions/" + f, evalSalDir + f)
       

    # Move bins to done from the model folder
    print("Move bins to done")
    allfiles = os.listdir(sessionManager.currentVelDir + "/")
    for f in allfiles:
        shutil.move(sessionManager.currentVelDir + "/" + f, sessionManager.doneVelDir + "/" + f)



    # Evaluate 
    print("Eval")
    stageLabel = sessionManager.stageDir + "/labels" + str(threadNum) + "/"
    labelFiles = glob.glob(stageLabel + "*.label")
    predFilesCyl = glob.glob(evalCylDir + "*.label")
    predFilesSal = glob.glob(evalSalDir + "*.label")
    predFilesSpv = glob.glob(evalSpvDir + "*.label")
    
    # Order the update files cronologically
    labelFiles = sorted(labelFiles)
    predFiles = {}
    predFiles["cyl"] = sorted(predFilesCyl)    
    predFiles["spv"] = sorted(predFilesSpv)        
    predFiles["sal"] = sorted(predFilesSal)    
    details = sorted(details, key=itemgetter('_id')) 
    for index in range(0, len(labelFiles)):

        # Get the base accuracy for a given scene
        baseAccuracy = baseAccRepository.getBaseAccuracy(details[index]["baseSequence"], details[index]["baseScene"])

        baseAccuracyAsset = None
        if "ADD" in details[index]["mutation"]:
            baseAccuracyAsset = baseAccRepository.getBaseAccuracy(details[index]["assetSequence"], details[index]["assetScene"])

        for model in sessionManager.models:
            # Get the base accuracy for a specific model
            baseAccModel = baseAccuracy[model]
            baseAccAssetModel = None
            if baseAccuracyAsset != None:
                baseAccAssetModel = baseAccuracyAsset[model]
            
            # Get the accuracy differentials
            modelResults = evalLabels(labelFiles[index], predFiles[model][index], baseAccModel, baseAccAssetModel, 
                                        details[index]["typeNum"], details[index]["mutation"])
            details[index][model] = modelResults
    

    # Move to done folder
    print("Move to done folder")    
    allfiles = os.listdir(stageLabel)
    for f in allfiles:
        shutil.move(stageLabel + f, sessionManager.doneLabelActualDir + "/" + f)

    allfiles = os.listdir(evalCylDir)
    for f in allfiles:
        shutil.move(evalCylDir + f, sessionManager.doneLabelDir + "/" + modelCyl + "/" + f)

    allfiles = os.listdir(evalSpvDir)
    for f in allfiles:
        shutil.move(evalSpvDir + f, sessionManager.doneLabelDir + "/" + modelSpv + "/" + f)

    allfiles = os.listdir(evalSalDir)
    for f in allfiles:
        shutil.move(evalSalDir + f, sessionManager.doneLabelDir + "/" + modelSal + "/" + f)


    return details


