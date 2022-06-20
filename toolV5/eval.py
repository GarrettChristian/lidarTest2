

from pymongo import MongoClient
import glob, os
import numpy as np
from os.path import basename
import shutil
from operator import itemgetter

import globals
import mongoUtil

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
    # 252: 'moving-car',
    # 253: 'moving-bicyclist',
    # 254: 'moving-person',
    # 255: 'moving-motorcyclist',
    # 256: 'moving-on-rails',
    # 257: 'moving-bus',
    # 258: 'moving-truck',
    # 259: 'moving-other-vehicle'
}

# https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
learning_map = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,    # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

learning_map_inv = { # inverse of previous map
    0: 0,      # "unlabeled", and others ignored
    1: 10,     # "car"
    2: 11,     # "bicycle"
    3: 15,     # "motorcycle"
    4: 18,     # "truck"
    5: 20,     # "other-vehicle"
    6: 30,     # "person"
    7: 31,     # "bicyclist"
    8: 32,     # "motorcyclist"
    9: 40,     # "road"
    10: 44,    # "parking"
    11: 48,    # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81,    # "traffic-sign"
}

learning_ignore = { # Ignore classes
    0: True,      # "unlabeled", and others ignored
    1: False,     # "car"
    2: False,     # "bicycle"
    3: False,     # "motorcycle"
    4: False,     # "truck"
    5: False,     # "other-vehicle"
    6: False,     # "person"
    7: False,     # "bicyclist"
    8: False,     # "motorcyclist"
    9: False,     # "road"
    10: False,    # "parking"
    11: False,    # "sidewalk"
    12: False,    # "other-ground"
    13: False,    # "building"
    14: False,    # "fence"
    15: False,    # "vegetation"
    16: False,    # "trunk"
    17: False,    # "terrain"
    18: False,    # "pole"
    19: False    # "traffic-sign"
}



pathToModels = "/home/garrett/Documents"

modelCylinder3D = "Cylinder3D"
modelSpvnas = "spvnas"
modelSalsaNext = "SalsaNext"

modelCyl = "cyl"
modelSpv = "spv"
modelSal = "sal"

# resultsDir = "/home/garrett/Documents/data/results"
# dataRoot = "/home/garrett/Documents/data/tmp/dataset"


models = [modelCyl, modelSpv, modelSal]

# ------------------

"""
Runner for the Cylinder3D model
"""
def runCyl():
    print("running {}".format(modelCylinder3D))

    runCommand = "python demo_folder.py "
    runCommand += "--demo-folder " + globals.dataRoot + "/sequences/00/velodyne " 
    runCommand += "--save-folder " + globals.resultCylDir

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelCylinder3D)

    # runCommand += "2> /dev/null"

    # Run Model
    os.system(runCommand)
    

"""
Runner for the SPVNAS model
"""
def runSpv():
    print("running {}".format(modelSpvnas))

    runCommand = "torchpack dist-run " 
    runCommand += "-np 1 python evaluate.py configs/semantic_kitti/default.yaml "
    runCommand += "--name SemanticKITTI_val_SPVNAS@65GMACs "
    runCommand += "--data-dir " + globals.dataRoot + "/sequences "
    runCommand += "--save-dir " + globals.resultSpvDir

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSpvnas)

    # runCommand += "2> /dev/null"

    # Run Model
    os.system(runCommand)


"""
Runner for the SalsaNext model
"""
def runSal():
    print("running {}".format(modelSalsaNext))

    runCommand = "python infer.py " 
    # Data to run on
    runCommand += "-d " + globals.dataRoot
    # Results
    runCommand += " -l " + globals.resultDir + "/" + modelSal
    # model
    runCommand += " -m /home/garrett/Documents/SalsaNext/pretrained "
    runCommand += "-s test -c 1"
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSalsaNext + "/train/tasks/semantic")

    # runCommand += "2> /dev/null"

    # Run Model
    os.system(runCommand)


# ------------------


"""
Jaccard accuracy 

Modified from SemanticKITTI Development Kit 
https://github.com/PRBonn/semantic-kitti-api
https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/np_ioueval.py
"""
class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    # print("[IOU EVAL] IGNORE: ", self.ignore)
    # print("[IOU EVAL] INCLUDE: ", self.include)

    # reset the class counters
    self.reset()

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == y_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.conf_matrix, idxs, 1)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()

    sumJac = 0
    classesNonZero = 0
    for className in self.include:
      unionClass = tp[className] + fp[className] + fn[className]
      if unionClass != 0:
        sumJac += tp[className] / unionClass
        classesNonZero += 1

    iou_mean = sumJac / classesNonZero

    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    # iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  # def getIoU(self):
  #   tp, fp, fn = self.getStats()
  #   intersection = tp
  #   union = tp + fp + fn + 1e-15
  #   iou = intersection / union
  #   iou_mean = (intersection[self.include] / union[self.include]).mean()
  #   return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"


# ------------------

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
def evalLabels(label_file, pred_file, model, details):

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

    evaluator = iouEval(numClasses, ignore)
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
    m_jaccard, class_jaccard = evaluator.getIoU()

    results = {}


    results["_id"] = model + "-" + fileName
    results["file"] = fileName
    results["model"] = model
    results["jaccard"] = m_jaccard.item()
    results["accuracy"] = m_accuracy.item()

    # collect classwise jaccard
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            results[name_label_mapping[learning_map_inv[i]]] = jacc


    baseAccuracy = mongoUtil.getBaseAccuracy(details["baseSequence"], details["baseScene"], model)

    jacChange = results["jaccard"] - baseAccuracy["jaccard"]
    accChange = results["accuracy"] - baseAccuracy["accuracy"]
    
    results["jaccardChange"] = jacChange
    results["accuracyChange"] = accChange

    # Asset accuracy change
    if "ADD" in details["mutation"]:
        baseAccuracyAsset = mongoUtil.getBaseAccuracy(details["assetSequence"], details["assetScene"], model)
        type = name_label_mapping[learning_map_inv[learning_map[details["typeNum"]]]]
        typeJacChange = results[type] - baseAccuracyAsset[type]
        results["jaccardChangeAsset"] = typeJacChange


    # Bucketing
    percentLossAcc = results["accuracyChange"] * 100

    bucketA = 0 # percentLoss >= 0.1 %
    if (percentLossAcc < -5):
        bucketA = 5
    elif (percentLossAcc < -2):
        bucketA = 4
    elif (percentLossAcc < -1):
        bucketA = 3
    elif (percentLossAcc < -0.5):
        bucketA = 2
    elif (percentLossAcc < -0.1):
        bucketA = 1

    results["percentLossAcc"] = percentLossAcc
    results["bucketA"] = bucketA

    percentLossJac = results["jaccardChange"] * 100

    bucketJ = 0 # percentLoss >= 0.1 %
    if (percentLossJac < -5):
        bucketJ = 5
    elif (percentLossJac < -2):
        bucketJ = 4
    elif (percentLossJac < -1):
        bucketJ = 3
    elif (percentLossJac < -0.5):
        bucketJ = 2
    elif (percentLossJac < -0.1):
        bucketJ = 1

    results["percentLossJac"] = percentLossJac
    results["bucketJ"] = bucketJ


    return results
    

# ------------------

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
def evalBatch(threadNum, details):

    # Lock mutex
    print("Lock mutex TODO")


    # move the bins to the velodyne folder to run the models on them
    print("move to vel folder")
    stageVel = globals.stageDir + "/velodyne" + str(threadNum) + "/"
    allfiles = os.listdir(stageVel)
    for f in allfiles:
        shutil.move(stageVel + f, globals.currentVelDir + "/" + f)

    # run all models on bin files
    print("Run models")
    runCyl()
    runSpv()
    runSal()

    # Move the model label files to the evaluation folder   
    print("Move to eval folder")
    evalCylDir = globals.evalDir + "/label" + str(threadNum) + "/" + modelCyl + "/"
    evalSpvDir = globals.evalDir + "/label" + str(threadNum) + "/" + modelSpv + "/"
    evalSalDir = globals.evalDir + "/label" + str(threadNum) + "/" + modelSal + "/"

    allfiles = os.listdir(globals.resultCylDir + "/")
    for f in allfiles:
        shutil.move(globals.resultCylDir + "/" + f, evalCylDir + f)

    allfiles = os.listdir(globals.resultSpvDir + "/")
    for f in allfiles:
        shutil.move(globals.resultSpvDir + "/" + f, evalSpvDir + f)

    allfiles = os.listdir(globals.resultSalDir + "/predictions/")
    for f in allfiles:
        shutil.move(globals.resultSalDir + "/predictions/" + f, evalSalDir + f)
       

    # Move bins to done from the model folder
    print("Move bins to done")
    allfiles = os.listdir(globals.currentVelDir + "/")
    for f in allfiles:
        shutil.move(globals.currentVelDir + "/" + f, globals.doneVelDir + "/" + f)


    # Unlock Mutex
    print("Unlock Mutex TODO")


    # Evaluate 
    print("Eval")
    stageLabel = globals.stageDir + "/labels" + str(threadNum) + "/"
    labelFiles = glob.glob(stageLabel + "*.label")
    predFilesCyl = glob.glob(evalCylDir + "*.label")
    predFilesSal = glob.glob(evalSalDir + "*.label")
    predFilesSpv = glob.glob(evalSpvDir + "*.label")
    
    # Order the update files cronologically
    labelFiles = sorted(labelFiles)
    predFilesCyl = sorted(predFilesCyl)    
    predFilesSal = sorted(predFilesSal)    
    predFilesSpv = sorted(predFilesSpv)        
    details = sorted(details, key=itemgetter('_id')) 
    for index in range(0, len(labelFiles)):

        cylResults = evalLabels(labelFiles[index], predFilesCyl[index], modelCyl, details[index])
        salResults = evalLabels(labelFiles[index], predFilesSal[index], modelSal, details[index])
        spvResults = evalLabels(labelFiles[index], predFilesSpv[index], modelSpv, details[index])

        details[index][modelCyl] = cylResults
        details[index][modelSal] = salResults
        details[index][modelSpv] = spvResults
    

    # Move to done folder
    print("Move to done folder")    
    allfiles = os.listdir(stageLabel)
    for f in allfiles:
        shutil.move(stageLabel + f, globals.doneLabelActualDir + "/" + f)

    allfiles = os.listdir(evalCylDir)
    for f in allfiles:
        shutil.move(evalCylDir + f, globals.doneLabelCylDir + "/" + f)

    allfiles = os.listdir(evalSpvDir)
    for f in allfiles:
        shutil.move(evalSpvDir + f, globals.doneLabelSpvDir + "/" + f)

    allfiles = os.listdir(evalSalDir)
    for f in allfiles:
        shutil.move(evalSalDir + f, globals.doneLabelSalDir + "/" + f)


    return details




# ------------------


"""
Prepares the final details dictionary preloading some of the keys

@return finalData dictionary with preloaded keys
"""
def prepFinalDetails():
    finalData = {}

    finalData["batchId"] = globals.batchId

    models = ["cyl", "spv", "sal"]

    for model in models:
        finalData[model] = {}

        for mutation in globals.mutationsEnabled:
            mutationString = str(mutation).replace("Mutation.", "")

            finalData[model][mutationString] = {}
            finalData[model][mutationString]["top_acc"] = []
            finalData[model][mutationString]["top_jac"] = []


    for mutation in globals.mutationsEnabled:
        mutationString = str(mutation).replace("Mutation.", "")
        finalData[mutationString] = {}
        finalData[mutationString]["count"] = 0
        finalData[mutationString]["bucketsA"] = {}
        finalData[mutationString]["bucketsA2"] = {}
        finalData[mutationString]["bucketsJ"] = {}
        finalData[mutationString]["bucketsJ2"] = {}

    finalData["bucketsA"] = {}
    finalData["bucketsA2"] = {}
    finalData["bucketsJ"] = {}
    finalData["bucketsJ2"] = {}

    return finalData
    

"""
Updates the final details dictionary after a batch
This removes the bins and labels that do not meet the top five criteria

@param details list of detail dictionarys that enumerates what occured in this transformation
@param finalData ditctionary that describes what should be saved and how many of each mutation occured
@return finalData dictionary updated with new mutations that occured
"""
def finalDetails(details, finalData):

    models = ["cyl", "spv", "sal"]

    potentialRemove = set()
    deleteFiles = []
    
    for detail in details:
        # Add count for mutation
        finalData[detail["mutation"]]["count"] = finalData[detail["mutation"]]["count"] + 1

        bucketFailKeyA = ""
        bucketFailKeyJ = ""
        bucketFailKeyA2 = ""
        bucketFailKeyJ2 = ""

        # Check if we have a lower accuracy change for this mutation
        for model in models:

            # Save top 5 Accuracy

            # don't have top_acc yet, add it 
            if (len(finalData[model][detail["mutation"]]["top_acc"]) < 5):
                finalData[model][detail["mutation"]]["top_acc"].append((detail["_id"], detail[model]["accuracyChange"]))
                finalData[model][detail["mutation"]]["top_acc"].sort(key = lambda x: x[1])

            # Do have top_acc check against current highest
            else:
                idRemove = detail["_id"]
                    
                # new lower change to acc
                if (finalData[model][detail["mutation"]]["top_acc"][4][1] > detail[model]["accuracyChange"]):
                    finalData[model][detail["mutation"]]["top_acc"].append((detail["_id"], detail[model]["accuracyChange"]))
                    finalData[model][detail["mutation"]]["top_acc"].sort(key = lambda x: x[1])
                    idRemove = finalData[model][detail["mutation"]]["top_acc"].pop()[0]
            
                potentialRemove.add(idRemove)


            # Top Jaccard Change

            # don't have top_acc yet, add it 
            if (len(finalData[model][detail["mutation"]]["top_jac"]) < 5):
                finalData[model][detail["mutation"]]["top_jac"].append((detail["_id"], detail[model]["jaccardChange"]))
                finalData[model][detail["mutation"]]["top_jac"].sort(key = lambda x: x[1])

            # Do have top_acc check against current highest
            else:
                idRemove = detail["_id"]
                
                # new lower change to acc
                if (finalData[model][detail["mutation"]]["top_jac"][4][1] > detail[model]["accuracyChange"]):
                    finalData[model][detail["mutation"]]["top_jac"].append((detail["_id"], detail[model]["jaccardChange"]))
                    finalData[model][detail["mutation"]]["top_jac"].sort(key = lambda x: x[1])
                    idRemove = finalData[model][detail["mutation"]]["top_jac"].pop()[0]
            
                potentialRemove.add(idRemove)



            # Accuracy bucket model 
            bucketA = detail[model]["bucketA"]
            if (bucketA != 0):
                bucketFailKeyA += (model + "_")
            if (bucketA > 1):
                bucketFailKeyA2 += (model + "_")

            countModelBucketA = finalData[model][detail["mutation"]].get("bucketA_" + str(bucketA), 0)
            finalData[model][detail["mutation"]]["bucketA_" + str(bucketA)] = countModelBucketA + 1

            # Jaccard bucket model
            bucketJ = detail[model]["bucketJ"]
            if (bucketJ > 0):
                bucketFailKeyJ += (model + "_")
            if (bucketJ > 1):
                bucketFailKeyJ2 += (model + "_")

            countModelBucket = finalData[model][detail["mutation"]].get("bucketJ_" + str(bucketJ), 0)
            finalData[model][detail["mutation"]]["bucketJ_" + str(bucketJ)] = countModelBucket + 1


        # Final bucket data all and mutation
        if bucketFailKeyA == "":
            bucketFailKeyA = "none_"
        bucketFailKeyA += "failed"
        
        failureCountMutant = finalData[detail["mutation"]]["bucketsA"].get(bucketFailKeyA, 0)
        finalData[detail["mutation"]]["bucketsA"][bucketFailKeyA] = failureCountMutant + 1
        failureCountAll = finalData["bucketsA"].get(bucketFailKeyA, 0)
        finalData["bucketsA"][bucketFailKeyA] = failureCountAll + 1

        # Final bucket data all and mutation
        if bucketFailKeyA2 == "":
            bucketFailKeyA2 = "none_"
        bucketFailKeyA2 += "failed"
        
        failureCountMutant = finalData[detail["mutation"]]["bucketsA2"].get(bucketFailKeyA2, 0)
        finalData[detail["mutation"]]["bucketsA2"][bucketFailKeyA2] = failureCountMutant + 1
        failureCountAll = finalData["bucketsA2"].get(bucketFailKeyA2, 0)
        finalData["bucketsA2"][bucketFailKeyA2] = failureCountAll + 1

        # Final bucket data all and mutation
        if bucketFailKeyJ == "":
            bucketFailKeyJ = "none_"
        bucketFailKeyJ += "failed"
        
        failureCountMutant = finalData[detail["mutation"]]["bucketsJ"].get(bucketFailKeyJ, 0)
        finalData[detail["mutation"]]["bucketsJ"][bucketFailKeyJ] = failureCountMutant + 1
        failureCountAll = finalData["bucketsJ"].get(bucketFailKeyJ, 0)
        finalData["bucketsJ"][bucketFailKeyJ] = failureCountAll + 1

        # Final bucket data all and mutation
        if bucketFailKeyJ2 == "":
            bucketFailKeyJ2 = "none_"
        bucketFailKeyJ2 += "failed"
        
        failureCountMutant = finalData[detail["mutation"]]["bucketsJ2"].get(bucketFailKeyJ2, 0)
        finalData[detail["mutation"]]["bucketsJ2"][bucketFailKeyJ2] = failureCountMutant + 1
        failureCountAll = finalData["bucketsJ2"].get(bucketFailKeyJ2, 0)
        finalData["bucketsJ2"][bucketFailKeyJ2] = failureCountAll + 1


    # Remove bin / labels that are not within the top 5
    idInUse = set()
    for mutation in globals.mutationsEnabled:
        mutationString = str(mutation).replace("Mutation.", "")
        for model in models:
            for detailRecord in finalData[model][mutationString]["top_acc"]:
                idInUse.add(detailRecord[0])
            for detailRecord in finalData[model][mutationString]["top_jac"]:
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

