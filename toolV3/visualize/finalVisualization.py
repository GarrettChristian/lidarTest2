


from laserscan import LaserScan, SemLaserScan
from laserscanvisFinal import LaserScanVis
import numpy as np
import glob, os
import argparse
import shutil
from pymongo import MongoClient
import json


mutationCollection = None


dataDir = ""
labelsDir = ""
toolDir = ""
finalVisDir = ""

dedup = set()
mutations = set()
models = ["cyl", "spv", "sal"]

color_map_alt = { # bgr
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}





"""
Connect to mongodb 
"""
def mongoConnect():
    global mutationCollection

    configFile = open("../../mongoconnect.txt", "r")
    mongoUrl = configFile.readline()
    print("Connecting to mongodb")
    configFile.close()
    
    client = MongoClient(mongoUrl)
    db = client["lidar_data"]
    
    mutationCollection = db["mutations"]


def setUpDir(finalData):
    global finalVisDir
    global mutations
    global models

    curDir = os.getcwd()

    """
    /finalvis
        /mutation
            /mutationId
                og-id : original scan with original labels
                new-id : new scan with new labels
                og-model-id : original scan with model labels (x3)
                og-model-asset-id : original asset scan with model labels (x3)
                new-model-id : new scan with model labels (x3)
    """

    
    finalVisDir = curDir + "/finalvis"
    if os.path.exists(finalVisDir):
        shutil.rmtree(finalVisDir, ignore_errors=True)
        print("Removing {}".format(finalVisDir))
    os.makedirs(finalVisDir, exist_ok=True)

    mutations = set()

    for mutation in finalData[models[0]].keys():
        os.makedirs(finalVisDir + "/" + mutation, exist_ok=True)
        mutations.add(mutation)

    ids = set()

    for mutation in mutations:
        for model in models:
            for mutationId in finalData[model][mutation]["five"]:
                if (mutationId[0] not in ids):
                    os.makedirs(finalVisDir + "/" + mutation + "/" + mutationId[0], exist_ok=True)
                    ids.add(mutationId[0])


def handleOne(mutation, mutationId):
    global dedup
    global finalVisDir
    global dataDir
    global labelsDir
    global toolDir
    global mutationCollection

    if (mutationId in dedup):
        return

    # Add the id to dedup in case models have duplicate top mutations 
    dedup.add(mutationId)

    """
    /finalvis
        /mutation
            /mutationId
                og-id : original scan with original labels
                new-id : new scan with new labels
                og-model-id : original scan with model labels (x3)
                og-model-asset-id : original asset scan with model labels (x3)
                new-model-id : new scan with model labels (x3)
    """

    print("Saving {} {}".format(mutation, mutationId))
    
    saveDir = finalVisDir + "/" + mutation + "/" + mutationId + "/"

    # Get the mutation data
    item = mutationCollection.find_one({ "_id" : mutationId })

    # assetSeq = 
    # assetScene =  


    color_dict = color_map_alt
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)

    # og-id : original scan with original labels
    origScan = saveDir + "actual-og-" + mutationId + ".png"
    vis = LaserScanVis(scan=scan,
                    scan_names=[dataDir + item["baseSequence"] + "/velodyne/" + item["baseScene"] + ".bin"],
                    label_names=[dataDir + item["baseSequence"] + "/labels/" + item["baseScene"] + ".label"])
    vis.save(origScan)
    vis.destroy()

    # new-id : new scan with new labels
    newScan = saveDir + "actual-new-" + mutationId + ".png"
    vis = LaserScanVis(scan=scan,
                    scan_names=[toolDir + "done/velodyne/" + mutationId + ".bin"],
                    label_names=[toolDir + "done/labels/actual/" + mutationId + ".label"])
    vis.save(newScan)
    vis.destroy()

    modelSaves = {}

    for model in models:

        # og-model-id : original scan with model labels (x3)
        ogModelSave = saveDir + model + "-og-" + mutationId + ".png"
        vis = LaserScanVis(scan=scan,
                        scan_names=[dataDir + item["baseSequence"] + "/velodyne/" + item["baseScene"] + ".bin"],
                        label_names=[labelsDir + "/" + item["baseSequence"] + "/" + model + "/" + item["baseScene"] + ".label"])
        vis.save(ogModelSave)
        vis.destroy()

        # og-model-asset-id : original asset scan with model labels (x3)
        # ogModelAssetSave = saveDir + "new-" + mutationId + ".png"
        # vis = LaserScanVis(scan=scan,
        #                 scan_names=[toolDir + "done/velodyne/" + mutationId + ".bin"],
        #                 label_names=[toolDir + "done/labels/actual/" + mutationId + ".label"])
        # vis.save(newScan)

        # new-model-id : new scan with model labels (x3)
        newModelSave = saveDir + model + "-new-" + mutationId + ".png"
        vis = LaserScanVis(scan=scan,
                        scan_names=[toolDir + "done/velodyne/" + mutationId + ".bin"],
                        label_names=[toolDir + "done/labels/" + model + "/" + mutationId + ".label"])
        vis.save(newModelSave)
        vis.destroy()

        modelSaves[model] = {}
        modelSaves[model]["og"] = ogModelSave
        modelSaves[model]["new"] = newModelSave
  
    



def createImages(finalData):
    global mutations
    global models

    for mutation in mutations:
        for model in models:
            for mutationId in finalData[model][mutation]["five"]:
                handleOne(mutation, mutationId[0])



def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    p.add_argument("-pdata", 
        help="Path to the semanticKITTI sequences", 
        nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
        default="/home/garrett/Documents/data/dataset/sequences/")
    p.add_argument("-plabels", 
        help="Path to the original label files from the models expecting this path to add 00-10", 
        nargs='?', const="/home/garrett/Documents/data/resultsBase/", 
        default="/home/garrett/Documents/data/resultsBase/")
    p.add_argument("-ptool", 
        help="Path to the data directory produced by the tool", 
        nargs='?', const="/home/garrett/Documents/lidarTest2/toolV3/data/", 
        default="/home/garrett/Documents/lidarTest2/toolV3/data/")
    
    return p.parse_args()

    


# ----------------------------------------------------------

def main():
    global dataDir
    global labelsDir
    global toolDir

    print("\n\n------------------------------")
    print("\n\nStarting Range Image Conversion\n\n")

    args = parse_args() 
    dataDir = args.pdata
    labelsDir = args.plabels
    toolDir = args.ptool
    
    mongoConnect()

    finalData = {}
    with open(toolDir + "finalData.json") as f:
        finalData = json.load(f)

    setUpDir(finalData)

    createImages(finalData)


if __name__ == '__main__':
    main()



