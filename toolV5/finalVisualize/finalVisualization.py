


from rangeImageFinalVis import LaserScanVis, SemLaserScan
import numpy as np
import glob, os
import argparse
import shutil
from pymongo import MongoClient
import json
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

mutationCollection = None


dataDir = ""
labelsDir = ""
toolDir = ""
finalVisDir = ""

dedup = set()
mutations = set()
models = ["cyl", "spv", "sal"]

scan = None
vis = None

color_map_alt = { # bgr
  0 : [0, 0, 0],
  1 : [0, 0, 0],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 0, 0],
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
  52: [0, 0, 0],
  60: [255, 0, 255],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [0, 0, 0],
  252: [245, 150, 100],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  256: [255, 0, 0],
  257: [250, 0, 0],
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
            for key in finalData[model][mutation].keys():
                if ("top" in key):
                    for mutationId in finalData[model][mutation][key]:
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

    # imageList = []
    imgs = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 16)
    
    saveDir = finalVisDir + "/" + mutation + "/" + mutationId + "/"

    # Get the mutation data
    item = mutationCollection.find_one({ "_id" : mutationId })


    # Intensity Version
    scan.setIntensityColorType(True)

    # og-id : original scan with original labels
    origScanIntensity = saveDir + "actual-og-intensity-" + mutationId + ".png"
    vis.set_new_pcd(scan_name=dataDir + item["baseSequence"] + "/velodyne/" + item["baseScene"] + ".bin",
                    label_name=dataDir + item["baseSequence"] + "/labels/" + item["baseScene"] + ".label")
    vis.save(origScanIntensity)

    # new-id : new scan with new labels
    newScanIntensity = saveDir + "actual-new-intensity" + mutationId + ".png"
    vis.set_new_pcd(scan_name=toolDir + "done/velodyne/" + mutationId + ".bin",
                    label_name=toolDir + "done/labels/actual/" + mutationId + ".label")
    vis.save(newScanIntensity)

    ogImageIntensity = Image.open(origScanIntensity)
    newImageIntensity = Image.open(newScanIntensity)
    drawOgIntensity = ImageDraw.Draw(ogImageIntensity)
    drawNewIntensity = ImageDraw.Draw(newImageIntensity)
    drawOgIntensity.text((0, 0), "Orig Int", (255,255,255), font=font)
    drawNewIntensity.text((0, 0), "New Int", (255,255,255), font=font)
    imgs.append(np.asarray(ogImageIntensity))
    imgs.append(np.asarray(newImageIntensity))

    # Semanitc version 
    scan.setIntensityColorType(False)

    # og-id : original scan with original labels
    origScan = saveDir + "actual-og-" + mutationId + ".png"
    vis.set_new_pcd(scan_name=dataDir + item["baseSequence"] + "/velodyne/" + item["baseScene"] + ".bin",
                    label_name=dataDir + item["baseSequence"] + "/labels/" + item["baseScene"] + ".label")
    vis.save(origScan)

    # new-id : new scan with new labels
    newScan = saveDir + "actual-new-" + mutationId + ".png"
    vis.set_new_pcd(scan_name=toolDir + "done/velodyne/" + mutationId + ".bin",
                    label_name=toolDir + "done/labels/actual/" + mutationId + ".label")
    vis.save(newScan)

    ogImage = Image.open(origScan)
    newImage = Image.open(newScan)
    drawOg = ImageDraw.Draw(ogImage)
    drawNew = ImageDraw.Draw(newImage)
    drawOg.text((0, 0), "Orig Sem", (255,255,255), font=font)
    drawNew.text((0, 0), "New Sem", (255,255,255), font=font)
    imgs.append(np.asarray(ogImage))
    imgs.append(np.asarray(newImage))


    for model in models:

        if "ADD" in mutation:
            # og-model-asset-id : original asset scan with model labels (x3)
            ogModelAssetSave = saveDir + model + "-asset-" + mutationId + ".png"
            vis.set_new_pcd(scan_name=dataDir + item["assetSequence"] + "/velodyne/" + item["assetScene"] + ".bin",
                        label_name=labelsDir + item["assetSequence"] + "/" + model + "/" + item["assetScene"] + ".label")
            vis.save(ogModelAssetSave)

            assetModelImage = Image.open(ogModelAssetSave)
            drawAsset = ImageDraw.Draw(assetModelImage)
            drawAsset.text((0, 0), "Asset " + model, (255,255,255), font=font)
            imgs.append(np.asarray(assetModelImage))

        # og-model-id : original scan with model labels (x3)
        ogModelSave = saveDir + model + "-og-" + mutationId + ".png"
        vis.set_new_pcd(scan_name=dataDir + item["baseSequence"] + "/velodyne/" + item["baseScene"] + ".bin",
                        label_name=labelsDir + item["baseSequence"] + "/" + model + "/" + item["baseScene"] + ".label")
        vis.save(ogModelSave)

        ogModelImage = Image.open(ogModelSave)
        drawOg = ImageDraw.Draw(ogModelImage)
        drawOg.text((0, 0), "Orig " + model, (255,255,255), font=font)
        imgs.append(np.asarray(ogModelImage))

        # new-model-id : new scan with model labels (x3)
        newModelSave = saveDir + model + "-new-" + mutationId + ".png"
        vis.set_new_pcd(scan_name=toolDir + "done/velodyne/" + mutationId + ".bin",
                        label_name=toolDir + "done/labels/" + model + "/" + mutationId + ".label")
        vis.save(newModelSave)
        
        newModelImage = Image.open(newModelSave)        
        drawNew = ImageDraw.Draw(newModelImage)        
        drawNew.text((0, 0), "New " + model, (255,255,255), font=font)
        imgs.append(np.asarray(newModelImage))


  
    # Combine into one image
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    combineSave = finalVisDir + "/" + mutation + "/" + mutationId + ".png"
    imgsComb = np.vstack( imgs )
    imgsComb = Image.fromarray(imgsComb)
    imgsComb.save(combineSave)

       



def createImages(finalData):
    global mutations
    global models

    for mutation in mutations:
        for model in models:
            for key in finalData[model][mutation].keys():
                if ("top" in key):
                    for mutationId in finalData[model][mutation][key]:
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
        nargs='?', const="/home/garrett/Documents/lidarTest2/toolV4/data/", 
        default="/home/garrett/Documents/lidarTest2/toolV4/data/")
    
    return p.parse_args()

    


# ----------------------------------------------------------

def main():
    global dataDir
    global labelsDir
    global toolDir
    global scan
    global vis

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

    # Set up visualization
    color_dict = color_map_alt
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)
    vis = LaserScanVis(scan=scan,
                    scan_name=dataDir + "00/velodyne/000000.bin",
                    label_name=dataDir + "00/labels/000000.label")

    createImages(finalData)


if __name__ == '__main__':
    main()



