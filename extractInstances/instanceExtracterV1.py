
from pymongo import MongoClient
import glob, os
import numpy as np


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
        252: 'moving-car',
        253: 'moving-bicyclist',
        254: 'moving-person',
        255: 'moving-motorcyclist',
        256: 'moving-on-rails',
        257: 'moving-bus',
        258: 'moving-truck',
        259: 'moving-other-vehicle'
}

sceneData = {}
globalData = {}
assetsToSave = []


"""
Connect to mongodb 
"""
def mongoConnect():
    configFile = open("../mongoconnect.txt", "r")
    mongoUrl = configFile.readline()
    print("Connecting to: ", mongoUrl)
    configFile.close()
    
    client = MongoClient(mongoUrl)
    db = client["lidar_data"]
    return db


def parseAsset(filename, scene, instance, semantics, labelInstance):
    global sceneData
    global globalData
    global assetsToSave

    mask = (labelInstance == instance)
    type = semantics[mask]

    typeName = name_label_mapping[type[0]]

    id = scene + "-" + filename + "-" + str(instance)
    
    asset = {}
    asset["_id"] = id
    asset["scene"] = scene
    asset["file"] = filename
    asset["instance"] = int(instance)
    asset["type"] = typeName
    asset["typeNum"] = int(type[0])
    asset["points"] = int(np.shape(type)[0])
    asset["sceneTypeNum"] = sceneData[scene].get(typeName, 0)
    asset["globalTypeNum"] = globalData.get(typeName, 0)

    sceneData[scene][typeName] = 1 + sceneData[scene].get(typeName, 0)
    globalData[typeName] = 1 + globalData.get(typeName, 0)

    assetsToSave.append(asset)


def parseAssets(labelsFileName, scene):

    label_arr = np.fromfile(labelsFileName, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 

    fileName = os.path.basename(labelsFileName).replace('.label', '')

    seenInst = set()
    for instance in labelInstance:
        seenInst.add(instance)
    
    for instance in seenInst:

        # Skip the unlabeled asset
        if (instance != 0):
            parseAsset(fileName, scene, instance, semantics, labelInstance)


def main():
    global sceneData
    global globalData
    global assetsToSave

    print("\n\n------------------------------")
    print("\n\nStarting Asset Loader\n\n")

    print("Connecting to Mongo")
    mdb = mongoConnect()
    mdbColAssets = mdb["assets"]
    mdbColAssetMetadata = mdb["asset_metadata"]
    print("Connected")

    path = "/home/garrett/Documents/data/dataset/sequences/"

    print("Parsing {} :".format(path))

    num = 0

    for x in range(0, 11):
        
        folderNum = str(x).rjust(2, '0')
        currPath = path + folderNum

        # Add Scene
        sceneData[folderNum] = {}
        sceneData[folderNum]["_id"] = folderNum

        files = np.array(glob.glob(currPath + "/labels/*.label", recursive = True))
        print("\n\nParsing ", folderNum)
        
        for file in files:
                
            parseAssets(file, folderNum)
            print(num, file)
            num += 1

            # Batch insert
            if (len(assetsToSave) >= 2000):
                mdbColAssets.insert_many(assetsToSave)
                assetsToSave = []

        print(sceneData[folderNum])


        # Save metadata for scene
        curScene = sceneData[folderNum]
        mdbColAssetMetadata.insert_one(curScene)

    # Batch insert any remaining
    if (len(assetsToSave) != 0):
        mdbColAssets.insert_many(assetsToSave)

    globalData["_id"] = "all"
    mdbColAssetMetadata.insert_one(globalData)

if __name__ == '__main__':
    main()



