

from pymongo import MongoClient


import globals
from fileIoUtil import openLabelBin
import numpy as np



assetCollection = None
assetMetadataCollection = None



"""
Connect to mongodb 
"""
def mongoConnect():
    global assetCollection
    global assetMetadataCollection

    configFile = open("../mongoconnect.txt", "r")
    mongoUrl = configFile.readline()
    print("Connecting to: ", mongoUrl)
    configFile.close()
    
    client = MongoClient(mongoUrl)
    db = client["lidar_data"]
    
    assetCollection = db["assets2"]
    assetMetadataCollection = db["asset_metadata2"]



"""
Gets the data from a given asset Record 
"""
def getInstanceFromAssetRecord(assetRecord):

    instance = assetRecord["instance"]
    sequence = assetRecord["sequence"]
    scene = assetRecord["scene"]

    pcdArr, intensity, semantics, labelInstance = openLabelBin(globals.path, sequence, scene)

    maskOnlyInst = (labelInstance == instance)

    pcdArr = pcdArr[maskOnlyInst, :]
    intensity = intensity[maskOnlyInst]
    semantics = semantics[maskOnlyInst]
    labelInstance = labelInstance[maskOnlyInst]

    return pcdArr, intensity, semantics, labelInstance 



"""
Gets a random asset from a specific sequence scene of type
"""
def getRandomAssetWithinScene(sequence, scene):

    asset = assetCollection.aggregate([
        { "$match": { "sequence" : sequence, "scene" : scene} },
        { "$sample": { "size": 1 } }
    ])

    return getInstanceFromAssetRecord(asset.next())



"""
Gets a random asset of type
"""
def getRandomAssetOfType(typeNum):

    asset = assetCollection.aggregate([
        { "$match": { "typeNum" : typeNum } },
        { "$sample": { "size": 1 } }
    ])

    return getInstanceFromAssetRecord(asset.next())



"""
Gets an asset by the id
"""
def getRandomAssetWithinScene(sequence, scene):

    asset = assetCollection.aggregate([
        { "$match": { "sequence" : sequence, "scene" : scene} },
        { "$sample": { "size": 1 } }
    ])

    return getInstanceFromAssetRecord(asset.next())



"""
Save mutation data
"""
def saveMutation(mutationData):

    print("SAVE TODO")










