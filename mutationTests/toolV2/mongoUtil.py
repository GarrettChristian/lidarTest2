
from pymongo import MongoClient


import globals
from fileIoUtil import openLabelBin
import numpy as np
import os
import json


assetCollection = None
assetMetadataCollection = None
mutationCollection = None
accuracyCollection = None



"""
Connect to mongodb 
"""
def mongoConnect():
    global assetCollection
    global assetMetadataCollection
    global mutationCollection
    global accuracyCollection

    configFile = open("../mongoconnect.txt", "r")
    mongoUrl = configFile.readline()
    print("Connecting to: ", mongoUrl)
    configFile.close()
    
    client = MongoClient(mongoUrl)
    db = client["lidar_data"]
    
    assetCollection = db["assets2"]
    assetMetadataCollection = db["asset_metadata2"]
    mutationCollection = db["mutations"]
    accuracyCollection = db["base_accuracy"]


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

    return pcdArr, intensity, semantics, labelInstance, assetRecord



"""
Gets a random asset from a specific sequence scene of type
"""
def getRandomAssetWithinScene(sequence, scene):

    asset = assetCollection.aggregate([
        { "$match": { "sequence" : sequence, "scene" : scene} },
        { "$sample": { "size": 1 } }
    ])

    assetRecord = None
    try:
        assetRecord = asset.next()
    except:
        print("Get assetRecord failed")
        return None, None, None, None, None

    return getInstanceFromAssetRecord(assetRecord)



"""
Gets a random asset of type
"""
def getRandomAssetOfType(typeNum):

    asset = assetCollection.aggregate([
        { "$match": { "typeNum" : typeNum } },
        { "$sample": { "size": 1 } }
    ])

    assetRecord = None
    try:
        assetRecord = asset.next()
    except:
        print("Get assetRecord failed")
        return None, None, None, None, None

    return getInstanceFromAssetRecord(assetRecord)



"""
Gets an asset by the id
"""
def getAssetById(id):

    asset = assetCollection.find_one({ "_id" : id })

    return getInstanceFromAssetRecord(asset)


"""
Gets an asset by the id
"""
def getBaseAccuracy(sequence, scene, model):

    baseAcc = accuracyCollection.find_one({"sequence": sequence, "scene": scene, "model": model})

    return baseAcc



"""
Save mutation data
"""
def saveMutation(mutationData):
    print("Save Mutation Record")
    mutationCollection.insert_many(mutationData)









