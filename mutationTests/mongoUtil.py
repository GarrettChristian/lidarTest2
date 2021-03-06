"""
mongoUtil 
Handles all database interaction

@Author Garrett Christian
@Date 6/23/22
"""

from pymongo import MongoClient

import globals
import toolV5.util.fileIoUtil as fileIoUtil

# --------------------------------------------------------------------------
# Collections



# --------------------------------------------------------------------------


class Mongo:
    def __init__(self, binsPath, labelsPath, mongoConnectPath):
        self.assetCollection = None
        self.assetMetadataCollection = None
        self.mutationCollection = None
        self.accuracyCollection = None
        self.finalDataCollection = None
        self.mongoConnect(mongoConnectPath)

    """
    Connect to mongodb 
    """
    def mongoConnect(mongoConnectPath):

        configFile = open(mongoConnectPath, "r")
        mongoUrl = configFile.readline()
        print("Connecting to mongo client")
        configFile.close()
        
        client = MongoClient(mongoUrl)
        db = client["lidar_data"]
        
        assetCollection = db["assets3"]
        assetMetadataCollection = db["asset_metadata3"]
        mutationCollection = db["mutations"]
        accuracyCollection = db["base_accuracy"]
        finalDataCollection = db["final_data"]


    """
    Gets the data from a given asset Record 
    """
    def getInstanceFromAssetRecord(assetRecord):

        instance = assetRecord["instance"]
        sequence = assetRecord["sequence"]
        scene = assetRecord["scene"]

        pcdArr, intensity, semantics, labelInstance = fileIoUtil.openLabelBin(globals.pathVel, globals.pathLbl, sequence, scene)

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
    Gets a random asset of type
    """
    def getRandomAsset():

        asset = assetCollection.aggregate([
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
    Gets a random asset of specified types
    """
    def getRandomAssetOfTypes(typeNums):

        typeQuery = []
        for type in typeNums:
            typeQuery.append({"typeNum": type})

        asset = assetCollection.aggregate([
            { "$match": {  
                "$or":  typeQuery
            }},
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
    Gets a random asset of specified types
    """
    def getRandomAssetOfTypesWithinScene(typeNums, sequence, scene):

        typeQuery = []
        for type in typeNums:
            typeQuery.append({"typeNum": type})

        asset = assetCollection.aggregate([
            { "$match": {  
                "sequence" : sequence, 
                "scene" : scene,
                "$or":  typeQuery
            }},
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
    Gets the base accuracy obtained on a given scene
    """
    def getBaseAccuracy(sequence, scene):

        baseAcc = accuracyCollection.find_one({"sequence": sequence, "scene": scene})

        return baseAcc



    """
    Save mutation data
    """
    def saveMutationDetails(mutationData):
        print("Save Mutation Record")
        mutationCollection.insert_many(mutationData)


    """
    Save final data
    """
    def saveFinalData(finalData):
        finalDataCollection.insert_one(finalData)





