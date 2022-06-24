
import csv
import argparse
import json
import sys
from pymongo import MongoClient

models = ["cyl", "spv", "sal"]

modelCombos = ["cyl", "spv", "sal", "cyl_spv", "cyl_sal", "spv_sal", "cyl_spv_sal"]

mutations = ["SCENE_DEFORM", 
            "SCENE_MIRROR_ROTATE", 
            "SCENE_REMOVE", 
            "ADD_ROTATE", 
            "ADD_MIRROR_ROTATE", 
            "SIGN_REPLACE", 
            "SCENE_INTENSITY", 
            "SCENE_SCALE"]

mutationSet = set(mutations)

def creatBucketCol(bucketData, bucketKey):
    
    col = [bucketKey]
    col.append(bucketData["total"])
    for model in models:
        col.append(bucketData["total_" + model])
        
    col.append("")

    col.append(bucketData["avg"]) 
    for model in models:
        col.append(bucketData["avg_" + model])

    if (bucketData["min"] == sys.maxsize):
        col.append("-")
    else:
        col.append(bucketData["min"])
    for model in models:
        if (bucketData["min_" + model] == sys.maxsize):
            col.append("-")
        else:
            col.append(bucketData["min_" + model])

    
    if (bucketData["max"] == sys.maxsize * -1):
        col.append("-")
    else:
        col.append(bucketData["max"])
    for model in models:
        if (bucketData["max_" + model] == sys.maxsize * -1):
            col.append("-")
        else:
            col.append(bucketData["max_" + model])
    
    col.append("")
    for modelCombo in modelCombos:
        if (modelCombo in bucketData["model_overlap"].keys()):
            col.append(bucketData["model_overlap"][modelCombo])
        else:
            col.append(0)

    return col

def creatAllCol(allData):
    
    col = ["All"]
    col.append(allData["total"])
    for _ in models:
        col.append("")
        
    col.append("")

    col.append(allData["avg"]) 
    for model in models:
        col.append(allData["avg_" + model])

    col.append(allData["min"])
    for model in models:
        col.append(allData["min_" + model])

    col.append(allData["max"])
    for model in models:
        col.append(allData["max_" + model])
    
    col.append("")
    for _ in modelCombos:
        col.append("")

    return col

def createMutationCsv(mutationDataAcc, mutation, accType):

    mutationData = mutationDataAcc[accType]

    cols = []

    titleCol = [mutation, "Total"]
    for model in models:
        titleCol.append(model + " Total")
        
    titleCol.append("")

    titleCol.append("Avg " + accType + " % Loss") 
    for model in models:
        titleCol.append("Avg " + accType + " % Loss {}".format(model))

    titleCol.append("Min " + accType + " % Loss")
    for model in models:
        titleCol.append("Min " + accType + " % Loss {}".format(model))

    titleCol.append("Max " + accType + " % Loss")
    for model in models:
        titleCol.append("Max " + accType + " % Loss {}".format(model))
    
    titleCol.append("")
    for modelCombo in modelCombos:
        titleCol.append(modelCombo)
    
    cols.append(titleCol)

    allCol = creatAllCol(mutationData["all"])
    cols.append(allCol)


    for bucketNum in range(0, 6):
        bucketKey = "bucket_" + str(bucketNum)
        bucketCol = creatBucketCol(mutationData[bucketKey], bucketKey)
        cols.append(bucketCol)

    cols.append([""] * len(allCol))

    keyCol = ["", "Key", 
            "bucket_0:         x >= -0.1%", 
            "bucket_1: -0.1% > x >= -0.5%", 
            "bucket_2: -0.5% > x >= -1%", 
            "bucket_3:   -1% > x >= -2%",
            "bucket_4:   -2% > x >= -5%",
            "bucket_5:   -5% > x"]
    while (len(keyCol) < len(allCol)):
        keyCol.append("")
    cols.append(keyCol)
     
    rows = zip(*cols)

    csvFile = open(mutation + "_" + accType + ".csv", "w")
    csvWriter = csv.writer(csvFile)

    for row in rows:
        csvWriter.writerow(row)

       
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
    
    return db["final_data"]



def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    p.add_argument("-data", 
        help="Path to the data directory produced by the tool", 
        nargs='?', const="/home/garrett/Documents/lidarTest2/toolV5/data/", 
        default="/home/garrett/Documents/lidarTest2/toolV5/data/")
    p.add_argument("-id", 
        help="Id to the final data to create the report for", 
        nargs='?', const=None, default=None)
    
    return p.parse_args()

    
# ----------------------------------------------------------

def main():

    print("\n\n------------------------------")
    print("\n\nStarting Mutation CSV Generator\n\n")

    args = parse_args() 
    dataDir = args.data
    dataId = args.id

    
    data = None
    
    # Get from database if id is provided
    if (dataId != None):
        print("Getting data from mongo with id {}".format(dataId))
        mdbCollection = mongoConnect()
        data = mdbCollection.find_one({"_id": dataId})

    # Get from last run
    if (data == None):
        print("Getting data from {}".format(dataDir + 'finalData.json'))
        # Opening JSON file
        f = open(dataDir + 'finalData.json')
        # returns JSON object as a dictionary
        data = json.load(f)

    # Create a csv with the data from the final data json run 
    for key in data.keys():
        if (key in mutationSet):
            
            createMutationCsv(data[key], key, "accuracy")
            createMutationCsv(data[key], key, "jaccard")
            

    
    

if __name__ == '__main__':
    main()



