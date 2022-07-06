"""
redoMutation
Recreates the mutation results for specific details

@Author Garrett Christian
@Date 6/29/22
"""

import argparse
import os
import shutil

import controllers.mutationTool.mutationRunner as mutationRunner

from domain.toolSessionManager import SessionManager

from data.mutationDetailsRepository import DetailsRepository
from data.assetRepository import AssetRepository
import data.fileIoUtil as fileIoUtil


# -------------------------------------------------------------

PAGE_LIMIT = 1000

# -------------------------------------------------------------


"""
batchRecreation
Get details by a batch Id then save
"""
def batchRecreation(batchId, sessionManager, saveVel, saveLabel):
    print("Recreating batch {}".format(batchId))

    # Connect to collections
    detailsRepository = DetailsRepository(sessionManager.mongoConnect)
    assetRepo = AssetRepository(sessionManager.binPath, sessionManager.binPath, sessionManager.mongoConnect)

    page = 0
    parsed = 0
    successNum = 0
    
    total = detailsRepository.countMutationDetailsBatchId(batchId)
    print("Total count for batchId {} is {}".format(batchId, total))

    # Parse the details in the batch
    while (parsed < total):
        mutationDetails = []
        bins = []
        labels = []

        page += 1
        print("Parsing details on page {} of size {}".format(page, PAGE_LIMIT))
        detailsCursor = detailsRepository.getMutationDetailsPaged(batchId, page, PAGE_LIMIT)

        detailsList = list(detailsCursor)
        count = len(detailsList)
        parsed += count
        print("Found {} on page {}, {}/{}".format(count, page, parsed, total))
        for details in detailsList:
            success = False
            success, _, xyziFinal, labelFinal = recreateOne(details, assetRepo, sessionManager)

            # Save batch
            if success:
                successNum += 1
                mutationDetails.append(details)
                bins.append(xyziFinal)
                labels.append(labelFinal)
            else:
                print("Could not successfully recreate {} in batch {}".format(details["_id"], batchId))

        # Catch in case there was an error
        if (count == 0):
            total = total

        # Save bin and labels
        for index in range(0, len(mutationDetails)):
            fileIoUtil.saveBinLabelPair(bins[index], labels[index], saveVel, saveLabel, mutationDetails[index]["_id"])



"""
mutationRecreation
Get a detail by Id then save
"""
def mutationRecreation(mutationId, sessionManager, saveVel, saveLabel):
    print("Recreating mutation {}".format(mutationId))

    # Get one recreate
    detailsRepository = DetailsRepository(sessionManager.mongoConnect)

    details = detailsRepository.getMutationDetailsById(mutationId)

    assetRepo = AssetRepository(sessionManager.binPath, sessionManager.binPath, sessionManager.mongoConnect)
    success, _, xyziFinal, labelFinal = recreateOne(details, assetRepo, sessionManager)

    # Save
    if (success):
        fileIoUtil.saveBinLabelPair(xyziFinal, labelFinal, saveVel, saveLabel, details["_id"])
    else:
        print("Could not successfully recreate {}".format(mutationId))



"""
recreateOne
Given a detail prepare the session to recreate this mutation then perform the mutation
"""
def recreateOne(details, assetRepo, sessionManager):
    print("Recreating {}".format(details["_id"]))

    # Base scene
    sessionManager.scene = details.get("baseScene", None)
    sessionManager.sequence = details.get("baseSequence", None)

    # Asset
    sessionManager.assetId = details.get("asset", None)

    # Rotate
    sessionManager.rotation = details.get("rotate", None)
    # Mirror
    sessionManager.mirrorAxis = details.get("mirror", None)
    # Scale
    sessionManager.scaleAmount = details.get("scale", None)
    # Sign Change
    sessionManager.signChange = details.get("sign", None)
    # Deform
    sessionManager.deformPercent = details.get("deformPercent", None)
    sessionManager.deformPoint = details.get("deformPoint", None)
    sessionManager.deformMu = details.get("deformMu", None)
    sessionManager.deformSigma = details.get("deformSigma", None)
    sessionManager.deformSeed = details.get("deformSeed", None)
    # Intensity
    sessionManager.intensity = details.get("intensity", None)


    return mutationRunner.performMutation(details["mutation"], assetRepo, sessionManager)




# -------------------------------------------------------------
# Arguments


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')

    # Required params
    p.add_argument("-binPath", 
        help="Path to the sequences folder of LiDAR scan bins", 
        nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
        default="/home/garrett/Documents/data/dataset/sequences/")

    p.add_argument("-labelPath", 
        help="Path to the sequences label files should corrispond with velodyne", 
        nargs='?', const="/home/garrett/Documents/data/dataset2/sequences/", 
        default="/home/garrett/Documents/data/dataset2/sequences/")

    p.add_argument("-mdb", 
        help="Path to the connection string for mongo", 
        nargs='?', const="/home/garrett/Documents/lidarTest2/mongoconnect.txt", 
        default="/home/garrett/Documents/lidarTest2/mongoconnect.txt")

    # Tool configurable params
    p.add_argument("-saveAt", 
        help="Location to save the tool output", 
        nargs='?', const=os.getcwd(), 
        default=os.getcwd())

    p.add_argument('-vis', 
        help='Visualize with Open3D',
        action='store_true', default=False)

    p.add_argument("-scaleLimit", 
        help="Limit to the number of points for scale", 
        nargs='?', const=10000, default=10000)

    # Recreation options
    p.add_argument("-batchId", 
        help="Batch to rerun (either batchId or mutationId will override tool)", 
        default="",
        required=False)

    p.add_argument("-mutationId", 
        help="Mutation to rerun (either batchId or mutationId will override tool)", 
        default="",
        required=False)
    
    return p.parse_args()

    


# ----------------------------------------------------------

def main():

    print("\n\n------------------------------")
    print("\n\nStarting Semantic LiDAR Fuzzer\n\n")
    
    # Get arguments 
    args = parse_args()
    
    # Create a session manager
    sessionManager = SessionManager(args, recreation=True)

    # Save folders
    outputDir = args.saveAt + "/output"
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)
    # Velodyne
    saveVel = outputDir + "/velodyne/"
    if os.path.exists(saveVel):
        shutil.rmtree(saveVel)
    os.mkdir(saveVel)
    # Label
    saveLabel = outputDir + "/labels/"
    if os.path.exists(saveLabel):
        shutil.rmtree(saveLabel)
    os.mkdir(saveLabel)

    # Start the mutation tool
    print("Starting Mutation Recreation")

    # Recreate a batch or specific mutation
    if (args.batchId != ""):
        batchRecreation(args.batchId, sessionManager, saveVel, saveLabel)
    elif (args.mutationId != ""):
        mutationRecreation(args.mutationId, sessionManager, saveVel, saveLabel)
    else:
        print("BatchId or MutationId required")



   


if __name__ == '__main__':
    main()









