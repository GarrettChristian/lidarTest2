"""
globals 
Handles all data that needs to be shared in the tool
Performs the initial setup
Enums, paths, flags

@Author Garrett Christian
@Date 6/23/22
"""


import numpy as np
import glob, os
import shutil






pathVel = ""
pathLbl = ""
visualize = ""
mutationsEnabled = []


rotation = None
mirrorAxis = None
intensityChange = None


scaleLimit = 0

vehicles = set()


evalMutationFlag = True
saveMutationFlag = True


batchNum = 0
expectedNum = 0





batchId = str(shortuuid.uuid())


assetId = None

models = ["cyl", "spv", "sal"]

# ---------------------------------------------------------------







def init(args):
    # Paths for base data
    global binFiles
    global labelFiles
    global pathVel
    global pathLbl

    # Mutations to choose from
    global mutationsEnabled

    # Specific mutation arguments
    global rotation
    global mirrorAxis
    global intensityChange
    global assetId

    global scaleLimit

    global vehicles

    # Flags for convience 
    global evalMutationFlag
    global saveMutationFlag
    global visualize

    global saveAt

    global batchNum
    global expectedNum


    print("Running Setup")

    batchNum = int(args.b)
    expectedNum = int(args.count)
    print("Will run until {} successful mutations are obtained".format({expectedNum}))
    print("Batch evaluating and saving every {}".format({batchNum}))

    # Flags for disabling Saving and evaluation
    saveMutationFlag = args.ns
    evalMutationFlag = args.ne and saveMutationFlag
    if not saveMutationFlag:
        print("Saving disabled")
    if not evalMutationFlag:
        print("Evaluation disabled")

    # Flag to use open3d to visualize your changes
    visualize = args.vis

    threads = args.t
    print("Threads: {}".format(threads))

    # Set up the mutations that will be used in this run
    print("Selecting mutations to use")
    mutationsEnabled = prepareMutations(args.m)


    # Set up the bins and label files to randomly select from
    pathVel = args.path
    pathLbl = args.lbls
    binFiles, labelFiles = getBinsLabels(args.path, args.lbls, args.seq, args.scene)

    
    # Specific mutation arguments
    if (args.intensity):
        intensityChange = int(args.intensity)
    if (args.rotate):
        rotation = int(args.rotate)
    if (args.mirror):
        mirrorAxis = int(args.mirror)
    if (args.assetId):
        assetId = args.assetId

    # Limit for number of points for scale
    scaleLimit = args.scaleLimit


    # Only reset the save location if saving is enabled
    if (saveMutationFlag):
        print("Setting up result folder pipeline")
        setUpDataFolders(threads)


    





    




