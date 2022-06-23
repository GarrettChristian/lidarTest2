"""
globals 
Handles all data that needs to be shared in the tool
Performs the initial setup
Enums, paths, flags

@Author Garrett Christian
@Date 6/23/22
"""


import numpy as np
from enum import Enum
import glob, os
import shutil
import shortuuid

# -------------------------------------------------------

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

instances = {
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

instancesVehicle = {
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    18: 'truck',
    20: 'other-vehicle',
    # 31: 'bicyclist',
    # 32: 'motorcyclist',
    252: 'moving-car',
    # 253: 'moving-bicyclist',
    # 255: 'moving-motorcyclist',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}


instancesWalls = {
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    71: 'trunk',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
}


color_map_alt = { # rgb
  0 : [0, 0, 0],
  1 : [0, 0, 0],
  10: [100, 150, 245],
  11: [100, 230, 245],
  13: [0, 0, 255],
  15: [30, 60, 150],
  16: [0, 0, 255],
  18: [80, 30, 180],
  20: [0, 0, 255],
  30: [255, 30, 30],
  31: [255, 40, 200],
  32: [150, 30, 90],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [175, 0, 75],
  50: [255, 200, 0],
  51: [255, 120, 50],
  52: [0, 0, 0],
  60: [255, 0, 255],
  70: [0, 175, 0],
  71: [135, 60, 0],
  72: [150, 240, 80],
  80: [255, 240, 150],
  81: [255, 0, 0],
  99: [0, 0, 0],
  252: [100, 150, 245],
  253: [255, 40, 200],
  254: [255, 30, 30],
  255: [150, 30, 90],
  256: [0, 0, 255],
  257: [0, 0, 255],
  258: [80, 30, 180],
  259: [0, 0, 255],
}


# Enum of the different types of mutations supported
class Mutation(Enum):
    ADD_ROTATE = "ADD_ROTATE",
    ADD_MIRROR_ROTATE = "ADD_MIRROR_ROTATE",
    SCENE_DEFORM = "SCENE_DEFORM",
    SCENE_INTENSITY = "SCENE_INTENSITY",
    SCENE_REMOVE = "SCENE_REMOVE",
    SIGN_REPLACE = "SIGN_REPLACE",
    SCENE_SCALE = "SCENE_SCALE"


# Enum of the different types of sign replacements
class Signs(Enum):
    YEILD = "YEILD",
    CROSSBUCK = "CROSSBUCK",
    WARNING = "WARNING",
    SPEED = "SPEED",
    STOP = "STOP",
    


binFiles = []
labelFiles = []
pathVel = ""
pathLbl = ""
visualize = ""
mutationsEnabled = []


rotation = None
mirrorAxis = None
intensityChange = None


vehicles = set()


evalMutationFlag = True
saveMutationFlag = True


batchNum = 0
expectedNum = 0


# Paths to the directories where specific things are stored
saveAt = ""
stageDir = ""
dataRoot = ""
resultDir = ""
currentVelDir = ""
doneVelDir = ""
resultCylDir = ""
resultSpvDir = ""
resultSalDir = ""
evalDir = ""
doneLabelActualDir = ""
doneLabelCylDir = ""
doneLabelSpvDir = ""
doneLabelSalDir = ""
dataDir = ""
doneLabelDir = ""


batchId = str(shortuuid.uuid())


assetId = None

models = ["cyl", "spv", "sal"]

# ---------------------------------------------------------------


def prepareMutations(mutationsGiven):
     # Get mutations to use
    mutations = []
    if (mutationsGiven != None):
        for mutation in mutationsGiven.split(","):
            try:
                mutantToAdd = Mutation[mutation]
            except KeyError:
                print("%s is not a valid option" % (mutation))
                exit()
            mutations.append(mutantToAdd)
    else:
        mutations = list(Mutation)

    print("Mutations: {}".format(mutations))
    return mutations

"""
Setup step to get all potential bin files
"""
def getBinsLabels(pathVel, pathLabel, sequence, scene):
    print("Parsing {} bins and {} labels :".format(pathVel, pathLabel))
    print("Collecting Labels and Bins for sequence {}".format(sequence))

    binFilesRun = []
    labelFilesRun = []

    # Specific Scene
    if (scene):
        folderNum = str(sequenceNum).rjust(2, '0')
        currPathVel = pathVel + folderNum
        currPathLbl = pathLabel + folderNum

        labelFilesRun = [currPathVel + "/labels/" + scene + ".label"]
        binFilesRun = [currPathLbl + "/velodyne/" + scene + ".bin"]

    # Any scene
    else:
        for sequenceNum in sequence:
        
            folderNum = str(sequenceNum).rjust(2, '0')
            currPathVel = pathVel + folderNum
            currPathLbl = pathLabel + folderNum

            labelFilesSequence = np.array(glob.glob(currPathLbl + "/labels/*.label", recursive = True))
            binFilesSequence = np.array(glob.glob(currPathVel + "/velodyne/*.bin", recursive = True))
            print("Parsing Scene {}".format(folderNum))
            

            # Sort
            labelFilesSequence = sorted(labelFilesSequence)
            binFilesSequence = sorted(binFilesSequence)
            
            for labelFile in labelFilesSequence:
                labelFilesRun.append(labelFile)
            
            for binFile in binFilesSequence:
                binFilesRun.append(binFile)

    return binFilesRun, labelFilesRun


def setUpDataFolders(threads):
    global stageDir
    global dataRoot
    global resultDir
    global currentVelDir
    global doneVelDir
    global resultCylDir
    global resultSpvDir
    global resultSalDir
    global evalDir
    global doneLabelDir
    global doneLabelActualDir
    global doneLabelCylDir 
    global doneLabelSpvDir
    global doneLabelSalDir
    global dataDir

    curDir = os.getcwd()

    """
    /data
        /staging
            /velodyne0
            /labels0
        /current/dataset/sequences/00
            /velodyne
        /results
            /cyl
            /spv
            /sal/sequences/00
        /eval
            /labels0
                /cyl
                /spv
                /sal
        /done
            /velodyne
            /labels
                /actual
                /cyl
                /spv
                /sal
    """

    # make a top level data dir
    dataDir = curDir + "/data"
    isExist = os.path.exists(dataDir)
    if not isExist:
        os.makedirs(dataDir)

    """
    /data
        /staging
                /velodyne0
                /labels0
    """

    # staging
    stageDir = dataDir + "/staging"
    isExist = os.path.exists(stageDir)
    if not isExist:
        os.makedirs(stageDir)

    for thread in range(0, threads):
        # staging vel
        stageVel = stageDir + "/velodyne" + str(thread)
        if os.path.exists(stageVel):
            shutil.rmtree(stageVel)
            print("Removing {}".format(stageVel))
        os.makedirs(stageVel)

        # staging label
        stagelabel = stageDir + "/labels" + str(thread)
        if os.path.exists(stagelabel):
            shutil.rmtree(stagelabel)
            print("Removing {}".format(stagelabel))
        os.makedirs(stagelabel)

    """
    /data
        /current/dataset/sequences/00
            /velodyne
    """

    # current
    dataRoot = dataDir + "/current/dataset"
    currentDir = dataRoot + "/sequences/00"
    os.makedirs(currentDir, exist_ok=True)
    currentVelDir = currentDir + "/velodyne"
    if os.path.exists(currentVelDir):
        shutil.rmtree(currentVelDir)
        print("Removing {}".format(currentVelDir))
    os.makedirs(currentVelDir)

    """
    /data
        /results
            /cyl
            /spv
            /sal/sequences/00
    """

    # results
    resultDir = dataDir + "/results"
    isExist = os.path.exists(resultDir)
    if not isExist:
        os.makedirs(resultDir)
    
    # Cyl
    resultCylDir = resultDir + "/cyl"
    if os.path.exists(resultCylDir):
        shutil.rmtree(resultCylDir)
        print("Removing {}".format(resultCylDir))
    os.makedirs(resultCylDir)

    # Spv
    resultSpvDir = resultDir + "/spv"
    if os.path.exists(resultSpvDir):
        shutil.rmtree(resultSpvDir)
        print("Removing {}".format(resultSpvDir))
    os.makedirs(resultSpvDir)

    # Sal
    resultSalDir = resultDir + "/sal/sequences/00"
    if os.path.exists(resultDir + "/sal"):
        shutil.rmtree(resultDir + "/sal", ignore_errors=True)
        print("Removing {}".format(resultSalDir))
    os.makedirs(resultSalDir, exist_ok=True)


    """
    /data
        /eval
            /labels0
                /cyl
                /spv
                /sal
    """

    # eval
    evalDir = dataDir + "/eval"
    if os.path.exists(evalDir):
        shutil.rmtree(evalDir, ignore_errors=True)
        print("Removing {}".format(evalDir))
    os.makedirs(evalDir, exist_ok=True)

    for thread in range(0, threads):
        # staging vel
        labelThreadDir = evalDir + "/label" + str(thread)
        os.makedirs(labelThreadDir)
        # cyl
        labelThreadCylDir = labelThreadDir + "/cyl"
        os.makedirs(labelThreadCylDir)
        # spv
        labelThreadSpvDir = labelThreadDir + "/spv"
        os.makedirs(labelThreadSpvDir)
        # sal
        labelThreadSalDir = labelThreadDir + "/sal"
        os.makedirs(labelThreadSalDir)

    """
    /data
        /done
            /velodyne
            /labels
                /actual
                /cyl
                /spv
                /sal
    """

    # done
    doneDir = dataDir + "/done"
    if os.path.exists(doneDir):
        shutil.rmtree(doneDir, ignore_errors=True)
        print("Removing {}".format(doneDir))
    # isExist = os.path.exists(doneDir)
    # if not isExist:
    os.makedirs(doneDir)
    
    # done
    doneVelDir = doneDir + "/velodyne"
    isExist = os.path.exists(doneVelDir)
    if not isExist:
        os.makedirs(doneVelDir)

    # labels
    doneLabelDir = doneDir + "/labels"
    isExist = os.path.exists(doneLabelDir)
    if not isExist:
        os.makedirs(doneLabelDir)

    # labels done
    doneLabelActualDir = doneLabelDir + "/actual"
    isExist = os.path.exists(doneLabelActualDir)
    if not isExist:
        os.makedirs(doneLabelActualDir)
    # cyl
    doneLabelCylDir = doneLabelDir + "/cyl"
    isExist = os.path.exists(doneLabelCylDir)
    if not isExist:
        os.makedirs(doneLabelCylDir)
    # spv
    doneLabelSpvDir = doneLabelDir + "/spv"
    isExist = os.path.exists(doneLabelSpvDir)
    if not isExist:
        os.makedirs(doneLabelSpvDir)
    # sal
    doneLabelSalDir = doneLabelDir + "/sal"
    isExist = os.path.exists(doneLabelSalDir)
    if not isExist:
        os.makedirs(doneLabelSalDir)





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

    global vehicles

    # Flags for convience 
    global evalMutationFlag
    global saveMutationFlag
    global visualize

    global saveAt

    global batchNum
    global expectedNum


    print("Running Setup")


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

    batchNum = int(args.b)
    expectedNum = int(args.count)
    
    # Specific mutation arguments
    if (args.intensity):
        intensityChange = int(args.intensity)
    if (args.rotate):
        rotation = int(args.rotate)
    if (args.mirror):
        mirrorAxis = int(args.mirror)
    if (args.assetId):
        assetId = args.assetId


    for vehicle in instancesVehicle.keys():
        vehicles.add(vehicle)

    # Only reset the save location if saving is enabled
    if (saveMutationFlag):
        print("Setting up result folder pipeline")
        setUpDataFolders(threads)


    





    




