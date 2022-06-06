
from glob import glob
import numpy as np
from enum import Enum
import glob, os
import shutil
import uuid


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
    32: 'motorcyclist',
    252: 'moving-car',
    253: 'moving-bicyclist',
    255: 'moving-motorcyclist',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}





# Enum of the different types of mutations supported
class Mutation(Enum):
    ADD_ROTATE = "ADD_ROTATE",
    ADD_MIRROR_ROTATE = "ADD_MIRROR_ROTATE",
    ADD_SCALE_ROTATE = "ADD_SCALE_ROTATE"
    SCENE_INTENSITY = "SCENE_INTENSITY"
    SCENE_REMOVE = "SCENE_REMOVE",
    SCENE_NOISE = "SCENE_NOISE"
    SCENE_REMOVE_TRANSLATE = "SCENE_REMOVE_TRANSLATE"
    SCENE_REMOVE_ROTATE = "SCENE_REMOVE_ROTATE"
    SCENE_SPARSIFY = "SCENE_SPARSIFY"
    SCENE_DENSIFY = "SCENE_DENSIFY"

# # Enum of the different types of mutations supported
# class Asset(Enum):
#     ADD = "ADD" # New Asset
#     SCENE = "SCENE" # Asset in a scene

# class Transformation(Enum):
#     ROTATE = "ROTATE",
#     MIRROR = "MIRROR",
#     INTENSITY = "INTENSITY"
#     REMOVE = "REMOVE",
#     NOISE = "NOISE"
#     TRANSLATE = "TRANSLATE"
    

# class Transformations(Enum):
#     ADD_ROTATE = "ADD_ROTATE",
#     ADD_MIRROR_ROTATE = "ADD_MIRROR_ROTATE",
#     ADD_SCALE_ROTATE = "ADD_SCALE_ROTATE"
#     SCENE_INTENSITY = "SCENE_INTENSITY"
#     SCENE_REMOVE = "SCENE_REMOVE",
#     SCENE_NOISE = "SCENE_NOISE"
#     SCENE_REMOVE_TRANSLATE = "SCENE_REMOVE_TRANSLATE"
#     SCENE_REMOVE_ROTATE = "SCENE_REMOVE_ROTATE"
#     SCENE_SPARSIFY = "SCENE_SPARSIFY"
#     SCENE_DENSIFY = "SCENE_DENSIFY"


# # Transformations for add / copy
# class TransformationsZ(Enum):
#     # LOCAL_ROTATE = "LOCAL_ROTATE_IN_SCENE"
#     ROTATE = "ROTATE"
#     # TRANSLATE = "TRANSLATE"
#     # MIRROR = "MIRROR"

#     ROTATE_MIRROR = "ROTATE_MIRROR"
#     INTENSITY = "INTENSITY"
#     REMOVE = "REMOVE"



binFiles = []
labelFiles = []
path = ""
visualize = ""
mutationsEnabled = []
tranformationsEnabled = []


rotation = None
mirrorAxis = None
intensityChange = None


vehicles = set()


evalMutationFlag = True
saveMutationFlag = True


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


batchId = str(uuid.uuid4())


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



# def prepareTransformations(transformationsGiven):
#      # Get mutations to use
#     transformations = []
#     if (transformationsGiven != None):
#         for transformation in transformationsGiven.split(","):
#             try:
#                 transformation = Transformations[transformation]
#             except KeyError:
#                 print("%s is not a valid option" % (transformation))
#                 exit()
#             transformations.append(transformation)
#     else:
#         transformations = list(Transformations)

#     print("Mutations: {}".format(transformations))
#     return transformations



"""
Setup step to get all potential bin files
"""
def getBinsLabels(path, sequence, scene):
    print("Parsing {} :".format(path))
    print("Collecting Labels and Bins for sequence {}".format(sequence))

    binFilesRun = []
    labelFilesRun = []

    # Specific Scene
    if (scene):
        currPath = path + str(sequence).rjust(2, '0')

        labelFilesRun = [currPath + "/labels/" + scene + ".label"]
        binFilesRun = [currPath + "/velodyne/" + scene + ".bin"]
    # Any scene
    else:
        for sequenceNum in sequence:
        
            folderNum = str(sequenceNum).rjust(2, '0')
            currPath = path + folderNum

            labelFilesSequence = np.array(glob.glob(currPath + "/labels/*.label", recursive = True))
            binFilesSequence = np.array(glob.glob(currPath + "/velodyne/*.bin", recursive = True))
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
    #     os.makedirs(doneDir)
    
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
    global binFiles
    global labelFiles
    global path
    global visualize
    global mutationsEnabled
    global tranformationsEnabled

    global rotation
    global mirrorAxis
    global intensityChange

    global vehicles

    global evalMutationFlag
    global saveMutationFlag


    print("Running Setup")

    binFiles = []
    labelFiles = []
    mutationsEnabled = []
    tranformationsEnabled = []
    path = ""
    visualize = ""

    # Saving and evaluation
    saveMutationFlag = args.ns
    evalMutationFlag = args.ne and saveMutationFlag
    if not saveMutationFlag:
        print("Saving disabled")
    if not evalMutationFlag:
        print("Evaluation disabled")

    threads = args.t
    print("Threads: {}".format(threads))

    print("Setting up result folder pipeline")
    setUpDataFolders(threads)

    print("Selecting mutations to use")
    mutationsEnabled = prepareMutations(args.m)
    # tranformationsEnabled = prepareTransformations(args.t)


    path = args.path
    binFiles, labelFiles = getBinsLabels(path, args.seq, args.scene)
    visualize = args.vis


    if (args.intensity):
        intensityChange = int(args.intensity)
    if (args.rotate):
        rotation = int(args.rotate)
    if (args.mirror):
        mirrorAxis = int(args.mirror)


    for vehicle in instancesVehicle.keys():
        vehicles.add(vehicle)



    





    




