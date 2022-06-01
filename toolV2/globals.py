
from glob import glob
import numpy as np
from enum import Enum
import glob, os


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
    ADD = "ADD" # New Asset
    SCENE = "SCENE" # make a copy of the asset
    # MOVE = "MOVE" # remove the asset and place somewhere else

# Enum of the different types of mutations supported
class Asset(Enum):
    ADD = "ADD" # New Asset
    SCENE = "SCENE" # Asset in a scene

class Transformation(Enum):
    ROTATE = "ROTATE",
    MIRROR = "MIRROR",
    INTENSITY = "INTENSITY"
    REMOVE = "REMOVE",
    NOISE = "NOISE"
    TRANSLATE = "TRANSLATE"
    

class Transformations(Enum):
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




# Transformations for add / copy
class TransformationsZ(Enum):
    # LOCAL_ROTATE = "LOCAL_ROTATE_IN_SCENE"
    ROTATE = "ROTATE"
    # TRANSLATE = "TRANSLATE"
    # MIRROR = "MIRROR"

    ROTATE_MIRROR = "ROTATE_MIRROR"
    INTENSITY = "INTENSITY"
    REMOVE = "REMOVE"



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


saveLabelPath = ""
saveBinPath = ""


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



def prepareTransformations(transformationsGiven):
     # Get mutations to use
    transformations = []
    if (transformationsGiven != None):
        for transformation in transformationsGiven.split(","):
            try:
                transformation = Transformations[transformation]
            except KeyError:
                print("%s is not a valid option" % (transformation))
                exit()
            transformations.append(transformation)
    else:
        transformations = list(Transformations)

    print("Mutations: {}".format(transformations))
    return transformations



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

    binFiles = []
    labelFiles = []
    mutationsEnabled = []
    tranformationsEnabled = []
    path = ""
    visualize = ""
    
    saveLabelPath = ""
    saveBinPath = ""
    saveMutationPath = ""

    mutationsEnabled = prepareMutations(args.m)
    tranformationsEnabled = prepareTransformations(args.t)
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



    





    




