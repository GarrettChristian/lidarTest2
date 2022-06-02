

from pymongo import MongoClient
import glob, os
import numpy as np
import open3d as o3d
from np_ioueval import iouEval
import sys
from os.path import basename


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
    # 252: 'moving-car',
    # 253: 'moving-bicyclist',
    # 254: 'moving-person',
    # 255: 'moving-motorcyclist',
    # 256: 'moving-on-rails',
    # 257: 'moving-bus',
    # 258: 'moving-truck',
    # 259: 'moving-other-vehicle'
}

# https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
learning_map = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,    # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

learning_map_inv = { # inverse of previous map
    0: 0,      # "unlabeled", and others ignored
    1: 10,     # "car"
    2: 11,     # "bicycle"
    3: 15,     # "motorcycle"
    4: 18,     # "truck"
    5: 20,     # "other-vehicle"
    6: 30,     # "person"
    7: 31,     # "bicyclist"
    8: 32,     # "motorcyclist"
    9: 40,     # "road"
    10: 44,    # "parking"
    11: 48,    # "sidewalk"
    12: 49,    # "other-ground"
    13: 50,    # "building"
    14: 51,    # "fence"
    15: 70,    # "vegetation"
    16: 71,    # "trunk"
    17: 72,    # "terrain"
    18: 80,    # "pole"
    19: 81,    # "traffic-sign"
}

learning_ignore = { # Ignore classes
    0: True,      # "unlabeled", and others ignored
    1: False,     # "car"
    2: False,     # "bicycle"
    3: False,     # "motorcycle"
    4: False,     # "truck"
    5: False,     # "other-vehicle"
    6: False,     # "person"
    7: False,     # "bicyclist"
    8: False,     # "motorcyclist"
    9: False,     # "road"
    10: False,    # "parking"
    11: False,    # "sidewalk"
    12: False,    # "other-ground"
    13: False,    # "building"
    14: False,    # "fence"
    15: False,    # "vegetation"
    16: False,    # "trunk"
    17: False,    # "terrain"
    18: False,    # "pole"
    19: False    # "traffic-sign"
}


# https://github.com/PRBonn/semantic-kitti-api/blob/master/evaluate_semantics.py
def defEvalV1(label_file, pred_file, seq, model):

    print()
    print(label_file)
    print(pred_file)

    # label_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/000000Act.label"
    # pred_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/000000Cyl.label"
    # label_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/bus.label"
    # pred_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/busCy.label"


    fileName = basename(label_file)
    fileName = fileName.replace(".label", "")

    numClasses = len(learning_map_inv)

    # make lookup table for mapping
    maxkey = max(learning_map.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(learning_map.keys())] = list(learning_map.values())

    # create evaluator
    ignore = []
    for cl, ign in learning_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    evaluator = iouEval(numClasses, ignore)
    evaluator.reset()

    label = np.fromfile(label_file, dtype=np.int32)
    label = label.reshape((-1))  # reshape to vector
    label = label & 0xFFFF       # get lower half for semantics

    # open prediction
    pred = np.fromfile(pred_file, dtype=np.int32)
    pred = pred.reshape((-1))    # reshape to vector
    pred = pred & 0xFFFF         # get lower half for semantics

    label = remap_lut[label] # remap to xentropy format
    pred = remap_lut[pred] # remap to xentropy format

    # add single scan to evaluation
    evaluator.addBatch(pred, label)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    results = {}


    results["_id"] = model + "-" + seq + "-" + fileName
    results["sequence"] = seq
    results["scene"] = fileName
    results["model"] = model
    results["jaccard"] = m_jaccard.item()
    results["accuracy"] = m_accuracy.item()

    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=name_label_mapping[learning_map_inv[i]], jacc=jacc))

            results[name_label_mapping[learning_map_inv[i]]] = jacc

    # print for spreadsheet
    print("*" * 80)
    print("below can be copied straight for paper table")
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()


    return results
    



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



def main():

    print("\n\n------------------------------")
    print("\n\nStarting Model Evaluation Upload\n\n")

    print("Connecting to Mongo")
    mdb = mongoConnect()
    mdbCol = mdb["base_accuracy"]
    print("Connected")




    # label_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/bus.label"
    # pred_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/busCy.label"

    labelBasePath = "/home/garrett/Documents/data/dataset/sequences/"
    predBasePath = "/home/garrett/Documents/data/resultsBase/"
    
    for x in range(0, 11):

        add = []

        folderNum = str(x).rjust(2, '0')

        labelPath = labelBasePath + folderNum + "/labels/"
        predPathCyl = predBasePath + folderNum + "/cyl/"
        predPathSal = predBasePath + folderNum + "/sal/"
        predPathSpv = predBasePath + folderNum + "/spv/"

        labelFiles = glob.glob(labelPath + "*.label")
        predFilesCyl = glob.glob(predPathCyl + "*.label")
        predFilesSal = glob.glob(predPathSal + "*.label")
        predFilesSpv = glob.glob(predPathSpv + "*.label")
        
        # Order the update files cronologically
        labelFiles = sorted(labelFiles)
        predFilesCyl = sorted(predFilesCyl)    
        predFilesSal = sorted(predFilesSal)    
        predFilesSpv = sorted(predFilesSpv)        
        for index in range(0, len(labelFiles)):

            cyl = defEvalV1(labelFiles[index], predFilesCyl[index], folderNum, "cyl")
            sal = defEvalV1(labelFiles[index], predFilesSal[index], folderNum, "sal")
            spv = defEvalV1(labelFiles[index], predFilesSpv[index], folderNum, "spv")

            add.append(cyl)
            add.append(sal)
            add.append(spv)

        mdbCol.insert_many(add)


if __name__ == '__main__':
    main()

































