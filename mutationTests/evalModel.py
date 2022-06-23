


import argparse
import numpy as np
from np_ioueval import iouEval
import sys

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

map_moving = {
    252: 10, # 'moving-car',
    253: 31, # 'moving-bicyclist',
    254: 30, # 'moving-person',
    255: 32, # 'moving-motorcyclist',
    256: 16, # 'moving-on-rails',
    257: 13, # 'moving-bus',
    258: 18, # 'moving-truck',
    259: 20  # 'moving-other-vehicle'
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



# def parse_args():
#     p = argparse.ArgumentParser(
#         description='Model Runner')
#     p.add_argument(
#         'label', help='actual labels')
#     p.add_argument(
#         'labelTest', help='labels to test against')

#     return p.parse_args()


def V1():

    print("\n\n------------------------------")
    print("\n\nStarting Model Eval\n\n")

    # args = parse_args()
    labelActual = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/test.label"
    labelModel = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/000000.label"

    labelArrActual = np.fromfile(labelActual, dtype=np.int32)
    semantics = labelArrActual & 0xFFFF

    labelArrModel = np.fromfile(labelModel, dtype=np.int32)
    semanticsModel = labelArrModel & 0xFFFF
    
    classes = name_label_mapping.keys()

    print(semantics)
    print(labelArrModel)

    print(np.shape(semantics))
    print(np.shape(semanticsModel))

    uniqueActual = set()
    for lbl in semantics:
        uniqueActual.add(lbl)

    unique = set()
    for lbl in semanticsModel:
        if (lbl not in uniqueActual):
            unique.add(lbl)

    print(uniqueActual)
    print(unique)
    for un in unique:
        if un in name_label_mapping.keys():
            print(name_label_mapping[un])
        else:
            print("not found", un)



    for label in semantics:
        if label in map_moving.keys():
            label = map_moving[label]


    uniqueActual = set()
    for lbl in semantics:
        uniqueActual.add(lbl)
    print(uniqueActual)

    

    jaSum = 0
    for classNum in classes:

        actualClass = semantics == classNum
        modelPredClass = semanticsModel == classNum

        tpMask = actualClass & modelPredClass
        tp = np.sum(tpMask)
        
        fpMask = np.logical_not(actualClass) & modelPredClass
        fp = np.sum(fpMask)

        fnMask = actualClass & np.logical_not(modelPredClass)
        fn = np.sum(fnMask)
        
        if ((tp + fp + fn) > 0):
            jaSum += tp / (tp + fp + fn)
        else:
            jaSum += 1

        print(name_label_mapping[classNum])
        print(classNum)
        print("Actual", np.sum(actualClass))
        print("Model ", np.sum(modelPredClass))
        print("tp", tp)
        print("fp", fp)
        print("fn", fn)
        print(jaSum)
        print()


    result = jaSum / len(classes)
    print(result)



# https://github.com/PRBonn/semantic-kitti-api/blob/master/evaluate_semantics.py
def main():

    # label_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/000000Act.label"
    # pred_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/000000Cyl.label"
    label_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/bus.label"
    pred_file = "/Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/eval/busCy.label"

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

    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=name_label_mapping[learning_map_inv[i]], jacc=jacc))

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



if __name__ == '__main__':
    main()






