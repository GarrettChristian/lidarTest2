"""
fileIoUtil 
Handles the file operations to get and save labels and bins

@Author Garrett Christian
@Date 6/23/22
"""

import numpy as np
import os


# --------------------------------------------------------------------------


"""
Rejoins xyz & i bin file and label file semantics & instance
"""
def prepareToSave(xyz, intensity, semantics, instances):

    labelsCombined = (instances << 16) | (semantics & 0xFFFF)

    xyzi = np.c_[xyz, intensity]
    xyziFlat = xyzi.flatten()

    xyziFlat = xyziFlat.astype(np.float32)
    labelsCombined = labelsCombined.astype(np.int32)

    return xyziFlat, labelsCombined


"""
Saves a modified bin file and label file rejoining the intensity 
"""
def saveBinLabelPair(xyzi, labels, saveBinPath, saveLabelPath, fileName):
    binFile = saveBinPath + fileName + ".bin"
    labelFile = saveLabelPath + fileName + ".label"

    xyzi = xyzi.astype(np.float32)
    labels = labels.astype(np.int32)

    xyzi.tofile(binFile)
    labels.tofile(labelFile)


"""
openLabelBin
For a specific sequence and scene
Opens a bin and label file splitting between xyz, intensity, semantics, instances 
"""
def openLabelBin(pathVel, pathLabel, sequence, scene):

    folderNum = str(sequence).rjust(2, '0')
    currPathVel = pathVel + folderNum
    currPathLbl = pathLabel + folderNum

    binFile = currPathVel + "/velodyne/" + scene + ".bin"
    labelFile = currPathLbl + "/labels/" + scene + ".label"

    return openLabelBinFiles(binFile, labelFile)



"""
openLabelBinFiles
Opens a bin and label file splitting between xyz, intensity, semantics, instances 
"""
def openLabelBinFiles(binFile, labelFile):
    # Label
    label_arr = np.fromfile(labelFile, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    instances = label_arr >> 16 

    # Bin File
    pcdArr = np.fromfile(binFile, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    
    intensity = pcdArr[:, 3]
    pcdArr = np.delete(pcdArr, 3, 1)

    return pcdArr, intensity, semantics, instances


"""
Attempts to remove files
"""
def removeFiles(files):
    for file in files:
        try:
            os.remove(file)
        except OSError:
            print("File {} not found when calling remove".format(file))






