


import numpy as np

import globals


"""
Saves a modified bin file and label file rejoining the intensity 
"""
def saveToBin(xyz, intensity, semantics, labelsInstance, file):
    binFile = globals.saveBinPath + file + ".bin"
    labelFile = globals.saveLabelPath + file + ".label"

    labelsCombined = (labelsInstance << 16) | (semantics & 0xFFFF)

    xyzi = np.c_[xyz, intensity]
    xyziFlat = xyzi.flatten()

    xyziFlat = xyziFlat.astype(np.float32)
    labelsCombined = labelsCombined.astype(np.int32)

    xyziFlat.tofile(binFile)
    labelsCombined.tofile(labelFile)


"""
Opens a bin and label file splitting between xyz, intensity, semantics, instances 
"""
def openLabelBin(path, sequence, scene):

    currPath = path + str(sequence).rjust(2, '0')

    labelFile = currPath + "/labels/" + scene + ".label"
    binFile = currPath + "/velodyne/" + scene + ".bin"

    return openLabelBinFiles(binFile, labelFile)



"""
Opens a bin and label file splitting between xyz, intensity, semantics, instances 
"""
def openLabelBinFiles(binFile, labelFile):
    # Label
    label_arr = np.fromfile(labelFile, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 

    # Bin File
    pcdArr = np.fromfile(binFile, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    
    intensity = pcdArr[:, 3]
    pcdArr = np.delete(pcdArr, 3, 1)

    return pcdArr, intensity, semantics, labelInstance











