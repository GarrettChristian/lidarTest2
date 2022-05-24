

from pymongo import MongoClient
import glob, os
import numpy as np
import open3d as o3d
import math
import random
import argparse


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
        20: 'person',
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

centerCamPoint = np.array([0, 0, 0.3])

# ------------------------------------------------------------------

def viewOne(binFileName, labelsFileName):
    print(binFileName)
    print(labelsFileName)

    # Label
    label_arr = np.fromfile(labelsFileName, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 

    # Bin File
    pcdArr = np.fromfile(binFileName, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    
    intensity = pcdArr[:, 3]
    pcdArr = np.delete(pcdArr, 3, 1)
   
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArr)
    
    colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
    colors[:, 0] = intensity
    
    pcdScene.colors = o3d.utility.Vector3dVector(colors)


    o3d.visualization.draw_geometries([pcdScene])
    
def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    # p.add_argument(
    #     'binLocation', help='Path to the dir to test')
    p.add_argument(
        "-num", help="specific scenario number provide full ie 002732")
    p.add_argument(
        "-scene", help="Scene number provide as 1 rather than 01 (default all labeled)", 
        nargs='?', const=0, default=range(0, 11))

    return p.parse_args()
    

def main():

    print("\n\n------------------------------")
    print("\n\nStarting open3D viewer\n\n")

    path = "/home/garrett/Documents/data/dataset/sequences/"

    print("Parsing {} :".format(path))

    num = 0

    args = parse_args()

    # Get scenes

    print("Collecting Labels and Bins for scenes {}".format(args.scene))

    binFiles = []
    labelFiles = []

    
            
    print("Starting Visualization")


    if (args.num):
        currPath = path + str(args.scene).rjust(2, '0')

        labelFiles = [currPath + "/labels/" + args.num + ".label"]
        binFiles = [currPath + "/velodyne/" + args.num + ".bin"]
    
    else:
        for sceneNum in args.scene:
        
            folderNum = str(sceneNum).rjust(2, '0')
            currPath = path + folderNum

            labelFilesScene = np.array(glob.glob(currPath + "/labels/*.label", recursive = True))
            binFilesScene = np.array(glob.glob(currPath + "/velodyne/*.bin", recursive = True))
            print("Parsing Scene {}".format(folderNum))

            # Sort
            labelFilesScene = sorted(labelFilesScene)
            binFilesScene = sorted(binFilesScene)
            
            for labelFile in labelFilesScene:
                labelFiles.append(labelFile)
            
            for binFile in binFilesScene:
                binFiles.append(binFile)

    try:
        idx = random.choice(range(len(labelFiles)))
        # for idx in range(len(labelFiles)):
        print(num, binFiles[idx])
        num += 1
        viewOne(binFiles[idx], labelFiles[idx])
        

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")



if __name__ == '__main__':
    main()



