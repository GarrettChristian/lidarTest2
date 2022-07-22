"""
modelValsInitial
Get the base model predictions

@Author Garrett Christian
@Date 6/30/22
"""


from pymongo import MongoClient
import glob, os
import shutil
import argparse

from service.models.cylRunner import CylRunner
from service.models.spvRunner import SpvRunner
from service.models.salRunner import SalRunner
from service.models.sq3Runner import Sq3Runner
from service.models.polRunner import PolRunner
from service.models.ranRunner import RanRunner


# -------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    p.add_argument("-bins", 
        help="Path to the scenes", 
        nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
        default="/home/garrett/Documents/data/dataset/sequences/")
    p.add_argument("-stage", 
        help="Path to the stage location", 
        nargs='?', const="/home/garrett/Documents/data/tmp/dataset/sequences/00/velodyne/", 
        default="/home/garrett/Documents/data/tmp/dataset/sequences/00/velodyne/")
    p.add_argument("-pred", 
        help="Path to save the predictions made by the tools", 
        nargs='?', const="/home/garrett/Documents/data/resultsBase/", 
        default="/home/garrett/Documents/data/resultsBase/")
    p.add_argument("-model", 
        help="Model name", 
        nargs='?', const="sq3", 
        default="sq3")
    
    return p.parse_args()


def main():

    print("\n\n------------------------------")
    print("\n\nStarting Model Base Prediction Maker\n\n")

    args = parse_args() 
    binBasePath = args.bins
    stagePath = args.stage
    predBasePath = args.pred
    model = args.model

    for x in range(0, 11):

        folderNum = str(x).rjust(2, '0')

        savePreds = predBasePath + folderNum + "/" + model + "/"
        print("\n\n\nSave At {}".format(savePreds))

        # Current folder bins
        curFolder = binBasePath + folderNum + "/velodyne"
        print("Cur folder to predict {}".format(savePreds))

        # Prepare the stage dir
        print("Removing files in stage {}:".format(stagePath))
        filelist = glob.glob(os.path.join(stagePath, "*"))
        for f in filelist:
            os.remove(f)

        # Copy bins in current folder to stage
        print("Copy folder {} to stage:".format(folderNum))
        filesInFolder = glob.glob(os.path.join(curFolder, "*")) 
        for f in filesInFolder:
            shutil.copy(f, stagePath)

        stageSeqFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.normpath(stagePath))))
        
        print("Starting predictions on {}".format(folderNum))

        # Predict for the current folder
        modelRunner.runDar(stageSeqFolder, savePreds)

    print("\n\n\nDone")
        

if __name__ == '__main__':
    main()







