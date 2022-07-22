"""
modelPredTester
Script to help test the models

@Author Garrett Christian
@Date 7/22/22
"""


from pymongo import MongoClient
import glob, os
import shutil
import argparse

from domain.modelConstants import Models
import domain.config as config

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
    p.add_argument("-data", 
        help="Path to the data to predict should end with dataset")
    p.add_argument("-pred", 
        help="Path to the save location from the predictions")
    p.add_argument("-model", 
        help="Model abreviation", 
        nargs='?', const="sq3", 
        default="sq3")
    
    return p.parse_args()




def main():

    print("\n\n------------------------------")
    print("\n\nStarting Model Tester\n\n")

    args = parse_args() 
    data = args.data
    pred = args.pred
    model = args.model
    modelRunner = None

    print("data {}")
    print("pred {}")
    print("model {}")

    # Get the Model Runner
    if model == Models.CYL.value:
        modelRunner = CylRunner(config.BASE_MODEL_DIRECTORY)
    elif model == Models.SPV.value:
        modelRunner = SpvRunner(config.BASE_MODEL_DIRECTORY)
    elif model == Models.SAL.value:
        modelRunner = SalRunner(config.BASE_MODEL_DIRECTORY)
    elif model == Models.SQ3.value:
        modelRunner = Sq3Runner(config.BASE_MODEL_DIRECTORY)
    elif model == Models.POL.value:
        modelRunner = PolRunner(config.BASE_MODEL_DIRECTORY)
    elif model == Models.RAN.value:
        modelRunner = RanRunner(config.BASE_MODEL_DIRECTORY)
    else:
        raise ValueError("Model {} not supported!".format(model))

    # Build the docker image
    modelRunner.buildDockerImage()

    # Run Prediction
    modelRunner.run(data, pred)

    print("\n\n\nDone")
        

if __name__ == '__main__':
    main()







