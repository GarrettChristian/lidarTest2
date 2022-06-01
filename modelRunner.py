
import argparse
import os

# ------------------

# TODO Hard coded paths fix

pathToModels = "/home/garrett/Documents"

modelCylinder3D = "Cylinder3D"

modelSpvnas = "spvnas"

modelSalsaNext = "SalsaNext"

modelCyl = "cyl"
modelSpv = "spv"
modelSal = "sal"

resultsDir = "/home/garrett/Documents/data/results/"
dataRoot = "/home/garrett/Documents/data/tmp/dataset"

# ------------------

def runCyl():
    print("running {}".format(modelCylinder3D))

    runCommand = "python demo_folder.py "
    runCommand += "--demo-folder " + dataRoot + "/sequences/00/velodyne " 
    runCommand += "--save-folder " + resultsDir + modelCyl

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelCylinder3D)

    # Run Model
    os.system(runCommand)
    


def runSpv():
    print("running {}".format(modelSpvnas))

    runCommand = "torchpack dist-run " 
    runCommand += "-np 1 python evaluate.py configs/semantic_kitti/default.yaml "
    runCommand += "--name SemanticKITTI_val_SPVNAS@65GMACs "
    runCommand += "--data-dir " + dataRoot + "/sequences "
    runCommand += "--save-dir " + resultsDir + modelSpv

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSpvnas)

    # Run Model
    os.system(runCommand)


def runSal():
    print("running {}".format(modelSalsaNext))

    runCommand = "python infer.py " 
    # Data to run on
    runCommand += "-d " + dataRoot
    # Results
    runCommand += " -l " + resultsDir + modelSal
    # model
    runCommand += " -m /home/garrett/Documents/SalsaNext/pretrained " 
    runCommand += "-s test -c 1"
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSalsaNext + "/train/tasks/semantic")

    # Run Model
    os.system(runCommand)


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')
    # p.add_argument(
    #     'binLocation', help='Path to the dir to test')
    p.add_argument(
        "-model", help="model to test: {}, {}, {}, all (default)".format(modelCyl, modelSpv, modelSal), 
        nargs='?', const="all", default="all")

    return p.parse_args()


def main():

    print("\n\n------------------------------")
    print("\n\nStarting Model Runner\n\n")

    args = parse_args()

    modelToRun = [args.model]

    if (args.model == "all"):
        modelToRun = [modelCyl, modelSpv, modelSal]
    
    for modelName in modelToRun:
        
        if (modelName == modelCyl):
            runCyl()
        elif (modelName == modelSpv):
            runSpv()
        elif (modelName == modelSal):
            runSal()
        else:
            print("Unsuported model {}".format(modelName))
        
        print("\n\n------------------------------\n\n")

if __name__ == '__main__':
    main()



