"""
modelRunner

Helper to run the models
"""

import os
import subprocess


# --------------------------------------------------------------------------
# Constants


pathToModels = "/home/garrett/Documents"

modelCylinder3D = "Cylinder3D"
modelSpvnas = "spvnas"
modelSalsaNext = "SalsaNext"
modelSqueezeSegV3 = "SqueezeSegV3"
modelPolarSeg = "PolarSeg"
modelRangeNet = "lidar-bonnetal"

modelCyl = "cyl"
modelSpv = "spv"
modelSal = "sal"
modelSq3 = "sq3"
modelPol = "pol"
modelDar = "dar"



# --------------------------------------------------------------------------
# Runners


"""
Runner for the Cylinder3D model
"""
def runCyl(sessionManager):
    print("running {}".format(modelCylinder3D))

    runCommand = "python demo_folder.py "
    runCommand += "--demo-folder " + sessionManager.dataRoot + "/sequences/00/velodyne " 
    runCommand += "--save-folder " + sessionManager.resultCylDir

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelCylinder3D)

    # runCommand += "2> /dev/null"
    print(pathToModels + "/" + modelCylinder3D)

    # Run Model
    # os.system(runCommand)
    subprocess.Popen(runCommand, shell=True).wait()


"""
Runner for the SPVNAS model
"""
def runSpv(sessionManager):
    print("running {}".format(modelSpvnas))

    runCommand = "torchpack dist-run " 
    runCommand += "-np 1 python evaluate.py configs/semantic_kitti/default.yaml "
    runCommand += "--name SemanticKITTI_val_SPVNAS@65GMACs "
    runCommand += "--data-dir " + sessionManager.dataRoot + "/sequences "
    runCommand += "--save-dir " + sessionManager.resultSpvDir

    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSpvnas)

    # Run Model
    # os.system(runCommand)
    subprocess.Popen(runCommand, shell=True).wait()


"""
Runner for the SalsaNext model
"""
def runSal(sessionManager):
    print("running {}".format(modelSalsaNext))

    runCommand = "python infer.py" 
    # Data to run on
    runCommand += " -d " + sessionManager.dataRoot
    # Results
    runCommand += " -l " + sessionManager.resultDir + "/" + modelSal
    # model
    runCommand += " -m /home/garrett/Documents/SalsaNext/pretrained"
    runCommand += " -s test -c 1"
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSalsaNext + "/train/tasks/semantic")

    # Run Model
    # os.system(runCommand)
    subprocess.Popen(runCommand, shell=True).wait()


"""
Runner for the SqueezeSeqV3 model
"""
def runSq3(dataToPredict, saveAt):
    print("running {}".format(modelSq3))

    runCommand = "python demo.py" 
    # Data to run on
    runCommand += " --dataset " + dataToPredict
    # Results
    runCommand += " --log " + saveAt
    # model
    runCommand += " --model /home/garrett/Documents/SqueezeSegV3/SSGV3-53 "
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelSqueezeSegV3 + "/src/tasks/semantic")

    # Run Model
    subprocess.Popen(runCommand, shell=True).wait()


"""
Runner for the PolarNet model
"""
def runPol(dataToPredict, saveAt):
    print("running {}".format(modelPolarSeg))

    runCommand = "python test_pretrain_SemanticKITTI.py" 
    # Data to run on
    runCommand += " --data_dir " + dataToPredict
    # Results
    runCommand += " --test_output_path " + saveAt
    # model
    runCommand += " --model_save_path pretrained_weight/SemKITTI_PolarSeg.pt"
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelPolarSeg)

    # runCommand += "2> /dev/null"

    # Run Model
    # os.system(runCommand)
    subprocess.Popen(runCommand, shell=True).wait()



"""
Runner for the rangeNet++ model
"""
def runDar(dataToPredict, saveAt):
    print("running {}".format(modelRangeNet))

    runCommand = "python infer.py" 
    # Data to run on
    runCommand += " --dataset " + dataToPredict
    # Results
    runCommand += " --log " + saveAt
    # model
    runCommand += " --model " + pathToModels + "/" + modelRangeNet + "/darknet53"
    
    # Change To Model Dir
    os.chdir(pathToModels + "/" + modelRangeNet + "/train/tasks/semantic")

    # runCommand += "2> /dev/null"

    # Run Model
    # os.system(runCommand)
    subprocess.Popen(runCommand, shell=True).wait()




