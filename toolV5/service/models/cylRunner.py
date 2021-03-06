"""
cylRunner 
Runner for the cylinder3d model
cyl
Cylinder3D
[1 / 3]

@Author Garrett Christian
@Date 7/22/22
"""

import os

import domain.modelConstants as modelConstants
from dockerRunner import DockerRunner

# --------------------------------------------------------------------------

class CylRunner(DockerRunner):
    def __init__(self,  modelBaseDir):
        super(DockerRunner, self).__init__(modelBaseDir, modelConstants.CYL_DIRECTORY_NAME)


    """
    Runs the cylinder3d docker image

    """
    def run(self, dataDirectory, predictionDirectory):
        # Normalize paths
        dataDir = os.path.normpath(dataDirectory)
        predictionDir = os.path.normpath(predictionDirectory)

        if (os.path.basename(dataDir) != "dataset"):
            raise ValueError("Expecting that the directory to predict ends with dataset {}".format(dataDir))

        # Command to run the model with
        runCommand = "python3 demo_folder.py"
        runCommand += " --demo-folder {}/sequences/00/velodyne".format(dataDir)
        runCommand += " --save-folder {}".format(predictionDir)

        # Location that command needs to be run from
        modelRunDir = self.modelDir

        self.runModelDocker(dataDir, predictionDir, modelRunDir, runCommand)

    


