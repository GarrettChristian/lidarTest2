"""
toolSessionManager 
Manages a given run of the semantic LiDAR fuzzer tool

@Author Garrett Christian
@Date 6/28/22
"""

import numpy as np
import glob, os
import shutil
import shortuuid


import domain.mutationsEnum as mutationsEnum


# --------------------------------------------------------------------------

class SessionManager:
    def __init__(self, args, recreation=False):

        # Run batch id
        self.batchId = str(shortuuid.uuid())
        self.models = ["cyl", "spv", "sal"]

        self.mongoConnect = args.mdb

        self.binPath = args.binPath
        self.labelPath = args.labelPath

        self.assetId = None
        self.scene = None
        self.sequence = range(0, 11)
        if (not recreation):
      
            # optional to set the bins and labels to be one specific scene
            self.scene = args.scene
            self.sequence = args.seq

        # Get the bins and labels
        self.binFiles = []
        self.labelFiles = []
        self.binFiles, self.labelFiles = self.getBinsLabels()



        # Configurable parameters for the tool
        self.scaleLimit = args.scaleLimit # (10,000 points)

        # Flag to use open3d to visualize the mutation
        self.visualize = args.vis
        self.verbose = False

        # Specific mutation arguments
        self.rotation = None
        self.mirrorAxis = None
        self.intensityChange = None
        self.scaleAmount = None
        self.signChange = None
        self.deformPercent = None
        self.deformPoint = None
        self.deformMu = None
        self.deformSigma = None
        self.deformSeed = None

        if (not recreation):
            # Mutations to choose from
            print("Selecting mutations to use")
            self.mutationsEnabled = self.prepareMutations(args.m)

            # Batch and total counts
            self.batchNum = int(args.b)
            self.expectedNum = int(args.count)
            print("Will run until {} successful mutations are obtained".format({self.expectedNum}))
            print("Batch evaluating and saving every {}".format({self.batchNum}))

            # Specific mutation arguments
            self.rotation = args.rotate
            self.mirrorAxis = args.mirror
            self.intensityChange = args.intensity
            self.scaleAmount = args.scale
            self.signChange = args.sign
            self.deformPercent = args.deformPercent
            self.deformPoint = args.deformPoint
            self.deformMu = args.deformMu
            self.deformSigma = args.deformSigma
            self.deformSeed = args.deformSeed

            # Specific asset
            self.assetId = args.assetId

            # Configurable parameters for the tool
            self.threads = args.t 
            self.verbose = args.verbose
            self.asyncEval = args.asyncEval

            # Flags for convience 
            self.saveMutationFlag = args.ns
            self.evalMutationFlag = args.ne and self.saveMutationFlag
            if not self.saveMutationFlag:
                print("Saving disabled")
            if not self.evalMutationFlag:
                print("Evaluation disabled")
            

            # Paths to the directories where specific things are stored
            self.saveAt = args.saveAt
            self.stageDir = ""
            self.dataRoot = ""
            self.resultDir = ""
            self.currentVelDir = ""
            self.doneVelDir = ""
            self.resultCylDir = ""
            self.resultSpvDir = ""
            self.resultSalDir = ""
            self.evalDir = ""
            self.doneLabelActualDir = ""
            self.dataDir = ""
            self.doneLabelDir = ""

            if (self.saveMutationFlag):
                print("Setting up result folder pipeline")
                self.setUpDataFolders(self.threads)



    """
    Selects the mutations that will be in use
    """
    def prepareMutations(self, mutationsGiven):
        # Get mutations to use
        mutations = []
        if (mutationsGiven != None):
            for mutation in mutationsGiven.split(","):
                try:
                    mutantToAdd = mutationsEnum.Mutation[mutation]
                except KeyError:
                    print("%s is not a valid option" % (mutation))
                    exit()
                mutations.append(mutantToAdd)
        else:
            mutations = [mutationsEnum.Mutation.ADD_ROTATE]

        print("Mutations: {}".format(mutations))
        return mutations

        
    """
    Setup step to get all potential bin files
    """
    def getBinsLabels(self):
        print("Parsing bins and label folders:")
        print("bins {}".format(self.binPath))
        print("labels {}".format(self.labelPath))
        print("Collecting Labels and Bins for sequence {}".format(self.sequence))

        binFilesRun = []
        labelFilesRun = []

        # Specific Scene
        if (self.scene):
            folderNum = str(self.sequence).rjust(2, '0')
            currBinPath = self.binPath + folderNum
            currLabelPath = self.labelPath + folderNum

            binFilesRun = [currLabelPath + "/velodyne/" + self.scene + ".bin"]
            labelFilesRun = [currBinPath + "/labels/" + self.scene + ".label"]

        # Any scene
        else:
            for sequenceNum in self.sequence:
            
                folderNum = str(sequenceNum).rjust(2, '0')
                currBinPath = self.binPath + folderNum
                currLabelPath = self.labelPath + folderNum

                binFilesSequence = np.array(glob.glob(currBinPath + "/velodyne/*.bin", recursive = True))
                labelFilesSequence = np.array(glob.glob(currLabelPath + "/labels/*.label", recursive = True))
                print("Parsing Scene {}".format(folderNum))
                
                # Sort
                labelFilesSequence = sorted(labelFilesSequence)
                binFilesSequence = sorted(binFilesSequence)
                
                for labelFile in labelFilesSequence:
                    labelFilesRun.append(labelFile)
                
                for binFile in binFilesSequence:
                    binFilesRun.append(binFile)

        return binFilesRun, labelFilesRun



    def setUpDataFolders(self, threads):

        """
        /output
            /staging
                /velodyne
                /labels
            /current/dataset/sequences/00
                /velodyne
            /results
                /cyl
                /spv
                /sal/sequences/00
            # /eval
            #     /labels
            #         /cyl
            #         /spv
            #         /sal
            /done
                /velodyne
                /labels
                    /actual
                    /cyl
                    /spv
                    /sal
        """

        # make a top level data dir
        self.dataDir = self.saveAt + "/output"
        isExist = os.path.exists(self.dataDir)
        if not isExist:
            os.makedirs(self.dataDir)

        """
        /data
            /staging
                    /velodyne0
                    /labels0
        """

        # staging
        self.stageDir = self.dataDir + "/staging"
        isExist = os.path.exists(self.stageDir)
        if not isExist:
            os.makedirs(self.stageDir)

        # for thread in range(0, threads):
        # staging vel
        self.stageVel = self.stageDir + "/velodyne"
        if os.path.exists(self.stageVel):
            shutil.rmtree(self.stageVel)
            print("Removing {}".format(self.stageVel))
        os.makedirs(self.stageVel)

        # staging label
        self.stagelabel = self.stageDir + "/labels"
        if os.path.exists(self.stagelabel):
            shutil.rmtree(self.stagelabel)
            print("Removing {}".format(self.stagelabel))
        os.makedirs(self.stagelabel)

        """
        /data
            /current/dataset/sequences/00
                /velodyne
        """

        # current
        self.dataRoot = self.dataDir + "/current/dataset"
        currentDir = self.dataRoot + "/sequences/00"
        os.makedirs(currentDir, exist_ok=True)
        self.currentVelDir = currentDir + "/velodyne"
        if os.path.exists(self.currentVelDir):
            shutil.rmtree(self.currentVelDir)
            print("Removing {}".format(self.currentVelDir))
        os.makedirs(self.currentVelDir)

        """
        /data
            /results
                /cyl
                /spv
                /sal/sequences/00
        """

        # results
        self.resultDir = self.dataDir + "/results"
        isExist = os.path.exists(self.resultDir)
        if not isExist:
            os.makedirs(self.resultDir)
        
        # Cyl
        self.resultCylDir = self.resultDir + "/cyl"
        if os.path.exists(self.resultCylDir):
            shutil.rmtree(self.resultCylDir)
            print("Removing {}".format(self.resultCylDir))
        os.makedirs(self.resultCylDir)

        # Spv
        self.resultSpvDir = self.resultDir + "/spv"
        if os.path.exists(self.resultSpvDir):
            shutil.rmtree(self.resultSpvDir)
            print("Removing {}".format(self.resultSpvDir))
        os.makedirs(self.resultSpvDir)

        # Sal
        self.resultSalDir = self.resultDir + "/sal/sequences/00"
        if os.path.exists(self.resultDir + "/sal"):
            shutil.rmtree(self.resultDir + "/sal", ignore_errors=True)
            print("Removing {}".format(self.resultSalDir))
        os.makedirs(self.resultSalDir, exist_ok=True)


        # """
        # /data
        #     /eval
        #         /labels0
        #             /cyl
        #             /spv
        #             /sal
        # """

        # # eval
        # self.evalDir = self.dataDir + "/eval"
        # if os.path.exists(self.evalDir):
        #     shutil.rmtree(self.evalDir, ignore_errors=True)
        #     print("Removing {}".format(self.evalDir))
        # os.makedirs(self.evalDir, exist_ok=True)

        # # for thread in range(0, threads):
        # # staging vel
        # labelThreadDir = self.evalDir + "/label"
        # os.makedirs(labelThreadDir)

        # for model in self.models:
        #     labelThreadModelDir = labelThreadDir + "/" + model
        #     os.makedirs(labelThreadModelDir)

        """
        /data
            /done
                /velodyne
                /labels
                    /actual
                    /cyl
                    /spv
                    /sal
        """

        # done
        self.doneDir = self.dataDir + "/done"
        if os.path.exists(self.doneDir):
            shutil.rmtree(self.doneDir, ignore_errors=True)
            print("Removing {}".format(self.doneDir))
        # isExist = os.path.exists(doneDir)
        # if not isExist:
        os.makedirs(self.doneDir)
        
        # done
        self.doneVelDir = self.doneDir + "/velodyne"
        isExist = os.path.exists(self.doneVelDir)
        if not isExist:
            os.makedirs(self.doneVelDir)

        # labels
        self.doneLabelDir = self.doneDir + "/labels"
        isExist = os.path.exists(self.doneLabelDir)
        if not isExist:
            os.makedirs(self.doneLabelDir)

        # labels done
        self.doneLabelActualDir = self.doneLabelDir + "/actual"
        isExist = os.path.exists(self.doneLabelActualDir)
        if not isExist:
            os.makedirs(self.doneLabelActualDir)
        
        for model in self.models:
            # Models done
            doneLabelModelDir = self.doneLabelDir + "/" + model
            isExist = os.path.exists(doneLabelModelDir)
            if not isExist:
                os.makedirs(doneLabelModelDir)




