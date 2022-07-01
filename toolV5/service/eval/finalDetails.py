"""
finalDetails
Class that amalgamates the details 
Collecting analytics such as top changes and averages 

@Author Garrett Christian
@Date 6/28/22
"""

import sys
import time


models = ["cyl", "spv", "sal"]


# --------------------------------------------------------------------------
# FINAL DETAILS


"""
Class to represent the final details


finalData {}
    mutation {}
        accuracy {}
            all {}
                total - int
                cyl - int
                spv - int
                sal - int
                min - float
                max - float
                avg - float
                min_cyl - float
                max_cyl - float
                avg_cyl - float
                min_spv - float
                max_spv - float
                avg_spv - float
                min_sal - float
                max_sal - float
                avg_sal - float
            bucket_*bucketNum {}
                ... same as all {} plus
                model_overlap {}
                    cyl - int
                    spv - int
                    sal - int
                    cyl_spv - int
                    cyl_sal - int
                    spv_sal - int          
                    cyl_spv_sal - int
        jaccard {}
            ... same as accuracy
    *model {}
        mutation {}
            top_acc [size of top num]
                (detail _id, - str 
                percent change - float)
            top_jac [size of top num]
                (detail _id, - str 
                percent change - float)
    count - int
    count_attempted - int
    seconds - str
    time - float
"""
class FinalDetails:
    def __init__(self, batchId, topNum=10, doneVelDir="", doneLabelDir=""):
        self.finalData = {}
        self.batchId = batchId
        self.mutationsEnabled = set()
        self.topNum = topNum
        self.buckets = list(range(0, 7))
        self.prepFinalDetails()

        # If removing non top bins/labels
        self.doneVelDir = doneVelDir
        self.doneLabelDir = doneLabelDir
    

    """
    Defines boundaries for buckets
    """
    def percentLossToBucket(self, percentLoss):

        bucket = 0 # percentLoss >= 0.1 %
        if (percentLoss < -5):
            bucket = 6
        elif (percentLoss < -4):
            bucket = 5
        elif (percentLoss < -3):
            bucket = 4
        elif (percentLoss < -2):
            bucket = 3
        elif (percentLoss < -1):
            bucket = 2
        elif (percentLoss < -0.1):
            bucket = 1

        return bucket



    """
    Prepares the final details dictionary preloading some of the keys
    """
    def prepFinalDetails(self):
        self.finalData = {}

        self.finalData["_id"] = self.batchId
        self.finalData["time"] = int(time.time())
        self.finalData["dateTime"] = time.ctime(time.time())

        self.finalData["mutations"] = []

        self.finalData["buckets"] = []
        for bucketNum in self.buckets:
            self.finalData["buckets"].append("bucket_" + str(bucketNum))

        # Top to save
        for model in models:
            self.finalData[model] = {}

        self.finalData["count"] = 0
        self.finalData["count_attempted"] = 0


    """
    Adds a new mutation to the final details dictionary preloading the keys
    """
    def addMutation(self, mutation):

        self.mutationsEnabled.add(mutation)
        self.finalData["mutations"].append(mutation)

        # Top to save
        for model in models:
            self.finalData[model][mutation] = {}
            self.finalData[model][mutation]["top_acc"] = []
            self.finalData[model][mutation]["top_jac"] = []

        # Accuracy and Jaccard metrics
        self.finalData[mutation] = {}
        self.finalData[mutation]["accuracy"] = {}
        self.finalData[mutation]["jaccard"] = {}
        for bucketNum in self.buckets:
            bucketKey = "bucket_" + str(bucketNum)
            # Accuracy
            self.finalData[mutation]["accuracy"][bucketKey] = {}
            self.finalData[mutation]["accuracy"][bucketKey]["total"] = 0
            for model in models:
                self.finalData[mutation]["accuracy"][bucketKey]["total_" + model] = 0
                self.finalData[mutation]["accuracy"][bucketKey]["min_" + model] = sys.maxsize
                self.finalData[mutation]["accuracy"][bucketKey]["max_" + model] = sys.maxsize * -1
                self.finalData[mutation]["accuracy"][bucketKey]["avg_" + model] = 0
            self.finalData[mutation]["accuracy"][bucketKey]["min"] = sys.maxsize
            self.finalData[mutation]["accuracy"][bucketKey]["max"] = sys.maxsize * -1
            self.finalData[mutation]["accuracy"][bucketKey]["avg"] = 0
            self.finalData[mutation]["accuracy"][bucketKey]["model_overlap"] = {}

            # Jaccard Accuracy
            self.finalData[mutation]["jaccard"][bucketKey] = {}
            self.finalData[mutation]["jaccard"][bucketKey]["total"] = 0
            for model in models:
                self.finalData[mutation]["jaccard"][bucketKey]["total_" + model] = 0
                self.finalData[mutation]["jaccard"][bucketKey]["min_" + model] = sys.maxsize
                self.finalData[mutation]["jaccard"][bucketKey]["max_" + model] = sys.maxsize * -1
                self.finalData[mutation]["jaccard"][bucketKey]["avg_" + model] = 0
            self.finalData[mutation]["jaccard"][bucketKey]["min"] = sys.maxsize
            self.finalData[mutation]["jaccard"][bucketKey]["max"] = sys.maxsize * -1
            self.finalData[mutation]["jaccard"][bucketKey]["avg"] = 0
            self.finalData[mutation]["jaccard"][bucketKey]["model_overlap"] = {}
        
        # Accuracy
        self.finalData[mutation]["accuracy"]["all"] = {}
        self.finalData[mutation]["accuracy"]["all"]["total"] = 0
        self.finalData[mutation]["accuracy"]["all"]["min"] = sys.maxsize
        self.finalData[mutation]["accuracy"]["all"]["max"] = sys.maxsize * -1
        self.finalData[mutation]["accuracy"]["all"]["avg"] = 0
        for model in models:
            self.finalData[mutation]["accuracy"]["all"][model] = 0
            self.finalData[mutation]["accuracy"]["all"]["min_" + model] = sys.maxsize
            self.finalData[mutation]["accuracy"]["all"]["max_" + model] = sys.maxsize * -1
            self.finalData[mutation]["accuracy"]["all"]["avg_" + model] = 0

        # Jaccard Accuracy
        self.finalData[mutation]["jaccard"]["all"] = {}
        self.finalData[mutation]["jaccard"]["all"]["total"] = 0
        self.finalData[mutation]["jaccard"]["all"]["min"] = sys.maxsize
        self.finalData[mutation]["jaccard"]["all"]["max"] = sys.maxsize * -1
        self.finalData[mutation]["jaccard"]["all"]["avg"] = 0
        for model in models:
            self.finalData[mutation]["jaccard"]["all"][model] = 0
            self.finalData[mutation]["jaccard"]["all"]["min_" + model] = sys.maxsize
            self.finalData[mutation]["jaccard"]["all"]["max_" + model] = sys.maxsize * -1
            self.finalData[mutation]["jaccard"]["all"]["avg_" + model] = 0



    """
    Updates the final details dictionary after a batch
    This removes the bins and labels that do not meet the save criteria (top five accuracy loss & top five jaccard loss)

    @param details list of detail dictionarys that enumerates what occured in this transformation
    @param finalData ditctionary that describes what should be saved and how many of each mutation occured
    @return finalData dictionary updated with new mutations that occured
    """
    def updateFinalDetails(self, details):

        print("Updating final details")

        potentialRemove = set()
        deleteFiles = []
        
        for detail in details:
            # Add count for mutation
            

            failKeyA = {}
            failKeyJ = {}
            for bucketNum in self.buckets:
                failKeyA["bucket_" + str(bucketNum)] = ""
                failKeyJ["bucket_" + str(bucketNum)] = ""

            mutation = detail["mutation"]

            # Validate that the final data has this mutation
            if mutation not in self.mutationsEnabled:
                self.addMutation(mutation)

            # Check if we have a lower accuracy change for this mutation
            for model in models:

                # Save top 5 Accuracy
                percentLossAcc = detail[model]["percentLossAcc"]

                # don't have top_acc yet, add it 
                if (len(self.finalData[model][mutation]["top_acc"]) < self.topNum):
                    self.finalData[model][mutation]["top_acc"].append((detail["_id"], percentLossAcc))
                    self.finalData[model][mutation]["top_acc"].sort(key = lambda x: x[1])

                # Do have top_acc check against current highest
                else:
                    idRemove = detail["_id"]
                        
                    # new lower change to acc
                    if (self.finalData[model][mutation]["top_acc"][4][1] > percentLossAcc):
                        self.finalData[model][mutation]["top_acc"].append((detail["_id"], percentLossAcc))
                        self.finalData[model][mutation]["top_acc"].sort(key = lambda x: x[1])
                        idRemove = self.finalData[model][mutation]["top_acc"].pop()[0]
                
                    potentialRemove.add(idRemove)


                # Top Jaccard Change
                percentLossJac = detail[model]["percentLossJac"]

                # don't have top_jacc yet, add it 
                if (len(self.finalData[model][mutation]["top_jac"]) < self.topNum):
                    self.finalData[model][mutation]["top_jac"].append((detail["_id"], percentLossJac))
                    self.finalData[model][mutation]["top_jac"].sort(key = lambda x: x[1])

                # Do have top_jacc check against current highest
                else:
                    idRemove = detail["_id"]
                    
                    # new lower change to jacc
                    if (self.finalData[model][mutation]["top_jac"][4][1] > percentLossJac):
                        self.finalData[model][mutation]["top_jac"].append((detail["_id"], percentLossJac))
                        self.finalData[model][mutation]["top_jac"].sort(key = lambda x: x[1])
                        idRemove = self.finalData[model][mutation]["top_jac"].pop()[0]
                
                    potentialRemove.add(idRemove)


                # Accuracy 

                # Update accuracy metrics for all
                self.finalData[mutation]["accuracy"]["all"]["min"] = min(percentLossAcc, self.finalData[mutation]["accuracy"]["all"]["min"])
                self.finalData[mutation]["accuracy"]["all"]["max"] = max(percentLossAcc, self.finalData[mutation]["accuracy"]["all"]["max"])
                self.finalData[mutation]["accuracy"]["all"]["avg"] = percentLossAcc + self.finalData[mutation]["accuracy"]["all"]["avg"]
                self.finalData[mutation]["accuracy"]["all"]["min_" + model] = min(percentLossAcc, self.finalData[mutation]["accuracy"]["all"]["min_" + model])
                self.finalData[mutation]["accuracy"]["all"]["max_" + model] = max(percentLossAcc, self.finalData[mutation]["accuracy"]["all"]["max_" + model])
                self.finalData[mutation]["accuracy"]["all"]["avg_" + model] = percentLossAcc + self.finalData[mutation]["accuracy"]["all"]["avg_" + model]

                # Update accuracy metrics for model
                bucketNum = self.percentLossToBucket(percentLossAcc)
                bucketKey = "bucket_" + str(bucketNum)
                self.finalData[mutation]["accuracy"][bucketKey]["total"] = 1 + self.finalData[mutation]["accuracy"][bucketKey]["total"]
                self.finalData[mutation]["accuracy"][bucketKey]["total_" + model] = 1 + self.finalData[mutation]["accuracy"][bucketKey]["total_" + model]
                self.finalData[mutation]["accuracy"][bucketKey]["min"] = min(percentLossAcc, self.finalData[mutation]["accuracy"][bucketKey]["min"])
                self.finalData[mutation]["accuracy"][bucketKey]["max"] = max(percentLossAcc, self.finalData[mutation]["accuracy"][bucketKey]["max"])
                self.finalData[mutation]["accuracy"][bucketKey]["avg"] = percentLossAcc + self.finalData[mutation]["accuracy"][bucketKey]["avg"]
                self.finalData[mutation]["accuracy"][bucketKey]["min_" + model] = min(percentLossAcc, self.finalData[mutation]["accuracy"][bucketKey]["min_" + model])
                self.finalData[mutation]["accuracy"][bucketKey]["max_" + model] = max(percentLossAcc, self.finalData[mutation]["accuracy"][bucketKey]["max_" + model])
                self.finalData[mutation]["accuracy"][bucketKey]["avg_" + model] = percentLossAcc + self.finalData[mutation]["accuracy"][bucketKey]["avg_" + model]

                if (failKeyA[bucketKey] == ""):
                    failKeyA[bucketKey] = model
                else:
                    failKeyA[bucketKey] = failKeyA[bucketKey] + "_" + model

                # Jaccard 

                # Update jaccard metrics for all
                self.finalData[mutation]["jaccard"]["all"]["min"] = min(percentLossJac, self.finalData[mutation]["jaccard"]["all"]["min"])
                self.finalData[mutation]["jaccard"]["all"]["max"] = max(percentLossJac, self.finalData[mutation]["jaccard"]["all"]["max"])
                self.finalData[mutation]["jaccard"]["all"]["avg"] = percentLossJac + self.finalData[mutation]["jaccard"]["all"]["avg"]
                self.finalData[mutation]["jaccard"]["all"]["min_" + model] = min(percentLossJac, self.finalData[mutation]["jaccard"]["all"]["min_" + model])
                self.finalData[mutation]["jaccard"]["all"]["max_" + model] = max(percentLossJac, self.finalData[mutation]["jaccard"]["all"]["max_" + model])
                self.finalData[mutation]["jaccard"]["all"]["avg_" + model] = percentLossJac + self.finalData[mutation]["jaccard"]["all"]["avg_" + model]

                # Update jaccard metrics for model
                bucketNum = self.percentLossToBucket(percentLossJac)
                bucketKey = "bucket_" + str(bucketNum)
                self.finalData[mutation]["jaccard"][bucketKey]["total"] = 1 + self.finalData[mutation]["jaccard"][bucketKey]["total"]
                self.finalData[mutation]["jaccard"][bucketKey]["total_" + model] = 1 + self.finalData[mutation]["jaccard"][bucketKey]["total_" + model]
                self.finalData[mutation]["jaccard"][bucketKey]["min"] = min(percentLossJac, self.finalData[mutation]["jaccard"][bucketKey]["min"])
                self.finalData[mutation]["jaccard"][bucketKey]["max"] = max(percentLossJac, self.finalData[mutation]["jaccard"][bucketKey]["max"])
                self.finalData[mutation]["jaccard"][bucketKey]["avg"] = percentLossJac + self.finalData[mutation]["jaccard"][bucketKey]["avg"]
                self.finalData[mutation]["jaccard"][bucketKey]["min_" + model] = min(percentLossJac, self.finalData[mutation]["jaccard"][bucketKey]["min_" + model])
                self.finalData[mutation]["jaccard"][bucketKey]["max_" + model] = max(percentLossJac, self.finalData[mutation]["jaccard"][bucketKey]["max_" + model])
                self.finalData[mutation]["jaccard"][bucketKey]["avg_" + model] = percentLossJac + self.finalData[mutation]["jaccard"][bucketKey]["avg_" + model]
                
                if (failKeyJ[bucketKey] == ""):
                    failKeyJ[bucketKey] = model
                else:
                    failKeyJ[bucketKey] = failKeyJ[bucketKey] + "_" + model

            # Total count
            self.finalData[mutation]["accuracy"]["all"]["total"] = 1 + self.finalData[mutation]["accuracy"]["all"]["total"]
            self.finalData[mutation]["jaccard"]["all"]["total"] = 1 + self.finalData[mutation]["jaccard"]["all"]["total"]

            # What model landed in what bucket
            for bucketNum in self.buckets:
                bucketKey = "bucket_" + str(bucketNum)
                # Accuracy 
                if (failKeyA[bucketKey] != ""):
                    key = failKeyA[bucketKey]
                    curCount = self.finalData[mutation]["accuracy"][bucketKey]["model_overlap"].get(key, 0)
                    self.finalData[mutation]["accuracy"][bucketKey]["model_overlap"][key] = curCount + 1
                # Jacc
                if (failKeyJ[bucketKey] != ""):
                    key = failKeyJ[bucketKey]
                    curCount = self.finalData[mutation]["jaccard"][bucketKey]["model_overlap"].get(key, 0)
                    self.finalData[mutation]["jaccard"][bucketKey]["model_overlap"][key] = curCount + 1


        # Remove bin / labels that are not within the top 5
        idInUse = set()
        for mutation in self.mutationsEnabled:
            mutation = str(mutation).replace("Mutation.", "")
            for model in models:
                for detailRecord in self.finalData[model][mutation]["top_acc"]:
                    idInUse.add(detailRecord[0])
                for detailRecord in self.finalData[model][mutation]["top_jac"]:
                    idInUse.add(detailRecord[0])

        for idRemove in potentialRemove:
            if idRemove not in idInUse:
                labelRemove = self.doneLabelDir + "/actual/" + idRemove + ".label"
                binRemove = self.doneVelDir + "/" + idRemove + ".bin"
                deleteFiles.append(binRemove)
                deleteFiles.append(labelRemove)
                for model in models:
                    modelLabelRemove = self.doneLabelDir + "/" + model + "/" + idRemove + ".label"
                    deleteFiles.append(modelLabelRemove)

        return deleteFiles




    """
    Collects the averages for each mutation and bucket 
    """
    def finalizeFinalDetails(self):

        for mutation in self.mutationsEnabled:

            # Model Avgs
            for model in models:
                # All model Avgs
                allCount = self.finalData[mutation]["accuracy"]["all"]["total"]
                if (allCount > 0):
                    self.finalData[mutation]["accuracy"]["all"]["avg_" + model] = self.finalData[mutation]["accuracy"]["all"]["avg_" + model] / allCount

                # Bucket model Avgs
                for bucketNum in self.buckets:
                    bucketKey = "bucket_" + str(bucketNum)
                    bucketCountModel = self.finalData[mutation]["accuracy"][bucketKey]["total_" + model]
                    if (bucketCountModel > 0):
                        self.finalData[mutation]["accuracy"][bucketKey]["avg_" + model] = self.finalData[mutation]["accuracy"][bucketKey]["avg_" + model] / bucketCountModel


                # All model Jaccard
                allCount = self.finalData[mutation]["jaccard"]["all"]["total"]
                if (allCount > 0):
                    self.finalData[mutation]["jaccard"]["all"]["avg_" + model] = self.finalData[mutation]["jaccard"]["all"]["avg_" + model] / allCount

                # Bucket model Avgs
                for bucketNum in self.buckets:
                    bucketKey = "bucket_" + str(bucketNum)
                    bucketCountModel = self.finalData[mutation]["jaccard"][bucketKey]["total_" + model]
                    if (bucketCountModel > 0):
                        self.finalData[mutation]["jaccard"][bucketKey]["avg_" + model] = self.finalData[mutation]["jaccard"][bucketKey]["avg_" + model] / bucketCountModel



            # Accuracy

            # All Avgs
            allCount = self.finalData[mutation]["accuracy"]["all"]["total"]
            if (allCount > 0):
                self.finalData[mutation]["accuracy"]["all"]["avg"] = self.finalData[mutation]["accuracy"]["all"]["avg"] / allCount

            # Bucket Avgs
            for bucketNum in self.buckets:
                bucketKey = "bucket_" + str(bucketNum)
                bucketCountAll = self.finalData[mutation]["accuracy"][bucketKey]["total"]
                if (bucketCountAll > 0):
                    self.finalData[mutation]["accuracy"][bucketKey]["avg"] = self.finalData[mutation]["accuracy"][bucketKey]["avg"] / bucketCountAll
                    
            # Jaccard

            # All Avgs
            allCount = self.finalData[mutation]["jaccard"]["all"]["total"]
            if (allCount > 0):
                self.finalData[mutation]["jaccard"]["all"]["avg"] = self.finalData[mutation]["jaccard"]["all"]["avg"] / allCount

            # Bucket Avgs
            for bucketNum in self.buckets:
                bucketKey = "bucket_" + str(bucketNum)
                bucketCountAll = self.finalData[mutation]["jaccard"][bucketKey]["total"]
                if (bucketCountAll > 0):
                    self.finalData[mutation]["jaccard"][bucketKey]["avg"] = self.finalData[mutation]["jaccard"][bucketKey]["avg"] / bucketCountAll
                    
            
        return self.finalData


    """
    Sets the time for the final details
    """
    def setTime(self, timeSeconds, timeFormatted):
        self.finalData["seconds"] = timeSeconds
        self.finalData["time"] = timeFormatted


    """
    Sets the attempts for the final details
    """
    def setAttempts(self, successCount, attemptCount):
        self.finalData["count"] = successCount
        self.finalData["count_attempted"] = attemptCount
        self.finalData["percent_success"] = (successCount / attemptCount) * 100








