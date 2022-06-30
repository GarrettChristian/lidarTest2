"""
BaseAccuracyRepository 
Handles all database interaction for base accuracy

@Author Garrett Christian
@Date 6/23/22
"""

from pymongo import MongoClient

import data.mongoRepository as mongoRepository

# --------------------------------------------------------------------------

class BaseAccuracyRepository(mongoRepository.MongoRepository):
    def __init__(self, mongoConnectPath):
        super(BaseAccuracyRepository, self).__init__(mongoConnectPath)
        self.baseAccuracyCollection = self.db["base_accuracy"]
        

    """
    Gets the base accuracy obtained on a given scene
    """
    def getBaseAccuracy(self, sequence, scene):

        baseAcc = self.baseAccuracyCollection.find_one({"sequence": sequence, "scene": scene})

        return baseAcc



