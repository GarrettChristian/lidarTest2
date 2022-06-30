"""
DetailsRepository 
Handles all database interaction for mutation details

@Author Garrett Christian
@Date 6/23/22
"""

from pymongo import MongoClient

import data.mongoRepository as mongoRepository

# --------------------------------------------------------------------------

class DetailsRepository(mongoRepository.MongoRepository):
    def __init__(self, mongoConnectPath):
        super(DetailsRepository, self).__init__(mongoConnectPath)
        self.mutationCollection = self.db["mutations"]
        

    """
    Get mutation details by id
    """
    def getMutationDetailsById(self, id):
        details = self.mutationCollection.find_one({"_id": id})

        return details

    """
    Save mutation data (Bulk)
    """
    def saveMutationDetails(self, mutationDetails):
        print("Saving {} Mutation Details".format(len(mutationDetails)))
        self.mutationCollection.insert_many(mutationDetails)






