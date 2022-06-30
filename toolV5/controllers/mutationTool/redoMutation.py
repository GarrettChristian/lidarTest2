"""
semFuzzLidar 
Main runner for the mutation tool

@Author Garrett Christian
@Date 6/23/22
"""


from email.policy import default
import numpy as np
import open3d as o3d
import random
import argparse
import shortuuid
import os
import time
import json

import domain.toolSessionManager as toolSessionManager
import controllers.mutationTool.mutationRunner as mutationRunner

import data.mutationDetailsRepository as mutationDetailsRepository
import data.assetRepository as assetRepository
import data.fileIoUtil as fileIoUtil


# -------------------------------------------------------------


def mutationRecreation(mutationId, sessionManager):
    print("Recreating mutation {}".format(mutationId))

    # Get one recreate
    detailsRepository = mutationDetailsRepository.DetailsRepository(sessionManager.mongoConnect)

    details = detailsRepository.getFinalDataById(mutationId)

    assetRepo = assetRepository.AssetRepository(sessionManager.binPath, sessionManager.binPath, sessionManager.mongoConnect)
    recreateOne(details, assetRepo, sessionManager)

    # TODO Save


def batchRecreation(batchId, sessionManager):
    print("Recreating batch {}".format(batchId))

    # Get in batches
    assetRepo = assetRepository.AssetRepository(sessionManager.binPath, sessionManager.binPath, sessionManager.mongoConnect)

    # TODO batch call recreateOne()
    # TODO save batch


def recreateOne(details, assetRepo, sessionManager):
    print("Recreating {}".format(details["_id"]))

    # Base scene
    sessionManager.scene = details.get("baseScene", default=None)
    sessionManager.sequence = details.get("baseSequence", default=None)

    # Asset
    sessionManager.assetId = details.get("asset", default=None)

    # Rotate
    sessionManager.rotation = details.get("rotate", default=None)
    # Mirror
    sessionManager.mirrorAxis = details.get("mirror", default=None)
    # Scale
    sessionManager.scaleAmount = details.get("rotate", default=None)
    # Sign Change
    sessionManager.signChange = details.get("sign", default=None)
    # Deform
    sessionManager.deformPercent = details.get("deformPercent", default=None)
    sessionManager.deformPoint = details.get("deformPoint", default=None)
    sessionManager.deformMu = details.get("deformMu", default=None)
    sessionManager.deformSigma = details.get("deformSigma", default=None)
    sessionManager.deformSeed = details.get("deformSeed", default=None)

    return mutationRunner.performMutation(details["mutation"], assetRepo, sessionManager)
     




