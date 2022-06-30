"""
Enums for the different mutations made up of 
asset location
and transformation
"""

from enum import Enum

# Where / what the asset utilized will be
class AssetLocation(Enum):
    ADD = "ADD"
    SCENE = "SCENE"
    SIGN = "SIGN"
    VEHICLE = "VEHICLE"


# What transformations will be performed on that asset
class Transformation(Enum):
    # Add
    ROTATE = "ROTATE"
    MIRROR = "MIRROR"
    # Scene
    REMOVE = "REMOVE"
    # Sign
    REPLACE = "REPLACE"
    # Vehicle
    INTENSITY = "INTENSITY"
    DEFORM = "DEFORM"
    SCALE = "SCALE"


# Enum of the different types of mutations supported
class Mutation(Enum):
    ADD_ROTATE = AssetLocation.ADD.name + "_" + Transformation.ROTATE.name,
    ADD_MIRROR_ROTATE = AssetLocation.ADD.name + "_" + Transformation.MIRROR.name + "_" + Transformation.ROTATE.name,
    SCENE_REMOVE = AssetLocation.SCENE.name + "_" + Transformation.REMOVE.name,
    SIGN_REPLACE = AssetLocation.SIGN.name + "_" + Transformation.REPLACE.name,
    VEHICLE_INTENSITY = AssetLocation.VEHICLE.name + "_" + Transformation.INTENSITY.name,
    VEHICLE_DEFORM = AssetLocation.VEHICLE.name + "_" + Transformation.DEFORM.name,
    VEHICLE_SCALE = AssetLocation.VEHICLE.name + "_" + Transformation.SCALE.name,




