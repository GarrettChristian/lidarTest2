
from pymongo import MongoClient
import glob, os
import numpy as np
import open3d as o3d

name_label_mapping = {
        0: 'unlabeled',
        1: 'outlier',
        10: 'car',
        11: 'bicycle',
        13: 'bus',
        15: 'motorcycle',
        16: 'on-rails',
        18: 'truck',
        20: 'other-vehicle',
        30: 'person',
        31: 'bicyclist',
        32: 'motorcyclist',
        40: 'road',
        44: 'parking',
        48: 'sidewalk',
        49: 'other-ground',
        50: 'building',
        51: 'fence',
        52: 'other-structure',
        60: 'lane-marking',
        70: 'vegetation',
        71: 'trunk',
        72: 'terrain',
        80: 'pole',
        81: 'traffic-sign',
        99: 'other-object',
        252: 'moving-car',
        253: 'moving-bicyclist',
        254: 'moving-person',
        255: 'moving-motorcyclist',
        256: 'moving-on-rails',
        257: 'moving-bus',
        258: 'moving-truck',
        259: 'moving-other-vehicle'
}

sequenceData = {}
globalData = {}
assetsToSave = []
centerCamPoint = np.array([0, 0, 0.3])

# Box for center points
centerArea = np.array([
        [ -2.5, -2.5, -3], # bottom right
        [ -2.5, -2.5, 3], 
        [ -2.5, 2.5, -3], # top right
        [ -2.5, 2.5, 3],
        [ 2.5, 2.5, -3], # top left
        [ 2.5, 2.5, 3],
        [ 2.5, -2.5, -3], # bottom left
        [ 2.5, -2.5, 3], 
        ]).astype("float64")

# -------------------------------------------------------------

"""
Connect to mongodb 
"""
def mongoConnect():
    configFile = open("../mongoconnect.txt", "r")
    mongoUrl = configFile.readline()
    print("Connecting to: ", mongoUrl)
    configFile.close()
    
    client = MongoClient(mongoUrl)
    db = client["lidar_data"]
    return db


# Asset Prechecks
def checkInclusionBasedOnTriangleMeshAsset(points, mesh):

    obb = mesh.get_oriented_bounding_box()

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacyMesh)

    pointsVector = o3d.utility.Vector3dVector(points)

    indexesWithinBox = obb.get_point_indices_within_bounding_box(pointsVector)

    foundNum = 0
    acceptableNum = 10

    pcdCenter = o3d.geometry.PointCloud()
    pcdCenter.points = o3d.utility.Vector3dVector(centerArea)

    #  Get the asset's bounding box
    centerBox = pcdCenter.get_oriented_bounding_box()
    ignoreIndex = centerBox.get_point_indices_within_bounding_box(pointsVector)
    ignoreIndex = set(ignoreIndex)
    
    for idx in indexesWithinBox:
        if (idx not in ignoreIndex):
            pt = points[idx]
            query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

            occupancy = scene.compute_occupancy(query_point)
            if (occupancy == 1): 
                foundNum += 1
                if (foundNum >= acceptableNum):
                    return True

    return False


def assetIsValid(pcdArr, labelInstance, instance, semantics):
    
    pcdsequence = o3d.geometry.PointCloud()
    pcdsequence.points = o3d.utility.Vector3dVector(pcdArr)

    instancePoints = pcdArr[labelInstance == instance]

    valid = False

    if (np.shape(instancePoints)[0] > 20):
        pcdItem = o3d.geometry.PointCloud()
        pcdItem.points = o3d.utility.Vector3dVector(instancePoints)
        hull, _ = pcdItem.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((0, 0, 1))


        maskInst = (labelInstance != instance) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
        pcdWithoutInstance = pcdArr[maskInst, :]

        pcdAsset = o3d.geometry.PointCloud()
        pcdAsset.points = o3d.utility.Vector3dVector(instancePoints)

        #  Get the asset's bounding box
        obb = pcdAsset.get_oriented_bounding_box()
        boxPoints = np.asarray(obb.get_box_points())
        
        boxVertices = np.vstack((boxPoints, centerCamPoint))

        pcdCastHull = o3d.geometry.PointCloud()
        pcdCastHull.points = o3d.utility.Vector3dVector(boxVertices)
        hull2, _ = pcdCastHull.compute_convex_hull()
        
        assetCenter = obb.get_center()

        # Dist is acceptable
        dist = np.linalg.norm(centerCamPoint - assetCenter)
        if dist < 50:
            incuded = checkInclusionBasedOnTriangleMeshAsset(pcdWithoutInstance, hull2)
            valid = not incuded
        
    return valid



def saveAsset(scene, sequence, instance, semantics, labelInstance):
    global sequenceData
    global globalData
    global assetsToSave

    mask = (labelInstance == instance)
    type = semantics[mask]

    typeName = name_label_mapping[type[0]]

    id = sequence + "-" + scene + "-" + str(instance)
    
    asset = {}
    asset["_id"] = id
    asset["sequence"] = sequence
    asset["scene"] = scene
    asset["instance"] = int(instance)
    asset["type"] = typeName
    asset["typeNum"] = int(type[0])
    asset["points"] = int(np.shape(type)[0])
    asset["sequenceTypeNum"] = sequenceData[sequence].get(typeName, 0)
    asset["globalTypeNum"] = globalData.get(typeName, 0)

    sequenceData[sequence][typeName] = 1 + sequenceData[sequence].get(typeName, 0)
    globalData[typeName] = 1 + globalData.get(typeName, 0)

    assetsToSave.append(asset)


def parseAssets(labelsFileName, binFileName, sequence):

    label_arr = np.fromfile(labelsFileName, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 

    fileName = os.path.basename(labelsFileName).replace('.label', '')

    # Label
    label_arr = np.fromfile(labelsFileName, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    labelInstance = label_arr >> 16 

    # Bin File
    pcdArr = np.fromfile(binFileName, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    pcdArr = np.delete(pcdArr, 3, 1)

    seenInst = set()
    for instance in labelInstance:
        seenInst.add(instance)
    
    for instance in seenInst:

        # Skip the unlabeled asset
        if (instance != 0 and assetIsValid(pcdArr, labelInstance, instance, semantics)):
            saveAsset(fileName, sequence, instance, semantics, labelInstance)


def main():
    global sequenceData
    global globalData
    global assetsToSave

    print("\n\n------------------------------")
    print("\n\nStarting Asset Loader\n\n")

    print("Connecting to Mongo")
    mdb = mongoConnect()
    mdbColAssets = mdb["assets2"]
    mdbColAssetMetadata = mdb["asset_metadata2"]
    print("Connected")

    path = "/home/garrett/Documents/data/dataset/sequences/"

    print("Parsing {} :".format(path))

    num = 0

    for x in range(0, 11):
        
        folderNum = str(x).rjust(2, '0')
        currPath = path + folderNum

        # Add Sequence
        sequenceData[folderNum] = {}
        sequenceData[folderNum]["_id"] = folderNum

        labelFiles = np.array(glob.glob(currPath + "/labels/*.label", recursive = True))
        binFiles = np.array(glob.glob(currPath + "/velodyne/*.bin", recursive = True))
        print("\n\nParsing ", folderNum)

        # Sort
        labelFiles = sorted(labelFiles)
        binFiles = sorted(binFiles)
        
        for index in range(len(labelFiles)):
                
            parseAssets(labelFiles[index], binFiles[index], folderNum)
            print(num, labelFiles[index])
            num += 1

            # Batch insert
            if (len(assetsToSave) >= 2000):
                mdbColAssets.insert_many(assetsToSave)
                assetsToSave = []

        print(sequenceData[folderNum])

        # Save metadata for sequence
        curSequence = sequenceData[folderNum]
        mdbColAssetMetadata.insert_one(curSequence)

    # Batch insert any remaining
    if (len(assetsToSave) != 0):
        mdbColAssets.insert_many(assetsToSave)

    globalData["_id"] = "all"
    mdbColAssetMetadata.insert_one(globalData)

if __name__ == '__main__':
    main()



