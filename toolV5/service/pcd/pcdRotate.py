
import numpy as np
import open3d as o3d
import random

import service.pcd.pcdCommon as pcdCommon

# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Rotation

def mirrorAsset(pcdArrAsset, details, mirrorAxis):

    axis = mirrorAxis
    if (not axis):
        axis = random.randint(0, 1)
    print("Mirror Axis: {}".format(axis))
    details["mirror"] = axis
    pcdArrAsset[:, axis] = pcdArrAsset[:, axis] * -1

    return pcdArrAsset, details


# --------------------------------------------------------------------------
# Rotation


def rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, details, rotation):
    attempts = 0
    success = False
    degrees = []
    
    degrees = getValidRotations(pcdArrAsset, pcdArr, semantics)

    # print(degrees)
    
    while (len(degrees) > 0 and attempts < 10 and not success):
            
        rotateDeg = rotation
        if (not rotateDeg):
            # modifier = random.randint(-4, 4)
            # rotateDeg = random.choice(degrees) + modifier
            # if rotateDeg < 0:
            #     rotateDeg = 0
            # elif rotateDeg > 360:
            #     rotateDeg = 360
            rotateDeg = random.choice(degrees) 
        # print(rotateDeg)
        details['rotate'] = rotateDeg

        pcdArrAssetNew = pcdCommon.rotatePoints(pcdArrAsset, rotateDeg)

        # # Get asset box
        # pcdAsset = o3d.geometry.PointCloud()
        # pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAssetNew)
        # hull, _ = pcdAsset.compute_convex_hull()
        # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        # hull_ls.paint_uniform_color((0, 0, 1))

        # # Get scene
        # pcdScene = o3d.geometry.PointCloud()
        # pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

        # # Color as intensity
        # colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
        # colors[:, 0] = intensity
        # pcdScene.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([hull_ls, pcdScene])
        

        # print("Check on ground")
        maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
        if (details["assetType"] == "traffic-sign"):
            maskGround = (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
            success = pointsAboveGround(pcdArrAssetNew, pcdArr, maskGround) 
        else:
            success = pointsAboveGround(pcdArrAssetNew, pcdArr, maskGround) or pointsWithinDist(pcdArrAssetNew, 5) 

        if (not success):
            details["issue"] = "Points not on correct ground"
        else:
            # print("On correct ground")

            # print("align to Z dim")
            pcdArrAssetNew = alignZdim(pcdArrAssetNew, pcdArr, semantics)

            # # Get asset box
            # pcdAsset = o3d.geometry.PointCloud()
            # pcdAsset.points = o3d.utility.Vector3dVector(pcdArrAssetNew)
            # hull, _ = pcdAsset.compute_convex_hull()
            # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            # hull_ls.paint_uniform_color((0, 0, 1))

            # # Get scene
            # pcdScene = o3d.geometry.PointCloud()
            # pcdScene.points = o3d.utility.Vector3dVector(pcdArr)

            # # Color as intensity
            # colors = np.zeros(np.shape(pcdArr), dtype=np.float64)
            # colors[:, 0] = intensity
            # pcdScene.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([hull_ls, pcdScene])
            

            # print("Check not in walls")
            success = not assetIntersectsWalls(pcdArrAssetNew, pcdArr, semantics)
            if (not success):
                details["issue"] = "Within the walls"
            else:
                # print("Not in walls")

                # print("Check unobsuccured")
                success = assetIsNotObsured(pcdArrAssetNew, pcdArr, semantics)
                if (not success):
                    details["issue"] = "Asset Obscured"
                else:
                    # print("Asset Unobscured")
                    details["issue"] = ""
                    pcdArrAsset = pcdArrAssetNew
                    # print("Removing shadow")
                    pcdArr, intensity, semantics, labelInstance, pointsRemoved = removeLidarShadow(pcdArrAssetNew, pcdArr, intensity, semantics, labelInstance)
                    details["pointsRemoved"] = pointsRemoved
                    details["pointsAdded"] = int(np.shape(pcdArrAssetNew)[0])
                    details["pointsAffected"] = int(np.shape(pcdArrAssetNew)[0]) + pointsRemoved


        attempts += 1
    
    return success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details

# --------------------------------------------------------------------------
# Cuts the points within the lidarShadow

"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def removeLidarShadow(asset, scene, intensity, semantics, instances):

    lidarShadowMesh = pcdCommon.getLidarShadowMesh(asset)

    mask = pcdCommon.checkInclusionBasedOnTriangleMesh(scene, lidarShadowMesh)
    pointsIncluded = int(np.sum(mask))
    mask = np.logical_not(mask)

    sceneResult = scene[mask, :]
    intensityResult = intensity[mask]
    semanticsResult = semantics[mask]
    instancesResult = instances[mask]

    return (sceneResult, intensityResult, semanticsResult, instancesResult, pointsIncluded)

# --------------------------------------------------------------------------
# Pre Check for valid placement

def getValidRotations(asset, scene, semantics):
    
    maskNotGround = (semantics != 0) & (semantics != 1) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    scene = scene[maskNotGround]
    scene[:, 2] = 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)

    #  Get the asset's bounding box
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    obb = pcdAsset.get_oriented_bounding_box()
    assetCenter = obb.get_center() 

    voxelGridWalls = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    linePoints = getPointsOnLine((0, 0), assetCenter, 40)
    degrees = []


    # for deg in range(0, 360, 5):
    for deg in range(0, 360):
        lineRotated = pcdCommon.rotatePoints(linePoints, deg)
        
        included = voxelGridWalls.check_if_included(o3d.utility.Vector3dVector(lineRotated))
        included = np.logical_or.reduce(included, axis=0)
        if (not included): 
            degrees.append(deg)     


    return degrees


def getPointsOnLine(p1, p2, count):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

    m = (y2 - y1) / (x2 - x1)

    b = y1 - (m * x1)

    minX = min(x1, x2)
    maxX = max(x1, x2)

    xVals = []
    yVals = []
    step = abs((maxX - minX) / count)
    curX = minX
    while curX <= maxX:
        y = m*curX + b

        xVals.append(curX)
        yVals.append(y)

        curX += step

    pointsOnLine = np.zeros((len(xVals), 3), dtype=np.float)
    pointsOnLine[:, 0] = xVals
    pointsOnLine[:, 1] = yVals

    return pointsOnLine




# --------------------------------------------------------------------------
# Post Checks



def assetIsNotObsured(points, scene, semantics):

    # Get everything, but the ground
    maskNotGround = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    pcdArrExceptGround = scene[maskNotGround, :]

    # Convert to open3d object
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(points)

    # Create a box between the camera location to the bounding box
    obb = pcdAsset.get_oriented_bounding_box()
    boxPoints = np.asarray(obb.get_box_points())
    boxVertices = np.vstack((boxPoints, pcdCommon.centerCamPoint))

    # Create the triangle mesh of the box to the center
    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(boxVertices)
    hull, _ = pcdCastHull.compute_convex_hull()    

    # If we don't find anything included the asset is not obscured
    return not checkInclusionBasedOnTriangleMeshAsset(pcdArrExceptGround, hull)   



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


"""
Checks that all points exist above the ground
"""
def pointsAboveGround(points, scene, maskGround):
    
    
    pcdArrGround = scene[maskGround, :]

    # Remove Z dim
    pcdArrGround[:, 2] = 0
    pointsCopy = np.copy(points)
    pointsCopy[:, 2] = 0

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArrGround)

    voxelGridGround = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdScene, voxel_size=1)

    included = voxelGridGround.check_if_included(o3d.utility.Vector3dVector(pointsCopy))
    # return np.logical_and.reduce(included, axis=0)

    return sum(included) >= (len(included) / 3)

"""
Alternative to existing on the ground is being close enough to the camera
"""
def pointsWithinDist(points, lessThanDist):
    
    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    dist = np.linalg.norm(centerOfPoints - pcdCommon.centerCamPoint)

    return dist < lessThanDist


"""
assetIntersectsWalls
Checks if a given asset intersects anything that isnt the ground

@param asset to check intersection
@param scene, full scene will remove the road
@param semantics for the given scene
"""
def assetIntersectsWalls(asset, scene, semantics):

    # Get everything, but the ground
    maskNotGround = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    pcdArrExceptGround = scene[maskNotGround, :]

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArrExceptGround)

    voxelGridNonRoad = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdScene, voxel_size=0.1)

    included = voxelGridNonRoad.check_if_included(o3d.utility.Vector3dVector(asset))
    return np.logical_or.reduce(included, axis=0)




"""
Aligns the asset to the average height of the ground
"""
def alignZdim(asset, scene, semantics):
    assetCopy = np.copy(asset)

    # Get the ground
    maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
    pcdArrGround = scene[maskGround, :]

    # Get the bounding box for the asset
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    aabb = pcdAsset.get_axis_aligned_bounding_box()
    boxPoints = np.asarray(aabb.get_box_points())

    # Scale the box Z dim to encompass the ground
    boxMinZ = np.min(boxPoints.T[2])
    boxMaxZ = np.max(boxPoints.T[2])
    # Lower the box's bottom
    bP1 = boxPoints[boxPoints[:, 2] == boxMinZ]
    bP1[:, 2] = bP1[:, 2] - 20
    # Increase the box's top
    bP2 = boxPoints[boxPoints[:, 2] == boxMaxZ]
    bP2[:, 2] = bP2[:, 2] + 20

    # Recreate the box
    boxPointsZLarger = np.vstack((bP1, bP2))
    largerBox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(boxPointsZLarger))

    # Reduce the ground to only the portion found within the new box
    mask = largerBox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pcdArrGround))
    onlyByCar = pcdArrGround[mask]

    # Get the average values for the ground
    groundAvg = 0
    if (np.shape(onlyByCar)[0] != 0):
        groundAvg = np.average(onlyByCar.T[2])
        # print("ground")
    else:
        groundAvg = np.average(pcdArrGround.T[2])
        # print("avg")

    # Use the boundin box min to get the change to add to the Z dim to align the asset to the 
    assetMin = np.min(asset[:, 2])
    assetMin = round(assetMin, 2)
    groundAvg = round(groundAvg, 2)
    change = groundAvg - boxMinZ

    # Align the asset to the ground
    assetCopy[:, 2] = assetCopy[:, 2] + change

    return assetCopy






