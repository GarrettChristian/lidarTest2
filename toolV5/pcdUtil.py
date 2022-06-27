"""
pcdUtil
Functions for performing the mutations or manipulating the point clouds

@Author Garrett Christian
@Date 6/23/22
"""

import numpy as np
import open3d as o3d
import math
from sklearn.neighbors import NearestNeighbors
import sys
import random
import copy

import globals


# --------------------------------------------------------------------------
# Constants

X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2
I_AXIS = 3

centerCamPoint = np.array([0, 0, 0.3])

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
# Util

def removeAssetScene(pcdArrAsset, pcdArr, intensity, semantics, labelInstance):
    mask = np.ones(np.shape(pcdArr)[0], dtype=bool)

    pointsSet = set()

    for point in pcdArrAsset:
        pointsSet.add((point[0], point[1], point[2]))

    for index in range(0, np.shape(pcdArr)[0]):
        if ((pcdArr[index][0], pcdArr[index][1], pcdArr[index][2]) in pointsSet):
            mask[index] = False

    pcdArrRemoved = pcdArr[mask]
    intensityRemoved = intensity[mask]
    semanticsRemoved = semantics[mask]
    labelInstanceRemoved = labelInstance[mask]

    return pcdArrRemoved, intensityRemoved, semanticsRemoved, labelInstanceRemoved


def combine(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset):
    pcdArrCombined = np.vstack((pcdArr, pcdArrAsset))
    intensityCombined = np.hstack((intensity, intensityAsset))
    semanticsCombined = np.hstack((semantics, semanticsAsset))
    labelInstanceCombined = np.hstack((labelInstance, labelInstanceAsset))

    return pcdArrCombined, intensityCombined, semanticsCombined, labelInstanceCombined


# --------------------------------------------------------------------------
# Deform

# http://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
def deform(asset, details):

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    # Select random point
    pointIndex = np.random.choice(asset.shape[0], 1, replace=False)
    print(pointIndex)

    # Nearest k points
    assetNumPoints = np.shape(asset)[0]
    percentDeform = random.uniform(0.05, 0.12)
    k = int(assetNumPoints * percentDeform)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAsset)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcdAsset.points[pointIndex], k)
    # np.asarray(pcdAsset.colors)[idx[1:], :] = [0, 0, 1]

    mu, sigma = 0.05, 0.04
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma, (k))
    # noise = np.lexsort((noise[:,0], noise[:,1]))
    noise = np.sort(noise)[::-1]
    # print(np.shape(noise))
    # print(noise)
    print("total points {}".format(assetNumPoints))
    print("deformPoints {}".format(k))
    print("deformPercent {}".format(percentDeform))
    print("deformMu {}".format(mu))
    print("deformSigma {}".format(sigma))
    details["deformPercent"] = percentDeform
    details["deformPoints"] = k
    details["deformMu"] = mu
    details["deformSigma"] = sigma

    for index in range(0, len(idx)):
        asset[idx[index]] = translatePointFromCenter(asset[idx[index]], noise[index])
        # if index != 0:
        #     pcdAsset.colors[idx[index]] = [0, 0, 1 - (index * .002)]

    # pcdAsset.points = o3d.utility.Vector3dVector(asset)

    # print(idx)

    # o3d.visualization.draw_geometries([pcdAsset])
    # o3d.visualization.draw_geometries([pcdAsset, pcd])

    return asset, details


# --------------------------------------------------------------------------
# Intensity

def intensityChange(intensityAsset, type, details):

    # Create a mask that represents the portion to change the intensity for
    mask = np.ones(np.shape(intensityAsset), dtype=bool)

    if (type in globals.vehicles):    
        dists = nearestNeighbors(intensityAsset, 2)
        class0 = intensityAsset[dists[:, 1] == 0]
        class1 = intensityAsset[dists[:, 1] == 1]

        mask = dists[:, 1] == 0
        if np.shape(class0)[0] < np.shape(class1)[0]:
            mask = dists[:, 1] == 1

    average = np.average(intensityAsset[mask])
    
    mod = random.uniform(.1, .3)
    if average > .1:
        mod = random.uniform(-.1, -.3)

    details["intensity"] = mod
    
    print("Intensity {}".format(mod))
    
    print(mask)
    print(intensityAsset)
    intensityAsset = np.where(mask, intensityAsset + mod, intensityAsset)
    intensityAsset = np.where(intensityAsset < 0, 0, intensityAsset)
    print(intensityAsset)
    print(average)

    return intensityAsset, details


"""
nearestNeighbors
Seperates the values into k groups using nearest neighbors

https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian
"""
def nearestNeighbors(values, nbr_neighbors):

    zeroCol = np.zeros((np.shape(values)[0],), dtype=bool)
    valuesResized = np.c_[values, zeroCol]
    # np.append(np.array(values), np.array(zeroCol), axis=1)
    # valuesResized = np.hstack((np.array(values), np.array(zeroCol)))

    nn = NearestNeighbors(n_neighbors=nbr_neighbors, metric='cosine', algorithm='brute').fit(valuesResized)
    dists, idxs = nn.kneighbors(valuesResized)

    return dists


# --------------------------------------------------------------------------
# Rotation

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
        lineRotated = rotatePoints(linePoints, deg)
        
        included = voxelGridWalls.check_if_included(o3d.utility.Vector3dVector(lineRotated))
        included = np.logical_or.reduce(included, axis=0)
        if (not included): 
            degrees.append(deg)     


    return degrees


def rotate(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, details):
    attempts = 0
    success = False
    degrees = []
    
    degrees = getValidRotations(pcdArrAsset, pcdArr, semantics)

    print(degrees)
    
    while (len(degrees) > 0 and attempts < 10 and not success):
            
        rotateDeg = globals.rotation
        if (not rotateDeg):
            # modifier = random.randint(-4, 4)
            # rotateDeg = random.choice(degrees) + modifier
            # if rotateDeg < 0:
            #     rotateDeg = 0
            # elif rotateDeg > 360:
            #     rotateDeg = 360
            rotateDeg = random.choice(degrees) 
        print(rotateDeg)
        details['rotate'] = rotateDeg

        pcdArrAssetNew = rotatePoints(pcdArrAsset, rotateDeg)

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
        

        print("Check on ground")
        maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
        if (details["assetType"] == "traffic-sign"):
            maskGround = (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
            success = pointsAboveGround(pcdArrAssetNew, pcdArr, maskGround) 
        else:
            success = pointsAboveGround(pcdArrAssetNew, pcdArr, maskGround) or pointsWithinDist(pcdArrAssetNew, 5) 

        if (success):
            print("On correct ground")

            print("align to Z dim")
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
            

            print("Check not in walls")
            success = not assetIntersectsWalls(pcdArrAssetNew, pcdArr, semantics)
            if (success):
                print("Not in walls")

                print("Check unobsuccured")
                success = assetIsNotObsured(pcdArrAssetNew, pcdArr, semantics)
                if (success):
                    print("Asset Unobscured")
                    pcdArrAsset = pcdArrAssetNew
                    print("Removing shadow")
                    pcdArr, intensity, semantics, labelInstance = removeLidarShadow(pcdArrAssetNew, pcdArr, intensity, semantics, labelInstance)

        attempts += 1
    
    return success, pcdArrAsset, pcdArr, intensity, semantics, labelInstance, details



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
    boxVertices = np.vstack((boxPoints, centerCamPoint))

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
    pcdArrGround[:, Z_AXIS] = 0
    pointsCopy = np.copy(points)
    pointsCopy[:, Z_AXIS] = 0

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

    dist = np.linalg.norm(centerOfPoints - centerCamPoint)

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
        print("ground")
    else:
        groundAvg = np.average(pcdArrGround.T[2])
        print("avg")

    # Use the boundin box min to get the change to add to the Z dim to align the asset to the 
    assetMin = np.min(asset[:, 2])
    assetMin = round(assetMin, 2)
    groundAvg = round(groundAvg, 2)
    change = groundAvg - boxMinZ

    # Align the asset to the ground
    assetCopy[:, 2] = assetCopy[:, 2] + change

    return assetCopy



"""
Translate a point to a new location based on the center

# https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
"""
def translatePointFromCenter(point, amount):

    # pcdPoints = o3d.geometry.PointCloud()
    # pcdPoints.points = o3d.utility.Vector3dVector(pointsCopy)
    # obb = pcdPoints.get_oriented_bounding_box()
    # centerOfPoints = obb.get_center()
    
    # Note the 0 here is the center point
    vX = point[0] - 0
    vY = point[1] - 0

    uX = vX / math.sqrt((vX * vX) + (vY * vY))
    uY = vY / math.sqrt((vX * vX) + (vY * vY))

    newPoint = np.array([0, 0, point[2]])
    newPoint[0] = point[0] + (amount * uX)
    newPoint[1] = point[1] + (amount * uY)

    return newPoint


"""
rotatePoints
Rotates points by a given angle around the center camera
"""
def rotatePoints(points, angle):
    # Preconditions for asset rotation
    if (angle < 0 or angle > 360):
        print("Only angles between 0 and 360 are accepable")
        exit()
    elif (not np.size(points)):
        print("Points are empty")
        exit()

    # Do nothing if asked to rotate to the same place
    if (angle == 0 or angle == 360):
        return points

    pointsRotated = np.copy(points)
    for point in pointsRotated:
        pt = (point[X_AXIS], point[Y_AXIS])
        newLocation = rotateOnePoint((0, 0), pt, angle)
        point[X_AXIS] = newLocation[X_AXIS]
        point[Y_AXIS] = newLocation[Y_AXIS]

    return pointsRotated


"""
Rotate a point counterclockwise by a given angle around a given origin.
In degrees
"""
def rotateOnePoint(origin, point, angle):

    radians = (angle * math.pi) / 180
    return rotateOnePointRadians(origin, point, radians)


"""
Rotate a point counterclockwise by a given angle around a given origin.
In radians
https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
"""
def rotateOnePointRadians(origin, point, radians):

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(radians) * (px - ox) - math.sin(radians) * (py - oy)
    qy = oy + math.sin(radians) * (px - ox) + math.cos(radians) * (py - oy)
    return qx, qy

"""
getAngleRadians
gets the angle in radians created between p0 -> p1 and p0 -> p2
p0 is the origin

https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
"""
def getAngleRadians(p0, p1, p2):
    p0x = p0[0] 
    p0y = p0[1]
    p1x = p1[0] 
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]

    p01 = math.sqrt(((p0x - p1x) * (p0x - p1x)) + ((p0y - p1y) * (p0y - p1y)))
    p02 = math.sqrt(((p0x - p2x) * (p0x - p2x)) + ((p0y - p2y) * (p0y - p2y)))
    p12 = math.sqrt(((p1x - p2x) * (p1x - p2x)) + ((p1y - p2y) * (p1y - p2y)))

    result = np.arccos(((p01 * p01) + (p02 * p02) - (p12 * p12)) / (2 * p01 * p02))

    return result


# --------------------------------------------------------------------------
# Mesh & Shadow


"""
Creates a mask the size of the points array
True is included in the mesh
False is not included in the mesh
"""
def checkInclusionBasedOnTriangleMesh(points, mesh):

    obb = mesh.get_oriented_bounding_box()

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    mask = np.zeros((np.shape(points)[0],), dtype=bool)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacyMesh)

    # pcdAsset = o3d.geometry.PointCloud()
    pointsVector = o3d.utility.Vector3dVector(points)

    indexesWithinBox = obb.get_point_indices_within_bounding_box(pointsVector)

    for idx in indexesWithinBox:
        pt = points[idx]
        query_point = o3d.core.Tensor([pt], dtype=o3d.core.Dtype.Float32)

        occupancy = scene.compute_occupancy(query_point)
        mask[idx] = (occupancy == 1)

    return mask


"""
https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
"""
def getLidarShadowMesh(asset):

    # Prepare asset and scene point clouds
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    #  Get the asset's hull mesh
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    
    castHullPoints = np.array([])
    for point1 in hullVertices:

        ba = centerCamPoint - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen

        pt2 = centerCamPoint + ((-100) * ba2)

        if (np.size(castHullPoints)):
            castHullPoints = np.vstack((castHullPoints, [pt2]))
        else:
            castHullPoints = np.array([pt2])

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(castHullPoints)
    hull2, _ = pcdCastHull.compute_convex_hull()

    # hull2.scale(0.5, hull2.get_center())
    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdShadow = o3d.geometry.PointCloud()
    pcdShadow.points = o3d.utility.Vector3dVector(combinedVertices)
    shadowMesh, _ = pcdShadow.compute_convex_hull()

    return shadowMesh


"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def removeLidarShadow(asset, scene, intensity, semantics, instances):

    cutPointsHull = getLidarShadowMesh(asset)

    mask = checkInclusionBasedOnTriangleMesh(scene, cutPointsHull)
    mask = np.logical_not(mask)

    sceneResult = scene[mask, :]
    intensityResult = intensity[mask]
    semanticsResult = semantics[mask]
    instancesResult = instances[mask]

    return (sceneResult, intensityResult, semanticsResult, instancesResult)



# --------------------------------------------------------------------------
# Remove


"""
Checks if point (c) is left of the line drawn from a to b
Left is the same as counter clockwise given a is the camera

https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
a = line point 1; b = line point 2; c = point to check against.
"""
def isLeft(lineP1, lineP2, point):
    aX = lineP1[0]
    bX = lineP2[0]
    cX = point[0]
    aY = lineP1[1]
    bY = lineP2[1]
    cY = point[1]
    return ((bX - aX) * (cY - aY) - (bY - aY) * (cX - aX)) > 0


"""
perpDistToLine
Gets the perpendicular distance from a point to the line between two points

https://math.stackexchange.com/questions/422602/convert-two-points-to-line-eq-ax-by-c-0
https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
"""
def perpDistToLine(lineP1, lineP2, point):
    x1 = lineP1[0]
    y1 = lineP1[1]
    x2 = lineP2[0]
    y2 = lineP2[1]
    x3 = point[0]
    y3 = point[1]

    a = y1 - y2
    b = x2 - x1
    c = (x1 * y2) - (x2 * y1)

    dist = abs((a * x3 + b * y3 + c)) / (math.sqrt(a * a + b * b))

    return dist


"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def replaceBasedOnShadow(asset, scene, intensity, semantics, instances, details):

    # Get the objects shadow
    shadow = getLidarShadowMesh(asset)
    shadowVertices = np.asarray(shadow.vertices)
    
    # Check above if this is a vehicle type
    # This avoids cases where is copies large chunks of road / sidewalk into the walls
    if (details["typeNum"] in globals.instancesVehicle.keys()): 
        shadowVerticesRaised = np.copy(shadowVertices)
        shadowVerticesRaised[:, 2] = shadowVerticesRaised[:, 2] + 1
        pcdShadowRaised = o3d.geometry.PointCloud()
        pcdShadowRaised.points = o3d.utility.Vector3dVector(shadowVerticesRaised)
        hullShadowRaised, _ = pcdShadowRaised.compute_convex_hull()
        # 70 veg, 50: 'building', 51: 'fence',
        maskAbove = (semantics == 70) | (semantics == 50) | (semantics == 51)
        sceneVegitation = scene[maskAbove]
        maskAbove = checkInclusionBasedOnTriangleMesh(sceneVegitation, hullShadowRaised)

        if (np.sum(maskAbove) > 30):
            print("TOO MANY ABOVE {} ".format(np.sum(maskAbove)))
            return False, None, None, None, None,

    # Remove

    #  Get the asset's convex hull
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    assetBox = pcdAsset.get_oriented_bounding_box()

    # Get the left and right points of the hull
    # Gets the perpendicular distance to center line taking max of each side 
    # NOTE Left is synonymous for counter clockwise
    midPoint = hull.get_center()
    leftMax = sys.maxsize * -1
    rightMax = sys.maxsize * -1
    leftPoint = [0, 0, 0]
    rightPoint = [0, 0, 0]
    for point in hullVertices:
        distFromCenterLine = perpDistToLine(centerCamPoint, midPoint, point)

        if (isLeft(midPoint, centerCamPoint, point)):
            if (distFromCenterLine > leftMax):
                leftMax = distFromCenterLine
                leftPoint = point
        else:
            if (distFromCenterLine > rightMax):
                rightMax = distFromCenterLine
                rightPoint = point

    # Provide two points for hull sides in center
    midPointMaxZ = hull.get_center() 
    midPointMinZ = hull.get_center()
    midPointMaxZ[2] = assetBox.get_max_bound()[2]
    midPointMinZ[2] = assetBox.get_min_bound()[2] - 1

    # Sort the shadow points into left and right
    replaceLeftShadow = [midPointMaxZ, midPointMinZ]
    replaceRightShadow = [midPointMaxZ, midPointMinZ]
    for point in shadowVertices:
        if (isLeft(midPoint, centerCamPoint, point)):
            replaceLeftShadow.append(point)
        else:
            replaceRightShadow.append(point)

    # Validate that each side has enough points to be constructed into a mask (4 points min)
    if (len(replaceLeftShadow) < 4 or len(replaceRightShadow) < 4):
        print("Not enough points left {} or right {} shadow".format(len(replaceLeftShadow), len(replaceRightShadow)))
        return False, None, None, None, None

    # Get the angles for left and right
    angleLeft = getAngleRadians(centerCamPoint, leftPoint, midPoint)
    angleRight = getAngleRadians(centerCamPoint, midPoint, rightPoint)
    angleLeft = angleLeft * (180 / math.pi)
    angleRight = angleRight * (180 / math.pi)
    
    # Rotate the halves of the shadow
    replaceLeftShadow = rotatePoints(replaceLeftShadow, 360 - angleLeft)
    replaceRightShadow = rotatePoints(replaceRightShadow, angleRight)

    # Convert shadow halves to masks
    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(replaceLeftShadow)
    shadowRotated, _ = pcdCastHull.compute_convex_hull()
    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(replaceRightShadow)
    shadowRotated2, _ = pcdCastHull.compute_convex_hull()

    # Get points included within the halves of the shadow
    maskIncluded = checkInclusionBasedOnTriangleMesh(scene, shadowRotated)
    pcdIncluded = scene[maskIncluded]
    intensityIncluded = intensity[maskIncluded]
    semanticsIncluded = semantics[maskIncluded]
    instancesIncluded = instances[maskIncluded]
    maskIncluded2 = checkInclusionBasedOnTriangleMesh(scene, shadowRotated2)
    pcdIncluded2 = scene[maskIncluded2]
    intensityIncluded2 = intensity[maskIncluded2]
    semanticsIncluded2 = semantics[maskIncluded2]
    instancesIncluded2 = instances[maskIncluded2]


    # Validate that the points don't include any semantic types that can't be used in a replacement
    # This prevents situations with half cars being copied
    semSetInval = set()
    for sem in semanticsIncluded:
        if (sem in globals.instancesVehicle.keys()
            or sem in globals.instancesWalls.keys()):
            semSetInval.add(sem)
    for sem in semanticsIncluded2:
        if (sem in globals.instancesVehicle.keys()
            or sem in globals.instancesWalls.keys()):
            semSetInval.add(sem)
    
    if (len(semSetInval) > 0):
        invalidSem = ""
        for sem in semSetInval:
            invalidSem += (globals.name_label_mapping[sem] + " ")
        print("Invalid semantics to replace with: {}".format(invalidSem))
        return False, None, None, None, None


    # Rotate any points included in the shadow halves to fill the hole
    if (len(pcdIncluded) > 0):
        pcdIncluded = rotatePoints(pcdIncluded, angleLeft)
    else:
        print("left points empty")
    if (len(pcdIncluded2) > 0):
        pcdIncluded2 = rotatePoints(pcdIncluded2, 360 - angleRight)
    else:
        print("right points empty")

    # Combine the left and right replacement points
    pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded = combine(pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded,
                                                                                pcdIncluded2, intensityIncluded2, semanticsIncluded2, instancesIncluded2)

    # TODO REMOVE Visualization
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(shadow)
    # hull_ls.paint_uniform_color((0, 0.5, 1))
    # hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated)
    # hull_ls2.paint_uniform_color((0, 1, 0.5))
    # hull_ls22 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated2)
    # hull_ls22.paint_uniform_color((1, 0, 0.5))

    # pcdNewAddition = o3d.geometry.PointCloud()
    # pcdNewAddition.points = o3d.utility.Vector3dVector(pcdIncluded)
    # pcdScene = o3d.geometry.PointCloud()
    # pcdScene.points = o3d.utility.Vector3dVector(scene)

    # pcdCast2 = o3d.geometry.PointCloud()
    # pcdCast2.points = o3d.utility.Vector3dVector(np.asarray([rightPoint, leftPoint, midPointMinZ, midPointMaxZ, midPoint]))
    # pcdCast2.paint_uniform_color((.6, 0, .6))

    # o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, pcdNewAddition, pcdScene, pcdCast2])


    # Combine the new points with the scene to fill the asset's hole 
    sceneReplace, intensityReplace, semanticsReplace, instancesReplace = combine(pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded,
        scene, intensity, semantics, instances)

    return True, sceneReplace, intensityReplace, semanticsReplace, instancesReplace



# --------------------------------------------------------------------------
# Sign Change


"""
Get closest to the two bounding points selected
"""
def closestBoundingTwo(minPt, maxPt, asset):

    # Remove Z dim
    assetCopy = np.copy(asset)
    assetCopy[:, 2] = 0

    # Find closest
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(assetCopy)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(minPt, 1)
    [k, idx2, _] = pcd_tree.search_knn_vector_3d(maxPt, 1)

    return asset[idx][0], asset[idx2][0]




def getSignMesh(details):

    sign = random.choice(list(globals.Signs))
    signMesh = None

    if (globals.Signs.SPEED == sign):

        signMesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.6, depth=0.75)

    elif (globals.Signs.STOP == sign):

        box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.75, depth=0.30)
        box2 = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.30, depth=0.75)

        box.translate([0, 0, 0], relative=False)
        box2.translate([0, 0, 0], relative=False)

        signVertices = np.vstack((np.array(box.vertices), np.array(box2.vertices)))

        pcdSign = o3d.geometry.PointCloud()
        pcdSign.points = o3d.utility.Vector3dVector(signVertices)

        #  Get the asset's hull mesh
        signMesh, _ = pcdSign.compute_convex_hull()

    elif (globals.Signs.CROSSBUCK == sign):

        box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.22, depth=1.22)
        box2 = o3d.geometry.TriangleMesh.create_box(width=0.05, height=1.22, depth=0.22)

        box.translate([0, 0, 0], relative=False)
        box2.translate([0, 0, 0], relative=False)

        signMesh = box + box2

        rotation2 = signMesh.get_rotation_matrix_from_xyz((45, 0, 0))
        signMesh.rotate(rotation2, center=box.get_center())

    elif (globals.Signs.WARNING == sign):

        signMesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.76, depth=0.76)    
        rotation = signMesh.get_rotation_matrix_from_xyz((45, 0, 0))
        signMesh.rotate(rotation, center=signMesh.get_center())
        
    elif (globals.Signs.YEILD == sign):

        yeildPoints = np.array([[0, -0.455, 0.91], 
                            [0, 0, 0.91], 
                            [0, 0.455, 0.91], 
                            [0, 0, 0],
                            [0.05, -0.455, 0.91], 
                            [0.05, 0, 0.91], 
                            [0.05, 0.455, 0.91], 
                            [0.05, 0, 0]])

        pcdSign = o3d.geometry.PointCloud()
        pcdSign.points = o3d.utility.Vector3dVector(yeildPoints)

        #  Get the asset's hull mesh
        signMesh, _ = pcdSign.compute_convex_hull()

    else: 
        print("Sign type {} not supported".format(sign))

    signName = str(sign).replace("Sign.", "")
    details["sign"] = signName

    return signMesh, details



def pointsToMesh(mesh, assetData, sceneData):

    asset, intensityAsset, semanticsAsset, instancesAsset = assetData
    scene, intensity, semantics, instances = sceneData

    # http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html
    # Calulate intersection for scene to mesh points
    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    sceneRays = o3d.t.geometry.RaycastingScene()
    sceneRays.add_triangles(legacyMesh)

    if (np.shape(scene)[0] < 1 or np.shape(asset)[0] < 1):
        print("SCENE or ASSET PROVIDED EMPTY: SCENE {}, ASSET {}".format(np.shape(scene)[0], np.shape(asset)[0]))
        return False, (None, None, None, None), (None, None, None, None)

    raysVectorsScene = []
    for point in scene:
        raysVectorsScene.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectorsScene, dtype=o3d.core.Dtype.Float32)
    ans = sceneRays.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))

    # Split between intersect and non intersect
    intensityIntersect = intensity[hit.numpy()]
    semanticsIntersect = semantics[hit.numpy()]
    instancesIntersect = instances[hit.numpy()]
    nonHit = np.logical_not(hit.numpy())
    sceneNonIntersect = scene[nonHit]
    intensityNonIntersect = intensity[nonHit]
    semanticsNonIntersect = semantics[nonHit]
    instancesNonIntersect = instances[nonHit]
    
    newAssetScene = []
    for vector in pointsOnMesh:
        newAssetScene.append(vector.numpy())

    # Calulate intersection for sign points
    raysVectorsMesh = []
    for point in asset:
        raysVectorsMesh.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectorsMesh, dtype=o3d.core.Dtype.Float32)
    ans = sceneRays.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))

    newAsset = []
    for vector in pointsOnMesh:
        newAsset.append(vector.numpy())
    
    intensityAsset = intensityAsset[hit.numpy()] 
    semanticsAsset = semanticsAsset[hit.numpy()]
    instancesAsset = instancesAsset[hit.numpy()]

    print(len(newAsset))
    print(len(newAssetScene))

    if len(newAsset) == 0 or len(newAssetScene) == 0:
        print("GOT NONE OF THE OG ASSET {} OR NONE OF SCENE {}".format(len(newAsset), len(newAssetScene)))
        return False, (None, None, None, None), (None, None, None, None)

    # Fix the intensity of each of the points in the scene that were pulled into the sign by using the closest sign point
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(newAsset)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    for pointIndex in range(0, len(newAssetScene)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(newAssetScene[pointIndex], 1)
        intensityIntersect[pointIndex] = intensityAsset[idx]
        semanticsIntersect[pointIndex] = semanticsAsset[idx]
        instancesIntersect[pointIndex] = instancesAsset[idx]

    # Combine the original points of the asset and the new scene points
    newAsset, intensityAsset, semanticsAsset, instancesAsset = combine(newAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                    newAssetScene, intensityIntersect, semanticsIntersect, instancesIntersect)

    # Return revised scene
    newAssetData = (newAsset, intensityAsset, semanticsAsset, instancesAsset)
    newSceneData = (sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect)
    return True, newAssetData, newSceneData


def signReplace(signAsset, intensityAsset, semanticsAsset, instancesAsset, scene, intensity, semantics, instances, details):

    # Seperate the two portions of a tagged sign
    pole = signAsset[semanticsAsset == 80]
    intensityPole = intensityAsset[semanticsAsset == 80]
    semanticsPole = semanticsAsset[semanticsAsset == 80]
    instancesPole = instancesAsset[semanticsAsset == 80]
    sign = signAsset[semanticsAsset == 81]
    intensitySign = intensityAsset[semanticsAsset == 81]
    semanticsSign = semanticsAsset[semanticsAsset == 81]
    instancesSign = instancesAsset[semanticsAsset == 81]

    # Validate that there are enough points to make a hull from
    if (np.shape(pole)[0] < 5 or np.shape(sign)[0] < 5):
        print("Sign {} pole {}, too little points".format(np.shape(sign)[0], np.shape(pole)[0]))
        return False, None, None, None, None, None, None, None, None, details
    
    # Get the bounding box for both the sign and pole
    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(sign)
    signBox = pcdSign.get_oriented_bounding_box()

    pcdPole = o3d.geometry.PointCloud()
    pcdPole.points = o3d.utility.Vector3dVector(pole)
    poleBox = pcdPole.get_oriented_bounding_box()

    # Center the new sign on the pole
    signCenter = poleBox.get_center()
    
    # Get bounds to align the sign to 
    minSign, maxSign = closestBoundingTwo(signBox.get_min_bound(), signBox.get_max_bound(), sign)

    minZSign = sys.maxsize
    for point in sign:
        minZSign = min(minZSign, point[2])

    maxSign[2] = minZSign
    minSign[2] = minZSign
    signCenter[2] = minZSign
    signLen = np.linalg.norm(maxSign - minSign)

    # Create mesh of new sign
    signMesh, details = getSignMesh(details)

    # Get the bounds and height of the current sign
    meshMin = signMesh.get_min_bound()
    meshMax = signMesh.get_max_bound()
    heightMesh = meshMax[2] - meshMin[2]
    meshMax[2] = minZSign
    meshMin[2] = minZSign
    
    # Validate that the sign is not too low to the ground
    if (minZSign < -1):
        print("Sign too low min {}".format(minZSign))
        return False, None, None, None, None, None, None, None, None, details

    # Validate that the og sign and new sign are roughly the same size
    meshLen = np.linalg.norm(meshMin - meshMin)
    if (np.absolute(meshLen - signLen) > 2):
        print("Distance to sign too great: mesh len {}, sign len {}".format(meshLen, signLen))
        return False, None, None, None, None, None, None, None, None, details


    # Move the mesh to new location based on: pole center, original sign min, and height of the new sign
    signCenter[2] = minZSign + (heightMesh / 2) 
    signMesh.translate(signCenter, relative=False)
    angleSign = getAngleRadians(signCenter, minSign, signMesh.get_min_bound())

    # Get two rotation options to match the original sign
    rotation = signMesh.get_rotation_matrix_from_xyz((0, 0, angleSign * -1))
    rotation2 = signMesh.get_rotation_matrix_from_xyz((0, 0, angleSign))
    signMeshRotate1 = copy.deepcopy(signMesh)
    signMeshRotate2 = copy.deepcopy(signMesh)
    signMeshRotate1.rotate(rotation, center=signMeshRotate1.get_center())
    signMeshRotate2.rotate(rotation2, center=signMeshRotate2.get_center())

    # Get the rotation that is closer to the original sign's angle 
    dist1 = np.linalg.norm(minSign - signMeshRotate1.get_min_bound())
    dist2 = np.linalg.norm(minSign - signMeshRotate2.get_min_bound())
    if (dist1 < dist2):
        signMesh = signMeshRotate1
    else:
        signMesh = signMeshRotate2


    # Add the pole points to the scene
    scene, intensity, semantics, instances = combine(scene, intensity, semantics, instances, 
                                                    pole, intensityPole, semanticsPole, instancesPole)


    # Pull the points in the scene to the new sign mesh
    assetData = (sign, intensitySign, semanticsSign, instancesSign)
    sceneData = (scene, intensity, semantics, instances)
    success, newAssetData, newSceneData = pointsToMesh(signMesh, assetData, sceneData)

    sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect = newSceneData
    sign, intensitySign, semanticsSign, instancesSign = newAssetData

    # Validate that the sign has points
    if (success and np.shape(sign)[0] < 15):
        print("Sign too little points")
        success = False

    return success, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect, sign, intensitySign, semanticsSign, instancesSign, details

# --------------------------------------------------------------------------
# Scale

def scaleVehicle(asset, intensityAsset, semanticsAsset, instancesAsset, 
                scene, intensity, semantics, instances, details):


    # Prepare to create the mesh estimating normals
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.estimate_normals()
    pcdAsset.orient_normals_towards_camera_location()

    
    # Check if count of points are greater than allowed to use ball pivoting on
    if (np.shape(asset)[0] > 10000):
        print("Point count {} exceeds scale limit {}".format(np.shape(asset)[0], 10000))
        return False, None, None, None, None, None, None, None, None, None
    
    # Create a mesh using the ball pivoting method
    radii = [0.15]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))

    # o3d.visualization.draw_geometries([mesh])

    # Check that the mesh is valid
    if (np.shape(np.array(mesh.vertices))[0] < 1 or np.shape(np.array(mesh.triangles))[0] < 1):
        print("MESH NOT SUFFICENT: Vertices {} Triangles {}".format(np.shape(np.array(mesh.vertices))[0], np.shape(np.array(mesh.triangles))[0]))
        return False, None, None, None, None, None, None, None, None, None
    
    # Scale the vehicle mesh
    scale = random.uniform(1.01, 1.05)
    details["scale"] = scale
    mesh.scale(scale, center=mesh.get_center())

    # Scale the points to use later for intensity 
    scaledPoints = copy.deepcopy(pcdAsset).scale(scale, center=mesh.get_center())

    # http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html
    # Calulate intersection for scene to mesh points
    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    sceneRays = o3d.t.geometry.RaycastingScene()
    sceneRays.add_triangles(legacyMesh)

    if (np.shape(scene)[0] < 1 or np.shape(asset)[0] < 1):
        print("SCENE or ASSET PROVIDED EMPTY: SCENE {}, ASSET {}".format(np.shape(scene)[0], np.shape(asset)[0]))
        return False, None, None, None, None, None, None, None, None, None

    raysVectorsScene = []
    for point in scene:
        raysVectorsScene.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectorsScene, dtype=o3d.core.Dtype.Float32)
    ans = sceneRays.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))

    # Split between intersect and non intersect
    intensityIntersect = intensity[hit.numpy()]
    semanticsIntersect = semantics[hit.numpy()]
    instancesIntersect = instances[hit.numpy()]
    nonHit = np.logical_not(hit.numpy())
    sceneNonIntersect = scene[nonHit]
    intensityNonIntersect = intensity[nonHit]
    semanticsNonIntersect = semantics[nonHit]
    instancesNonIntersect = instances[nonHit]
    
    newAssetScene = []
    for vector in pointsOnMesh:
        newAssetScene.append(vector.numpy())

    # Calulate intersection for sign points
    raysVectorsMesh = []
    for point in asset:
        raysVectorsMesh.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectorsMesh, dtype=o3d.core.Dtype.Float32)
    ans = sceneRays.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))

    newAsset = []
    for vector in pointsOnMesh:
        newAsset.append(vector.numpy())
    
    newIntensityAsset = intensityAsset[hit.numpy()] 
    newSemanticsAsset = semanticsAsset[hit.numpy()]
    newInstancesAsset = instancesAsset[hit.numpy()]

    print(len(newAsset))
    print(len(newAssetScene))

    if len(newAsset) == 0 or len(newAssetScene) == 0:
        print("GOT NONE OF THE OG ASSET {} OR NONE OF SCENE {}".format(len(newAsset), len(newAssetScene)))
        return False, None, None, None, None, None, None, None, None, None

    # Fix the intensity of each of the points in the scene that were pulled into the asset by using the closest scaled asset point
    pcd_tree = o3d.geometry.KDTreeFlann(scaledPoints)
    for pointIndex in range(0, len(newAssetScene)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(newAssetScene[pointIndex], 1)
        intensityIntersect[pointIndex] = intensityAsset[idx]
        semanticsIntersect[pointIndex] = semanticsAsset[idx]
        instancesIntersect[pointIndex] = instancesAsset[idx]

    newAsset, intensityAsset, semanticsAsset, instancesAsset = combine(newAsset, newIntensityAsset, newSemanticsAsset, newInstancesAsset, 
                                                                    newAssetScene, intensityIntersect, semanticsIntersect, instancesIntersect)


    # print(np.shape(newAsset))
    if (np.shape(newAsset)[0] < 20):
        print("New asset too little points {}".format(np.shape(newAsset)[0]))
        return False, None, None, None, None, None, None, None, None, None


    # Return revised scene with scaled vehicle 
    return True, newAsset, intensityAsset, semanticsAsset, instancesAsset, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect, details

















