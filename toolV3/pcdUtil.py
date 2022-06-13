


import numpy as np
import open3d as o3d
import math
from sklearn.neighbors import NearestNeighbors
import sys
import random
import copy

import globals

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



# ------------------------------------------------------------------

# Asset Prechecks

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
def pointsAboveGround(points, scene, semantics):
    
    maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
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
def pointsWithinDist(points):
    
    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    dist = np.linalg.norm(centerOfPoints - centerCamPoint)

    return dist < 5

    


def removeLidarShadowLines(asset):

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

    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdCut = o3d.geometry.PointCloud()
    pcdCut.points = o3d.utility.Vector3dVector(combinedVertices)
    cutPointsHull, _ = pcdCut.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(cutPointsHull)
    hull_ls.paint_uniform_color((0, 1, 1))

    return hull_ls


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


def combineRemoveInstance(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset, assetInstance):
    pcdArrWithoutAsset = pcdArr[labelInstance != assetInstance]
    intensityWithoutAsset = intensity[labelInstance != assetInstance]
    semanticsWithoutAsset = semantics[labelInstance != assetInstance]
    labelInstanceWithoutAsset = labelInstance[labelInstance != assetInstance]

    return combine(pcdArrWithoutAsset, intensityWithoutAsset, semanticsWithoutAsset, labelInstanceWithoutAsset, 
                    pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset)


def combine(pcdArr, intensity, semantics, labelInstance, pcdArrAsset, intensityAsset, semanticsAsset, labelInstanceAsset):
    pcdArrCombined = np.vstack((pcdArr, pcdArrAsset))
    intensityCombined = np.hstack((intensity, intensityAsset))
    semanticsCombined = np.hstack((semantics, semanticsAsset))
    labelInstanceCombined = np.hstack((labelInstance, labelInstanceAsset))

    return pcdArrCombined, intensityCombined, semanticsCombined, labelInstanceCombined


# -------------------------------------------------------------


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


# -------------------------------------------------------------


def scaleV1(points, scale):
    pointsCopy = np.copy(points)

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(pointsCopy)
    hull, _ = pcdPoints.compute_convex_hull()

    pcdPoints.scale(scale, hull.get_center())

    return np.asarray(pcdPoints.points)




"""
Flip over axis
"""
def mirrorPoints(points, axis):
    if (axis != X_AXIS and axis != Y_AXIS):
        print("Axis must be 0 (X) or 1 (Y)")
        exit()
    
    # Flip
    points[:, axis] = points[:, axis] * -1

    return points



"""
Translate a group of points to a new location based on the center
"""
def translatePointsXYCenter(points, destination):
    pointsCopy = np.copy(points)

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(pointsCopy)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    addX = destination[X_AXIS] - point[X_AXIS]
    addY = destination[Y_AXIS] - point[Y_AXIS]

    pointsCopy[:, X_AXIS] = pointsCopy[:, X_AXIS] + addX
    pointsCopy[:, Y_AXIS] = pointsCopy[:, Y_AXIS] + addY

    return pointsCopy


"""
Translate a group of points to a new location x y
"""
def translatePointsXY(points, x, y):
    pointsCopy = np.copy(points)

    pointsCopy[:, X_AXIS] = pointsCopy[:, X_AXIS] + x
    pointsCopy[:, Y_AXIS] = pointsCopy[:, Y_AXIS] + y

    return pointsCopy




"""
Translate a group of points to a new location based on the center

# https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
"""
def translatePointsFromCenter(points, amount):
    pointsCopy = np.copy(points)

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(pointsCopy)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()
    
    vX = centerOfPoints[X_AXIS] - 0
    vY = centerOfPoints[Y_AXIS] - 0

    uX = vX / math.sqrt((vX * vX) + (vY * vY))
    uY = vY / math.sqrt((vX * vX) + (vY * vY))

    pointsCopy[:, X_AXIS] = pointsCopy[:, X_AXIS] + (amount * uX)
    pointsCopy[:, Y_AXIS] = pointsCopy[:, Y_AXIS] + (amount * uY)

    return pointsCopy


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



def rotatePointsLocalized(points, angle):
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

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    center = (centerOfPoints[X_AXIS], centerOfPoints[Y_AXIS])

    for point in points:
        pointXY = (point[X_AXIS], point[Y_AXIS])
        newX, newY = rotateOnePoint(center, pointXY, angle)
        point[X_AXIS] = newX
        point[Y_AXIS] = newY
    
    return points



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
def removeLidarShadow(asset, scene, intensity, semantics, labelsInstance):

    cutPointsHull = getLidarShadowMesh(asset)

    mask = checkInclusionBasedOnTriangleMesh(scene, cutPointsHull)
    mask = np.logical_not(mask)

    sceneResult = scene[mask, :]
    intensityResult = intensity[mask]
    semanticsResult = semantics[mask]
    labelsInstanceResult = labelsInstance[mask]

    return (sceneResult, intensityResult, semanticsResult, labelsInstanceResult)


# https://stackoverflow.com/questions/3306838/algorithm-for-reflecting-a-point-across-a-line
def flipPointsOverLine(a, b, points):
    pointsCopy = np.copy(points)

    x2 = a[0] # center
    y2 = a[1] 
    x3 = b[0] # flip bound
    y3 = b[1]
    m = (y3 - y2) / (x3 - x2)
    c = ((x3 * y2) - (x2 * y3)) / (x3 - x2)

    for point in pointsCopy:
        point = handleFlip(point, c, m)

    return pointsCopy

def handleFlip(point, c, m):
    x1 = point[0]
    y1 = point[1]
    d = (x1 + ((y1 - c) * m)) / (1 + (m * m))
    x4 = (2 * d) - x1
    y4 = (2 * d * m) - y1 + (2 * c)
    point[0] = x4
    point[1] = y4

    return point


# https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
# Where a = line point 1; b = line point 2; c = point to check against.
def isLeft(a, b, c):
    aX = a[0]
    bX = b[0]
    cX = c[0]
    aY = a[1]
    bY = b[1]
    cY = c[1]
    return ((bX - aX) * (cY - aY) - (bY - aY) * (cX - aX)) > 0




"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def replaceBasedOnShadow(asset, scene, intensity, semantics, labelsInstance):

    shadow = getLidarShadowMesh(asset)
    shadowVertices = np.asarray(shadow.vertices)
    shadowVertices2 = np.copy(shadowVertices)
    shadowVerticesRaised = np.copy(shadowVertices)
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(shadow)
    hull_ls.paint_uniform_color((0, 0.5, 1))


    # Check above
    shadowVerticesRaised[:, 2] = shadowVerticesRaised[:, 2] + 1
    pcdShadowRaised = o3d.geometry.PointCloud()
    pcdShadowRaised.points = o3d.utility.Vector3dVector(shadowVerticesRaised)
    hullShadowRaised, _ = pcdShadowRaised.compute_convex_hull()
    maskNonGround = (semantics != 40) | (semantics != 44) | (semantics != 48) | (semantics != 49) | (semantics != 60) | (semantics != 72)
    sceneWithoutGround = scene[maskNonGround]
    maskAbove = checkInclusionBasedOnTriangleMesh(sceneWithoutGround, hullShadowRaised)
    print("Above: ", np.sum(maskAbove))

    if (np.sum(maskAbove) > 30):
        print("TOO MANY ABOVE")
        return False, None, None, None, None,

    # Remove

    #  Get the asset's convex hull
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)

    # Find the min and max for that hull
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = sys.maxsize * -1
    maxY = sys.maxsize * -1
    for pt in np.asarray(hullVertices):
        maxX = max(maxX, pt[0])
        maxY = max(maxY, pt[1])
        minX = min(minX, pt[0])
        minY = min(minY, pt[1])

    # Get distances from the center
    minsMaxArr = np.asarray([[maxX, maxY, 0], [maxX, minY, 0], [minX, maxY, 0], [minX, minY, 0]])

    distArr = np.asarray([
        np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[1] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[2] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[3] - np.asarray([0, 0, 0]))])

    permutation = distArr.argsort()
    minsMaxArr = minsMaxArr[permutation]

    # print("maxMax box {} [{}, {}]".format(distArr[0], maxX, maxY))
    # print("maxMin box {} [{}, {}]".format(distArr[1], maxX, minY))
    # print("minMax box {} [{}, {}]".format(distArr[2], minX, maxY))
    # print("minMin box {} [{}, {}]".format(distArr[3], minX, minY))

    # print("gg box {} {}".format(minsMaxArr[1], minsMaxArr[2]))
    minBox = minsMaxArr[1]
    maxBox = minsMaxArr[2]

    # Get closest to the two bounding points selected
    assetCopy = np.copy(asset)
    assetCopy[:, 2] = 0
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(assetCopy)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(minBox, 1)
    [k, idx2, _] = pcd_tree.search_knn_vector_3d(maxBox, 1)

    print(asset[idx])
    print(asset[idx2])

    minBox = asset[idx][0]
    maxBox = asset[idx2][0]

    midPointX = (minBox[0] + maxBox[0]) / 2
    midPointY = (minBox[1] + maxBox[1]) / 2
    midPoint = np.array([midPointX, midPointY, 0])

    left = minBox
    right = maxBox
    if (isLeft(centerCamPoint, midPoint, maxBox)):
        left = maxBox
        right = minBox

    replaceLeftShadow = [midPoint]
    replaceRightShadow = [midPoint]
    for point in shadowVertices:
        if (not isLeft(centerCamPoint, midPoint, point)):
            replaceLeftShadow.append(point)
        else:
            replaceRightShadow.append(point)

    
    # leftShadow = flipPointsOverLine(centerCamPoint, left, shadowVertices)
    # rightShadow = flipPointsOverLine(centerCamPoint, right, shadowVertices2)
    midLeft = flipPointsOverLine(centerCamPoint, left, [midPoint])[0]
    midRight = flipPointsOverLine(centerCamPoint, right, [midPoint])[0]

    midLeftPointX = (left[0] + midPoint[0]) / 2
    midLeftPointY = (left[1] + midPoint[1]) / 2
    midLeftPoint = np.array([midPointX, midPointY, 0])

    # lx = midLeft[0] - left[0]
    # ly = midLeft[1] - left[1]
    # rx = midRight[0] - right[0]
    # ry = midRight[1] - right[1]
    # replaceLeftShadow = translatePointsXY(replaceLeftShadow, lx, ly)
    # replaceRightShadow = translatePointsXY(replaceRightShadow, rx, ry)

    # print(midLeft)
    # print(left)
    # print(midPoint)
    # print(right)
    # print(midRight)

    angleLeft = getAngleRadians(centerCamPoint, left, midPoint)
    angleRight = getAngleRadians(centerCamPoint, midPoint, right)
    angleLeft = angleLeft * (180 / math.pi)
    angleRight = angleRight * (180 / math.pi)
    print(angleLeft)
    print(angleRight)
    if angleLeft < 0:
        angleLeft = angleLeft * -1
    if angleRight < 0:
        angleRight = angleRight * -1
    angleRight = 360 - angleRight

    if (len(replaceLeftShadow) > 0):
        replaceLeftShadow = rotatePoints(replaceLeftShadow, angleRight)
    else:
        print("left shadow empty")
    
    if (len(replaceRightShadow) > 0):
        replaceRightShadow = rotatePoints(replaceRightShadow, angleLeft)
    else:
        print("right shadow empty")

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(replaceLeftShadow)
    shadowRotated, _ = pcdCastHull.compute_convex_hull()

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(replaceRightShadow)
    shadowRotated2, _ = pcdCastHull.compute_convex_hull()

    # Copy items
    maskIncluded = checkInclusionBasedOnTriangleMesh(scene, shadowRotated)

    pcdIncluded = scene[maskIncluded]
    intensityIncluded = intensity[maskIncluded]
    semanticsIncluded = semantics[maskIncluded]
    labelsInstanceIncluded = labelsInstance[maskIncluded]

    maskIncluded2 = checkInclusionBasedOnTriangleMesh(scene, shadowRotated2)
    pcdIncluded2 = scene[maskIncluded2]
    intensityIncluded2 = intensity[maskIncluded2]
    semanticsIncluded2 = semantics[maskIncluded2]
    labelsInstanceIncluded2 = labelsInstance[maskIncluded2]



    semSetInval = set()
    # Print sem data
    for sem in semanticsIncluded:
        if (sem in globals.instancesVehicle.keys()
            or sem in globals.instancesWalls.keys()):
            semSetInval.add(sem)
    for sem in semanticsIncluded2:
        if (sem in globals.instancesVehicle.keys()
            or sem in globals.instancesWalls.keys()):
            semSetInval.add(sem)
    for sem in semSetInval:
        print(globals.name_label_mapping[sem])


    # pcdIncluded = flipPointsOverLine(centerCamPoint, left, pcdIncluded)
    # pcdIncluded2 = flipPointsOverLine(centerCamPoint, right, pcdIncluded2)

    if (len(pcdIncluded) > 0):
        pcdIncluded = rotatePoints(pcdIncluded, angleLeft)
    else:
        print("left points empty")
    if (len(pcdIncluded2) > 0):
        pcdIncluded2 = rotatePoints(pcdIncluded2, angleRight)
    else:
        print("right points empty")


    pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded = combine(pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded,
        pcdIncluded2, intensityIncluded2, semanticsIncluded2, labelsInstanceIncluded2)


    # maskGround = (semanticsIncluded == 40) | (semanticsIncluded == 44) | (semanticsIncluded == 48) | (semanticsIncluded == 49) | (semanticsIncluded == 60) | (semanticsIncluded == 72)
    # maskGround2 = (semanticsIncluded2 == 40) | (semanticsIncluded2 == 44) | (semanticsIncluded2 == 48) | (semanticsIncluded2 == 49) | (semanticsIncluded2 == 60) | (semanticsIncluded2 == 72)

    # print(np.sum(maskGround))
    # print(np.sum(maskGround2))

    # if np.sum(maskGround) < np.sum(maskGround2):
    #     pcdIncluded = scene[maskIncluded2]
    #     intensityIncluded = intensity[maskIncluded2]
    #     semanticsIncluded = semantics[maskIncluded2]
    #     labelsInstanceIncluded = labelsInstance[maskIncluded2]
    #     m = mMax
    #     c = cMax

    # print(np.shape(pcdIncluded))

    # for point in pcdIncluded:
    #     x1 = point[X_AXIS]
    #     y1 = point[Y_AXIS]
    #     d = (x1 + ((y1 - c) * m)) / (1 + (m * m))
    #     x4 = (2 * d) - x1
    #     y4 = (2 * d * m) - y1 + (2 * c)
    #     point[X_AXIS] = x4
    #     point[Y_AXIS] = y4

    hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated)
    hull_ls2.paint_uniform_color((0, 1, 0.5))
    hull_ls22 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated2)
    hull_ls22.paint_uniform_color((1, 0, 0.5))
    hull_ls44 = o3d.geometry.LineSet.create_from_triangle_mesh(hullShadowRaised)
    hull_ls44.paint_uniform_color((0, 0.2, 1))

    pcdNewAddition = o3d.geometry.PointCloud()
    pcdNewAddition.points = o3d.utility.Vector3dVector(pcdIncluded)
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(scene)


    pcdCast2 = o3d.geometry.PointCloud()
    pcdCast2.points = o3d.utility.Vector3dVector(np.asarray([minBox, maxBox]))
    pcdCast2.paint_uniform_color((.6, 0, .6))

    # o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, hull_ls44, pcdNewAddition, pcdScene, pcdCast2])
    # o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, pcdNewAddition, pcdScene, pcdCast2])
    # o3d.visualization.draw_geometries([hull, pcdCastHull22, pcdCast2])
    


    sceneReplace, intensityReplace, semanticsReplace, labelsInstanceReplace = combine(pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded,
        scene, intensity, semantics, labelsInstance)



    return True, sceneReplace, intensityReplace, semanticsReplace, labelsInstanceReplace


"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def replaceBasedOnShadowV3(asset, scene, intensity, semantics, labelsInstance):

    shadow = getLidarShadowMesh(asset)
    shadowVertices = np.asarray(shadow.vertices)
    shadowVertices2 = np.copy(shadowVertices)
    shadowVerticesRaised = np.copy(shadowVertices)
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(shadow)
    hull_ls.paint_uniform_color((0, 0.5, 1))


    # Check above
    shadowVerticesRaised[:, 2] = shadowVerticesRaised[:, 2] + 1
    pcdShadowRaised = o3d.geometry.PointCloud()
    pcdShadowRaised.points = o3d.utility.Vector3dVector(shadowVerticesRaised)
    hullShadowRaised, _ = pcdShadowRaised.compute_convex_hull()
    maskNonGround = (semantics != 40) | (semantics != 44) | (semantics != 48) | (semantics != 49) | (semantics != 60) | (semantics != 72)
    sceneWithoutGround = scene[maskNonGround]
    maskAbove = checkInclusionBasedOnTriangleMesh(sceneWithoutGround, hullShadowRaised)
    print("Above: ", np.sum(maskAbove))

    # Remove

    #  Get the asset's convex hull
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)

    # Find the min and max for that hull
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = sys.maxsize * -1
    maxY = sys.maxsize * -1
    for pt in np.asarray(hullVertices):
        maxX = max(maxX, pt[0])
        maxY = max(maxY, pt[1])
        minX = min(minX, pt[0])
        minY = min(minY, pt[1])

    # Get distances from the center
    minsMaxArr = np.asarray([[maxX, maxY, 0], [maxX, minY, 0], [minX, maxY, 0], [minX, minY, 0]])

    distArr = np.asarray([
        np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[1] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[2] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[3] - np.asarray([0, 0, 0]))])

    permutation = distArr.argsort()
    minsMaxArr = minsMaxArr[permutation]

    # print("maxMax box {} [{}, {}]".format(distArr[0], maxX, maxY))
    # print("maxMin box {} [{}, {}]".format(distArr[1], maxX, minY))
    # print("minMax box {} [{}, {}]".format(distArr[2], minX, maxY))
    # print("minMin box {} [{}, {}]".format(distArr[3], minX, minY))

    # print("gg box {} {}".format(minsMaxArr[1], minsMaxArr[2]))
    minBox = minsMaxArr[1]
    maxBox = minsMaxArr[2]

    # Get closest to the two bounding points selected
    assetCopy = np.copy(asset)
    assetCopy[:, 2] = 0
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(assetCopy)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(minBox, 1)
    [k, idx2, _] = pcd_tree.search_knn_vector_3d(maxBox, 1)

    print(asset[idx])
    print(asset[idx2])

    minBox = asset[idx][0]
    maxBox = asset[idx2][0]

    midPointX = (minBox[0] + maxBox[0]) / 2
    midPointY = (minBox[1] + maxBox[1]) / 2
    midPoint = np.array([midPointX, midPointY, 0])

    left = minBox
    right = maxBox
    if (isLeft(centerCamPoint, midPoint, maxBox)):
        left = maxBox
        right = minBox
    
    leftShadow = flipPointsOverLine(centerCamPoint, left, shadowVertices)
    rightShadow = flipPointsOverLine(centerCamPoint, right, shadowVertices2)
    midLeft = flipPointsOverLine(centerCamPoint, left, [midPoint])[0]
    midRight = flipPointsOverLine(centerCamPoint, right, [midPoint])[0]

    # print(midLeft)
    # print(left)
    # print(midPoint)
    # print(right)
    # print(midRight)

    replaceLeftShadow = []
    for point in leftShadow:
        if (not isLeft(centerCamPoint, midLeft, point)):
            replaceLeftShadow.append(point)
    
    replaceRightShadow = []
    for point in rightShadow:
        if (isLeft(centerCamPoint, midRight, point)):
            replaceRightShadow.append(point)

    
    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(replaceLeftShadow)
    shadowRotated, _ = pcdCastHull.compute_convex_hull()

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(replaceRightShadow)
    shadowRotated2, _ = pcdCastHull.compute_convex_hull()

    maskIncluded = checkInclusionBasedOnTriangleMesh(scene, shadowRotated)

    pcdIncluded = scene[maskIncluded]
    intensityIncluded = intensity[maskIncluded]
    semanticsIncluded = semantics[maskIncluded]
    labelsInstanceIncluded = labelsInstance[maskIncluded]

    maskIncluded2 = checkInclusionBasedOnTriangleMesh(scene, shadowRotated2)
    pcdIncluded2 = scene[maskIncluded2]
    intensityIncluded2 = intensity[maskIncluded2]
    semanticsIncluded2 = semantics[maskIncluded2]
    labelsInstanceIncluded2 = labelsInstance[maskIncluded2]

    pcdIncluded = flipPointsOverLine(centerCamPoint, left, pcdIncluded)
    pcdIncluded2 = flipPointsOverLine(centerCamPoint, right, pcdIncluded2)


    pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded = combine(pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded,
        pcdIncluded2, intensityIncluded2, semanticsIncluded2, labelsInstanceIncluded2)


    # maskGround = (semanticsIncluded == 40) | (semanticsIncluded == 44) | (semanticsIncluded == 48) | (semanticsIncluded == 49) | (semanticsIncluded == 60) | (semanticsIncluded == 72)
    # maskGround2 = (semanticsIncluded2 == 40) | (semanticsIncluded2 == 44) | (semanticsIncluded2 == 48) | (semanticsIncluded2 == 49) | (semanticsIncluded2 == 60) | (semanticsIncluded2 == 72)

    # print(np.sum(maskGround))
    # print(np.sum(maskGround2))

    # if np.sum(maskGround) < np.sum(maskGround2):
    #     pcdIncluded = scene[maskIncluded2]
    #     intensityIncluded = intensity[maskIncluded2]
    #     semanticsIncluded = semantics[maskIncluded2]
    #     labelsInstanceIncluded = labelsInstance[maskIncluded2]
    #     m = mMax
    #     c = cMax

    # print(np.shape(pcdIncluded))

    # for point in pcdIncluded:
    #     x1 = point[X_AXIS]
    #     y1 = point[Y_AXIS]
    #     d = (x1 + ((y1 - c) * m)) / (1 + (m * m))
    #     x4 = (2 * d) - x1
    #     y4 = (2 * d * m) - y1 + (2 * c)
    #     point[X_AXIS] = x4
    #     point[Y_AXIS] = y4

    hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated)
    hull_ls2.paint_uniform_color((0, 1, 0.5))
    hull_ls22 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated2)
    hull_ls22.paint_uniform_color((1, 0, 0.5))
    hull_ls44 = o3d.geometry.LineSet.create_from_triangle_mesh(hullShadowRaised)
    hull_ls44.paint_uniform_color((0, 0.2, 1))

    pcdNewAddition = o3d.geometry.PointCloud()
    pcdNewAddition.points = o3d.utility.Vector3dVector(pcdIncluded)
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(scene)


    pcdCast2 = o3d.geometry.PointCloud()
    pcdCast2.points = o3d.utility.Vector3dVector(np.asarray([minBox, maxBox]))
    pcdCast2.paint_uniform_color((.6, 0, .6))

    # o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, hull_ls44, pcdNewAddition, pcdScene, pcdCast2])
    # o3d.visualization.draw_geometries([hull, pcdCastHull22, pcdCast2])
    

    return combine(pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded,
        scene, intensity, semantics, labelsInstance)





"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def replaceBasedOnShadowV1(asset, scene, intensity, semantics, labelsInstance):

    shadow = getLidarShadowMesh(asset)
    shadowVertices = np.asarray(shadow.vertices)
    shadowVertices2 = np.copy(shadowVertices)
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(shadow)
    hull_ls.paint_uniform_color((0, 0.5, 1))

    #  Get the asset's bounding box
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    obb = pcdAsset.get_oriented_bounding_box()

    # draw a line through the min bound
    # minBox = obb.get_min_bound()
    # maxBox = obb.get_max_bound()

    # print("min box", minBox)
    # print("max box", maxBox)

    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)

    minX = sys.maxsize
    minY = sys.maxsize
    maxX = sys.maxsize * -1
    maxY = sys.maxsize * -1
    # for pt in np.asarray(obb.get_box_points()):
    for pt in np.asarray(hullVertices):
        print(pt)
        maxX = max(maxX, pt[0])
        maxY = max(maxY, pt[1])
        minX = min(minX, pt[0])
        minY = min(minY, pt[1])

    # dist = np.linalg.norm(centerOfPoints - centerCamPoint)

    minsMaxArr = np.asarray([[maxX, maxY, 0], [maxX, minY, 0], [minX, maxY, 0], [minX, minY, 0]])

    distArr = np.asarray([
        np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[1] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[2] - np.asarray([0, 0, 0])),
        np.linalg.norm(minsMaxArr[3] - np.asarray([0, 0, 0]))])

    permutation = distArr.argsort()
    minsMaxArr = minsMaxArr[permutation]
    
    # maxMax = np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0]))
    # maxMin = np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0]))
    # minMax = np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0]))
    # minMin = np.linalg.norm(minsMaxArr[0] - np.asarray([0, 0, 0]))

    print("maxMax box {} [{}, {}]".format(distArr[0], maxX, maxY))
    print("maxMin box {} [{}, {}]".format(distArr[1], maxX, minY))
    print("minMax box {} [{}, {}]".format(distArr[2], minX, maxY))
    print("minMin box {} [{}, {}]".format(distArr[3], minX, minY))

    print("gg box {} {}".format(minsMaxArr[1], minsMaxArr[2]))
    minBox = minsMaxArr[1]
    maxBox = minsMaxArr[2]


    assetCopy = np.copy(asset)
    assetCopy[:, 2] = 0
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(assetCopy)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(minBox, 1)
    [k, idx2, _] = pcd_tree.search_knn_vector_3d(maxBox, 1)

    print(asset[idx])
    print(asset[idx2])

    minBox = asset[idx][0]
    maxBox = asset[idx2][0]

    # https://stackoverflow.com/questions/3306838/algorithm-for-reflecting-a-point-across-a-line
    x2 = 0 # center
    y2 = 0 
    x3 = minBox[X_AXIS] # flip bound
    y3 = minBox[Y_AXIS]
    m = (y3 - y2) / (x3 - x2)
    c = ((x3 * y2) - (x2 * y3)) / (x3 - x2)

    for point in shadowVertices:
        x1 = point[X_AXIS]
        y1 = point[Y_AXIS]
        d = (x1 + ((y1 - c) * m)) / (1 + (m * m))
        x4 = (2 * d) - x1
        y4 = (2 * d * m) - y1 + (2 * c)
        point[X_AXIS] = x4
        point[Y_AXIS] = y4

    # Flip over Max
    x3Max = maxBox[X_AXIS] # flip bound
    y3Max = maxBox[Y_AXIS]
    mMax = (y3Max - y2) / (x3Max - x2)
    cMax = ((x3Max * y2) - (x2 * y3Max)) / (x3Max - x2)

    for point in shadowVertices2:
        x1 = point[X_AXIS]
        y1 = point[Y_AXIS]
        d = (x1 + ((y1 - cMax) * mMax)) / (1 + (mMax * mMax))
        x4 = (2 * d) - x1
        y4 = (2 * d * mMax) - y1 + (2 * cMax)
        point[X_AXIS] = x4
        point[Y_AXIS] = y4
    
    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(shadowVertices)
    shadowRotated, _ = pcdCastHull.compute_convex_hull()

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(shadowVertices2)
    shadowRotated2, _ = pcdCastHull.compute_convex_hull()

    maskIncluded = checkInclusionBasedOnTriangleMesh(scene, shadowRotated)
    maskNotIncluded = np.logical_not(maskIncluded)

    pcdIncluded = scene[maskIncluded]
    intensityIncluded = intensity[maskIncluded]
    semanticsIncluded = semantics[maskIncluded]
    labelsInstanceIncluded = labelsInstance[maskIncluded]

    maskIncluded2 = checkInclusionBasedOnTriangleMesh(scene, shadowRotated2)
    semanticsIncluded2 = semantics[maskIncluded2]

    maskGround = (semanticsIncluded == 40) | (semanticsIncluded == 44) | (semanticsIncluded == 48) | (semanticsIncluded == 49) | (semanticsIncluded == 60) | (semanticsIncluded == 72)
    maskGround2 = (semanticsIncluded2 == 40) | (semanticsIncluded2 == 44) | (semanticsIncluded2 == 48) | (semanticsIncluded2 == 49) | (semanticsIncluded2 == 60) | (semanticsIncluded2 == 72)

    print(np.sum(maskGround))
    print(np.sum(maskGround2))

    if np.sum(maskGround) < np.sum(maskGround2):
        pcdIncluded = scene[maskIncluded2]
        intensityIncluded = intensity[maskIncluded2]
        semanticsIncluded = semantics[maskIncluded2]
        labelsInstanceIncluded = labelsInstance[maskIncluded2]
        m = mMax
        c = cMax

    print(np.shape(pcdIncluded))

    for point in pcdIncluded:
        x1 = point[X_AXIS]
        y1 = point[Y_AXIS]
        d = (x1 + ((y1 - c) * m)) / (1 + (m * m))
        x4 = (2 * d) - x1
        y4 = (2 * d * m) - y1 + (2 * c)
        point[X_AXIS] = x4
        point[Y_AXIS] = y4

    hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated)
    hull_ls2.paint_uniform_color((0, 1, 0.5))
    hull_ls22 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated2)
    hull_ls22.paint_uniform_color((1, 0, 0.5))
    pcdCastHull22 = o3d.geometry.PointCloud()
    pcdCastHull22.points = o3d.utility.Vector3dVector(pcdIncluded)
    pcdCastHull23 = o3d.geometry.PointCloud()
    pcdCastHull23.points = o3d.utility.Vector3dVector(scene)


    pcdCast2 = o3d.geometry.PointCloud()
    pcdCast2.points = o3d.utility.Vector3dVector(np.asarray([minBox, maxBox]))
    pcdCast2.paint_uniform_color((.6, 0, .6))

    o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, pcdCastHull22, pcdCastHull23, obb, pcdCast2])
    # o3d.visualization.draw_geometries([hull, pcdCastHull22, pcdCast2])
    

    return combine(pcdIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded,
        scene, intensity, semantics, labelsInstance)


def removeCenterPoints(points):

    pointsVector = o3d.utility.Vector3dVector(points)

    pcdCenter = o3d.geometry.PointCloud()
    pcdCenter.points = o3d.utility.Vector3dVector(centerArea)

    #  Get the asset's bounding box
    centerBox = pcdCenter.get_oriented_bounding_box()
    ignoreIndex = centerBox.get_point_indices_within_bounding_box(pointsVector)
    
    mask = np.ones(np.shape(points)[0], dtype=int)
    mask[ignoreIndex] = 0

    pointsWithoutCenter = points[mask == 1]
    return pointsWithoutCenter


# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
# https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
def pointInTriangle (p0, p1, p2, point):
    p0x, p0y = p0
    p1x, p1y = p1
    p2x, p2y = p2
    px, py = point
    Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
    s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
    t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)

    return s > 0 and t > 0 and ((1 - s) - t) > 0


def getValidRotations(points, scene, semantics):

    # Remove the road
    maskNotGround = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    pcdArrExceptGround = scene[maskNotGround, :]

    # Ignore anything in center
    pcdArrExceptGround[:, Z_AXIS] = 0
    sceneWithoutCenter = removeCenterPoints(pcdArrExceptGround)
    
    # Create a set of unique points
    uniquePoints = set()
    for point in sceneWithoutCenter:
        pt = (point[X_AXIS], point[Y_AXIS])
        uniquePoints.add(pt)

    #  Get the asset's bounding box
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(points)
    obb = pcdAsset.get_oriented_bounding_box()

    # Create a triangle from the oriented box    
    minBox = obb.get_min_bound()
    maxBox = obb.get_max_bound()
    triangle = np.asarray([centerCamPoint, maxBox, minBox])
    
    degrees = []

    for deg in range(0, 360, 5):
        triangleRotated = rotatePoints(triangle, deg)

        p0 = (triangleRotated[0][X_AXIS], triangleRotated[0][Y_AXIS])
        p1 = (triangleRotated[1][X_AXIS], triangleRotated[1][Y_AXIS])
        p2 = (triangleRotated[2][X_AXIS], triangleRotated[2][Y_AXIS])

        empty = True
        for point in uniquePoints:
            # print(point)
            empty = empty and not pointInTriangle(p0, p1, p2, point)
        
        if (empty): 
            degrees.append(deg)
            
    return degrees




# https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian
def nearestNeighbors(values, nbr_neighbors):

    zeroCol = np.zeros((np.shape(values)[0],), dtype=bool)
    valuesResized = np.c_[values, zeroCol]
    # np.append(np.array(values), np.array(zeroCol), axis=1)
    # valuesResized = np.hstack((np.array(values), np.array(zeroCol)))

    nn = NearestNeighbors(n_neighbors=nbr_neighbors, metric='cosine', algorithm='brute').fit(valuesResized)
    dists, idxs = nn.kneighbors(valuesResized)

    return dists



# https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
def getAngleRadians(p0, p1, p2):
    p0x = p0[0] 
    p0y = p0[1]
    p1x = p1[0] 
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]

    result = math.atan2(p2y - p0y, p2x - p0x) - math.atan2(p1y - p0y, p1x - p0x)


    # degreesResult = result * (180 / math.pi)

    return result
    # return degreesResult

    # a = np.asarray(p0x - p1x, p0y - p1y)
    # b = np.asarray(p1x - p2x, p0y - p2y)

    # return np.arccos((a * b) / (np.absolute(a) * np.absolute(b)))








def alignZdim(asset, scene, semantics):
    assetCopy = np.copy(asset)

    maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
    pcdArrGround = scene[maskGround, :]

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    aabb = pcdAsset.get_axis_aligned_bounding_box()
    aabb.color = (0, 0, 1)
    boxPoints = np.asarray(aabb.get_box_points())

    boxMinZ = np.min(boxPoints.T[2])

    bP1 = boxPoints[boxPoints[:, 2] == boxMinZ]
    bP2 = boxPoints[boxPoints[:, 2] != boxMinZ]
    bP1[:, 2] = bP1[:, 2] + 20
    bP2[:, 2] = bP2[:, 2] - 20

    boxPointsZLarger = np.vstack((bP1, bP2))

    largerBox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(boxPointsZLarger))
    largerBox.color = (1, 0, 1)

    mask = largerBox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pcdArrGround))
    onlyByCar = pcdArrGround[mask]

    groundAvg = 0
    if (np.shape(onlyByCar)[0] != 0):
        groundAvg = np.average(onlyByCar.T[2])
        print("ground")
    else:
        groundAvg = np.average(pcdArrGround.T[2])
        print("avg")

    boxMinZ = round(boxMinZ, 2)
    groundAvg = round(groundAvg, 2)

    print("curr min", boxMinZ)
    print("ground min", groundAvg)

    change = groundAvg - boxMinZ

    assetCopy[:, 2] = assetCopy[:, 2] + change

    return assetCopy





# Get closest to the two bounding points selected
def closestBoundingTwo(min, max, asset):

    # Remove Z dim
    assetCopy = np.copy(asset)
    assetCopy[:, 2] = 0

    # Find closest
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(assetCopy)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(min, 1)
    [k, idx2, _] = pcd_tree.search_knn_vector_3d(max, 1)

    return asset[idx][0], asset[idx2][0]






def createStopSign(center):
    

    box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.75, depth=0.30)
    box2 = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.30, depth=0.75)

    box.translate(center, relative=False)
    box2.translate(center, relative=False)

    signVertices = np.vstack((np.array(box.vertices), np.array(box2.vertices)))

    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(signVertices)

    #  Get the asset's hull mesh
    sign, _ = pcdSign.compute_convex_hull()

    return sign



def getSignAsset(scene, intensity, semantics, labelsInstance):
    
    maskSign = (semantics == 81)

    onlySigns = scene[maskSign, :]
    intensitySigns = intensity[maskSign]
    semanticsSigns = semantics[maskSign]
    labelsInstanceSigns = labelsInstance[maskSign]

    # Check that there are signs 
    if (np.shape(onlySigns)[0] < 1):
        print("NO SIGNS FOUND")
        return False, None, None, None, None

    pcdSigns = o3d.geometry.PointCloud()
    pcdSigns.points = o3d.utility.Vector3dVector(onlySigns)

    labels = np.array(pcdSigns.cluster_dbscan(eps=2, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    if (max_label < 0):
        print("NO SIGNS FOUND")
        return False, None, None, None, None

    signIndex = random.randint(0, max_label)
    print(labels)
    print(signIndex)

    oneSign = onlySigns[labels == signIndex, :]
    intensitySign = intensitySigns[labels == signIndex]
    semanticsSign = semanticsSigns[labels == signIndex]
    labelsInstanceSign= labelsInstanceSigns[labels == signIndex]

    return True, oneSign, intensitySign, semanticsSign, labelsInstanceSign


def pointsToMesh(mesh, asset, intensityAsset, semanticsAsset, labelsInstanceAsset, 
                scene, intensity, semantics, labelsInstance):
    # http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html
    # Calulate intersection for scene to mesh points
    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    sceneRays = o3d.t.geometry.RaycastingScene()
    sceneRays.add_triangles(legacyMesh)

    if (np.shape(scene)[0] < 1 or np.shape(asset)[0] < 1):
        print("SCENE or ASSET PROVIDED EMPTY: SCENE {}, ASSET {}".format(np.shape(scene)[0], np.shape(asset)[0]))
        return False, None, None, None, None, None, None, None, None

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
    labelsInstanceIntersect = labelsInstance[hit.numpy()]
    nonHit = np.logical_not(hit.numpy())
    sceneNonIntersect = scene[nonHit]
    intensityNonIntersect = intensity[nonHit]
    semanticsNonIntersect = semantics[nonHit]
    labelsInstanceNonIntersect = labelsInstance[nonHit]
    
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
    labelsInstanceAsset = labelsInstanceAsset[hit.numpy()]

    print(len(newAsset))
    print(len(newAssetScene))

    if len(newAsset) == 0 or len(newAssetScene) == 0:
        print("GOT NONE OF THE OG SIGN {} OR NONE OF SCENE {}".format(len(newAsset), len(newAssetScene)))
        return False, None, None, None, None, None, None, None, None

    # Fix the intensity of each of the points in the scene that were pulled into the sign by using the closest sign point
    pcdAssetNearest = o3d.geometry.PointCloud()
    pcdAssetNearest.points = o3d.utility.Vector3dVector(newAsset)
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAssetNearest)
    for pointIndex in range(0, len(newAssetScene)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        intensityIntersect[pointIndex] = intensityAsset[idx]
        semanticsIntersect[pointIndex] = semanticsAsset[idx]
        labelsInstanceIntersect[pointIndex] = labelsInstanceAsset[idx]

    newAsset, intensityAsset, semanticsAsset, labelsInstanceAsset = combine(newAsset, intensityAsset, semanticsAsset, labelsInstanceAsset, 
                                                                    newAssetScene, intensityIntersect, semanticsIntersect, labelsInstanceIntersect)

    return True, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, labelsInstanceNonIntersect, newAsset, intensityAsset, semanticsAsset, labelsInstanceAsset


def signReplace(sign, intensitySign, semanticsSign, labelsInstanceSign, scene, intensity, semantics, labelsInstance, details):
    
    # Get bounds to align the sign to 
    print(np.shape(sign))
    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(sign)
    obb = pcdSign.get_oriented_bounding_box()
    signCenter = obb.get_center()
    minSign, maxSign = closestBoundingTwo(obb.get_min_bound(), obb.get_max_bound(), sign)

    # Create shape mesh
    signMesh = None
    signType = random.randint(0, 1)
    if (signType == 0):
        signMesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.6, depth=0.75)
        details["sign"] = "speed"
    else:
        signMesh = createStopSign(signCenter)
        details["sign"] = "stop"

    # Move the mesh to the 
    signMesh.translate(signCenter, relative=False)
    angleSign = getAngleRadians(obb.get_center(), minSign, signMesh.get_min_bound())

    # Get two rotation options
    rotation = signMesh.get_rotation_matrix_from_xyz((0, 0, angleSign * -1))
    rotation2 = signMesh.get_rotation_matrix_from_xyz((0, 0, angleSign))
    signMeshRotate1 = copy.deepcopy(signMesh)
    signMeshRotate2 = copy.deepcopy(signMesh)
    signMeshRotate1.rotate(rotation, center=signMeshRotate1.get_center())
    signMeshRotate2.rotate(rotation2, center=signMeshRotate2.get_center())

    # Get the rotation that is closer to the sign
    dist1 = np.linalg.norm(minSign - signMeshRotate1.get_min_bound())
    dist2 = np.linalg.norm(minSign - signMeshRotate2.get_min_bound())
    if (dist1 < dist2):
        signMesh = signMeshRotate1
    else:
        signMesh = signMeshRotate2

    success, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, labelsInstanceNonIntersect, sign, intensitySign, semanticsSign, labelsInstanceSign = pointsToMesh(signMesh, sign, intensitySign, semanticsSign, labelsInstanceSign, 
                                                                                                                                                                    scene, intensity, semantics, labelsInstance)

    return success, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, labelsInstanceNonIntersect, sign, intensitySign, semanticsSign, labelsInstanceSign, details


def scaleVehicle(asset, intensityAsset, semanticsAsset, labelsInstanceAsset, 
                scene, intensity, semantics, labelsInstance, details):


    # Prepare to create the mesh estimating normals
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.estimate_normals()
    pcdAsset.orient_normals_towards_camera_location()

    # Create a mesh using the ball pivoting method
    radii = [0.15, 0.15, 0.15, 0.15]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))

    # Check that the mesh is valid
    if (np.shape(np.array(mesh.vertices))[0] < 1 or np.shape(np.array(mesh.triangles))[0] < 1):
        print("MESH NOT SUFFICENT: Vertices {} Triangles {}".format(np.shape(np.array(mesh.vertices))[0], np.shape(np.array(mesh.triangles))[0]))
        return False, None, None, None, None, None, None, None, None, None
    
    # Scale the vehicle
    mesh.scale(1.2, center=mesh.get_center())
    details["scale"] = 1.2

    # Prepare the shadow of the new mesh to see what is included
    shadow = getLidarShadowMesh(np.array(mesh.vertices))
    sceneMask = checkInclusionBasedOnTriangleMesh(scene, shadow)
    
    # Included in shadow
    sceneIncluded = scene[sceneMask]
    intensityIncluded = intensity[sceneMask]
    semanticsIncluded = semantics[sceneMask]
    labelsInstanceIncluded = labelsInstance[sceneMask]

    # Not Included in the shadow
    sceneMaskNot = np.logical_not(sceneMask)
    sceneNotIncluded = scene[sceneMaskNot]
    intensityNotIncluded = intensity[sceneMaskNot]
    semanticsNotIncluded = semantics[sceneMaskNot]
    labelsInstanceNotIncluded = labelsInstance[sceneMaskNot]

    

    # Prepare the mesh for ray casting to move points to mesh
    success, _, _, _, _, newAsset, newIntensityAsset, newSemanticsAsset, newLabelsInstanceAsset = pointsToMesh(mesh, asset, intensityAsset, semanticsAsset, labelsInstanceAsset, 
                                                                                                        sceneIncluded, intensityIncluded, semanticsIncluded, labelsInstanceIncluded)

    # newAsset = alignZdim(newAsset, sceneNotIncluded, semanticsNotIncluded)

    # print(np.shape(newAsset))
    if (success and np.shape(newAsset)[0] < 10):
        success = False

    return success, sceneNotIncluded, intensityNotIncluded, semanticsNotIncluded, labelsInstanceNotIncluded, newAsset, newIntensityAsset, newSemanticsAsset, newLabelsInstanceAsset, details




































