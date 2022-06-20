

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


def intensityChange(intensityAsset, type, details):

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
            success = pointsAboveGround(pcdArrAssetNew, pcdArr, maskGround) or pointsWithinDist(pcdArrAssetNew) 

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
def pointsWithinDist(points):
    
    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    dist = np.linalg.norm(centerOfPoints - centerCamPoint)

    return dist < 5


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
def removeLidarShadow(asset, scene, intensity, semantics, instances):

    cutPointsHull = getLidarShadowMesh(asset)

    mask = checkInclusionBasedOnTriangleMesh(scene, cutPointsHull)
    mask = np.logical_not(mask)

    sceneResult = scene[mask, :]
    intensityResult = intensity[mask]
    semanticsResult = semantics[mask]
    instancesResult = instances[mask]

    return (sceneResult, intensityResult, semanticsResult, instancesResult)


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
def replaceBasedOnShadow(asset, scene, intensity, semantics, instances):

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

    if (len(replaceLeftShadow) > 4):
        replaceLeftShadow = rotatePoints(replaceLeftShadow, angleRight)
    else:
        print("Left shadow empty {}".format(len(replaceLeftShadow)))
        return False, None, None, None, None
    
    if (len(replaceRightShadow) > 4):
        replaceRightShadow = rotatePoints(replaceRightShadow, angleLeft)
    else:
        print("Right shadow empty {}".format(len(replaceRightShadow)))
        return False, None, None, None, None

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
    instancesIncluded = instances[maskIncluded]

    maskIncluded2 = checkInclusionBasedOnTriangleMesh(scene, shadowRotated2)
    pcdIncluded2 = scene[maskIncluded2]
    intensityIncluded2 = intensity[maskIncluded2]
    semanticsIncluded2 = semantics[maskIncluded2]
    instancesIncluded2 = instances[maskIncluded2]



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


    pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded = combine(pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded,
        pcdIncluded2, intensityIncluded2, semanticsIncluded2, instancesIncluded2)


    # maskGround = (semanticsIncluded == 40) | (semanticsIncluded == 44) | (semanticsIncluded == 48) | (semanticsIncluded == 49) | (semanticsIncluded == 60) | (semanticsIncluded == 72)
    # maskGround2 = (semanticsIncluded2 == 40) | (semanticsIncluded2 == 44) | (semanticsIncluded2 == 48) | (semanticsIncluded2 == 49) | (semanticsIncluded2 == 60) | (semanticsIncluded2 == 72)

    # print(np.sum(maskGround))
    # print(np.sum(maskGround2))

    # if np.sum(maskGround) < np.sum(maskGround2):
    #     pcdIncluded = scene[maskIncluded2]
    #     intensityIncluded = intensity[maskIncluded2]
    #     semanticsIncluded = semantics[maskIncluded2]
    #     instancesIncluded = instances[maskIncluded2]
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
    


    sceneReplace, intensityReplace, semanticsReplace, instancesReplace = combine(pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded,
        scene, intensity, semantics, instances)



    return True, sceneReplace, intensityReplace, semanticsReplace, instancesReplace


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


# def getValidRotations(points, scene, semantics):

#     # Remove the road
#     # maskNotGround = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
#     maskNotGround = (semantics != 0) & (semantics != 1) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
#     pcdArrExceptGround = scene[maskNotGround, :]

#     # Ignore anything in center
#     pcdArrExceptGround[:, Z_AXIS] = 0
#     # sceneWithoutCenter = removeCenterPoints(pcdArrExceptGround)
    
#     # Create a set of unique points
#     uniquePoints = set()
#     # for point in sceneWithoutCenter:
#     for point in pcdArrExceptGround:
#         pt = (point[X_AXIS], point[Y_AXIS])
#         uniquePoints.add(pt)

#     #  Get the asset's bounding box
#     pcdAsset = o3d.geometry.PointCloud()
#     pcdAsset.points = o3d.utility.Vector3dVector(points)
#     obb = pcdAsset.get_oriented_bounding_box()

#     # Create a triangle from the oriented box    
#     minBox = obb.get_min_bound()
#     maxBox = obb.get_max_bound()
#     triangle = np.asarray([centerCamPoint, maxBox, minBox])
    
#     degrees = []

#     for deg in range(0, 360, 5):
#         triangleRotated = rotatePoints(triangle, deg)

#         p0 = (triangleRotated[0][X_AXIS], triangleRotated[0][Y_AXIS])
#         p1 = (triangleRotated[1][X_AXIS], triangleRotated[1][Y_AXIS])
#         p2 = (triangleRotated[2][X_AXIS], triangleRotated[2][Y_AXIS])

#         empty = True
#         for point in uniquePoints:
#             # print(point)
#             empty = empty and not pointInTriangle(p0, p1, p2, point)
        
#         if (empty): 
#             degrees.append(deg)
            
#     return degrees




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


def createCrossbuck(center):
    
    box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.22, depth=1.22)
    box2 = o3d.geometry.TriangleMesh.create_box(width=0.05, height=1.22, depth=0.22)

    box.translate(center, relative=False)
    box2.translate(center, relative=False)

    box += box2

    rotation2 = box.get_rotation_matrix_from_xyz((45, 0, 0))
    box.rotate(rotation2, center=box.get_center())

    return box


def createYeild():
    
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
    sign, _ = pcdSign.compute_convex_hull()

    return sign


def getSignAsset(scene, intensity, semantics, instances):
    
    maskSign = (semantics == 81)

    onlySigns = scene[maskSign, :]
    intensitySigns = intensity[maskSign]
    semanticsSigns = semantics[maskSign]
    instancesSigns = instances[maskSign]

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
    instancesSign= instancesSigns[labels == signIndex]

    return True, oneSign, intensitySign, semanticsSign, instancesSign


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
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        intensityIntersect[pointIndex] = intensityAsset[idx]
        semanticsIntersect[pointIndex] = semanticsAsset[idx]
        instancesIntersect[pointIndex] = instancesAsset[idx]

    newAsset, intensityAsset, semanticsAsset, instancesAsset = combine(newAsset, intensityAsset, semanticsAsset, instancesAsset, 
                                                                    newAssetScene, intensityIntersect, semanticsIntersect, instancesIntersect)

    # Return revised scene
    newAssetData = (newAsset, intensityAsset, semanticsAsset, instancesAsset)
    newSceneData = (sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect)
    return True, newAssetData, newSceneData


def signReplace(signAsset, intensityAsset, semanticsAsset, instancesAsset, scene, intensity, semantics, instances, details):

    pole = signAsset[semanticsAsset == 80]
    intensityPole = intensityAsset[semanticsAsset == 80]
    semanticsPole = semanticsAsset[semanticsAsset == 80]
    instancesPole = instancesAsset[semanticsAsset == 80]
    sign = signAsset[semanticsAsset == 81]
    intensitySign = intensityAsset[semanticsAsset == 81]
    semanticsSign = semanticsAsset[semanticsAsset == 81]
    instancesSign = instancesAsset[semanticsAsset == 81]

    if (np.shape(pole)[0] < 5 or np.shape(sign)[0] < 5):
        print("Sign {} pole {}, too little points".format(np.shape(sign)[0], np.shape(pole)[0]))
        return False, None, None, None, None, None, None, None, None, details
    
    # Get bounds to align the sign to 
    print(np.shape(sign))
    pcdSign = o3d.geometry.PointCloud()
    pcdSign.points = o3d.utility.Vector3dVector(sign)
    signBox = pcdSign.get_oriented_bounding_box()

    pcdPole = o3d.geometry.PointCloud()
    pcdPole.points = o3d.utility.Vector3dVector(pole)
    poleBox = pcdPole.get_oriented_bounding_box()
    signCenter = poleBox.get_center()
    
    minSign, maxSign = closestBoundingTwo(signBox.get_min_bound(), signBox.get_max_bound(), sign)

    minZSign = sys.maxsize
    for point in sign:
        minZSign = min(minZSign, point[2])

    maxSign[2] = minZSign
    minSign[2] = minZSign
    signCenter[2] = minZSign
    signLen = np.linalg.norm(maxSign - minSign)

    # Create shape mesh
    signMesh = None
    signType = random.randint(0, 5)
    if (signType == 0):
        signMesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.6, depth=0.75)
        details["sign"] = "speed"
    elif (signType == 1):
        signMesh = createCrossbuck(signCenter)
        details["sign"] = "crossbuck"
    elif (signType == 2):
        signMesh = createYeild()
        details["sign"] = "yeild"
    elif (signType == 3):
        signMesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.76, depth=0.76)    
        rotation = signMesh.get_rotation_matrix_from_xyz((45, 0, 0))
        signMesh.rotate(rotation, center=signMesh.get_center())
        details["sign"] = "warning"
    else:
        signMesh = createStopSign(signCenter)
        details["sign"] = "stop"


    meshMin = signMesh.get_min_bound()
    meshMax = signMesh.get_max_bound()
    heightMesh = meshMax[2] - meshMin[2]
    meshMax[2] = minZSign
    meshMin[2] = minZSign
    meshLen = np.linalg.norm(meshMin - meshMin)

    if (minZSign < -1):
        print("Sign too low min {}".format(minZSign))
        return False, None, None, None, None, None, None, None, None, details

    if (np.absolute(meshLen - signLen) > 2):
        print("Distance to sign too great: mesh len {}, sign len {}".format(meshLen, signLen))
        return False, None, None, None, None, None, None, None, None, details


    # Move the mesh to the center of the based on pole center, sign min, and height of the sign
    signCenter[2] = minZSign + (heightMesh / 2) 
    signMesh.translate(signCenter, relative=False)
    angleSign = getAngleRadians(signCenter, minSign, signMesh.get_min_bound())

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


    

    # Add the pole points to the scene
    scene, intensity, semantics, instances = combine(scene, intensity, semantics, instances, 
                                                    pole, intensityPole, semanticsPole, instancesPole)


    # Pull the points to the mesh
    assetData = (sign, intensitySign, semanticsSign, instancesSign)
    sceneData = (scene, intensity, semantics, instances)
    success, newAssetData, newSceneData = pointsToMesh(signMesh, assetData, sceneData)

    sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect = newSceneData
    sign, intensitySign, semanticsSign, instancesSign = newAssetData

    if (success and np.shape(sign)[0] < 20):
        print("Sign too little points")
        success = False

    return success, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect, sign, intensitySign, semanticsSign, instancesSign, details


def scaleVehicle(asset, intensityAsset, semanticsAsset, instancesAsset, 
                scene, intensity, semantics, instances, details):


    # Prepare to create the mesh estimating normals
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.estimate_normals()
    pcdAsset.orient_normals_towards_camera_location()

    # Create a mesh using the ball pivoting method
    mesh = None
    if (np.shape(asset)[0] < 5000):
        radii = [0.15, 0.15, 0.15, 0.15]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))
    else:
        mesh, _ = pcdAsset.compute_convex_hull()

    # Check that the mesh is valid
    if (np.shape(np.array(mesh.vertices))[0] < 1 or np.shape(np.array(mesh.triangles))[0] < 1):
        print("MESH NOT SUFFICENT: Vertices {} Triangles {}".format(np.shape(np.array(mesh.vertices))[0], np.shape(np.array(mesh.triangles))[0]))
        return False, None, None, None, None, None, None, None, None, None
    
    # Scale the vehicle mesh
    scale = random.uniform(1.01, 1.05)
    details["scale"] = scale
    mesh.scale(scale, center=mesh.get_center())

    # Scale the points
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
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        intensityIntersect[pointIndex] = intensityAsset[idx]
        semanticsIntersect[pointIndex] = semanticsAsset[idx]
        instancesIntersect[pointIndex] = instancesAsset[idx]

    newAsset, intensityAsset, semanticsAsset, instancesAsset = combine(newAsset, newIntensityAsset, newSemanticsAsset, newInstancesAsset, 
                                                                    newAssetScene, intensityIntersect, semanticsIntersect, instancesIntersect)


    # print(np.shape(newAsset))
    if (np.shape(newAsset)[0] < 20):
        print("New asset too small {}".format(np.shape(newAsset)[0]))
        return False, None, None, None, None, None, None, None, None, None


    # Return revised scene
    return True, newAsset, intensityAsset, semanticsAsset, instancesAsset, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect, details



















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

