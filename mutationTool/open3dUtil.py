

from xml.etree.ElementInclude import include
import numpy as np
import open3d as o3d
import math
from sklearn.neighbors import NearestNeighbors

from globals import centerCamPoint
from globals import centerArea
from globals import X_AXIS
from globals import Y_AXIS
from globals import Z_AXIS
from globals import I_AXIS


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
Checks that all points exist above the ground
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
def translatePointsXY(points, destination):
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


def rotateOnePoint(origin, point, angle):

    radians = (angle * math.pi) / 180
    rotateOnePointRadians(origin, point, radians)

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



"""
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon
"""
def replaceBasedOnShadow(asset, scene, intensity, semantics, labelsInstance):

    shadow = getLidarShadowMesh(asset)

    #  Get the asset's bounding box
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    obb = pcdAsset.get_oriented_bounding_box()

    # Create a triangle from the oriented box    
    minBox = obb.get_min_bound()
    maxBox = obb.get_max_bound()

    radians = getAngleRadians((minBox[X_AXIS], minBox[Y_AXIS]), (maxBox[X_AXIS], maxBox[Y_AXIS]))
    angle = (radians * 180) / math.pi
    print(radians)
    print(angle)

    shadowVertices = np.asarray(shadow.vertices)
    
    for vertice in shadowVertices:
        print(vertice)
        newX, newY = rotateOnePointRadians((0, 0), (vertice[X_AXIS], vertice[Y_AXIS]), radians)
        vertice[X_AXIS] = newX
        vertice[Y_AXIS] = newY
        print(vertice)

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(shadowVertices)
    shadowRotated, _ = pcdCastHull.compute_convex_hull()

    maskIncluded = checkInclusionBasedOnTriangleMesh(scene, shadowRotated)
    maskNotIncluded = np.logical_not(maskIncluded)

    print(maskIncluded)

    sceneInMesh = scene[maskIncluded, :]
    intensityInMesh = intensity[maskIncluded]
    semanticsInMesh = semantics[maskIncluded]
    labelsInstanceInMesh = labelsInstance[maskIncluded]

    sceneNotInMesh = scene[maskNotIncluded, :]
    intensityNotInMesh = intensity[maskNotIncluded]
    semanticsNotInMesh = semantics[maskNotIncluded]
    labelsInstanceNotInMesh = labelsInstance[maskNotIncluded]

    for point in sceneInMesh:
        newX, newY = rotateOnePointRadians((0, 0), (point[X_AXIS], point[Y_AXIS]), radians)
        point[X_AXIS] = newX
        point[Y_AXIS] = newY
    
    return combine(sceneNotInMesh, intensityNotInMesh, semanticsNotInMesh, labelsInstanceNotInMesh,
                    sceneInMesh, intensityInMesh, semanticsInMesh, labelsInstanceInMesh)


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

    for deg in range(0, 360):
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
def getAngleRadians(p1, p2):
    p0x, p0y, _= centerCamPoint
    p1x, p1y = p1
    p2x, p2y = p2

    # result = math.atan2(p2y - p0y, p2x - p0x) - math.atan2(p1y - p0y, p1x - p1x)
    # return result
    a = np.asarray(p0x - p1x, p0y - p1y)
    b = np.asarray(p1x - p2x, p0y - p2y)

    return np.arccos((a * b) / (np.absolute(a) * np.absolute(b)))




