

import numpy as np
import open3d as o3d
import math


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


def assetIsValid(asset, sceneWithoutInstance):

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    #  Get the asset's bounding box
    obb = pcdAsset.get_oriented_bounding_box()
    boxPoints = np.asarray(obb.get_box_points())
    

    boxVertices = np.vstack((boxPoints, centerCamPoint))

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(boxVertices)
    hull2, _ = pcdCastHull.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull2)
    
    assetCenter = obb.get_center()

    # Dist is acceptable
    dist = np.linalg.norm(centerCamPoint - assetCenter)
    if dist > 50:
        return hull_ls, False
    
    incuded = checkInclusionBasedOnTriangleMeshAsset(sceneWithoutInstance, hull2)
    if (incuded):
        hull_ls.paint_uniform_color((1, 0.2, 0.2))
    else:
        hull_ls.paint_uniform_color((0.2, 1, 0.2))

    return hull_ls, not incuded


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


def viewOne(binFileName, labelsFileName):
    print(binFileName)
    print(labelsFileName)

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
    
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(pcdArr)
        

    # mask1 = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    # tmp = pcdArr[mask1, :]
    # pcdScene = o3d.geometry.PointCloud()
    # pcdScene.points = o3d.utility.Vector3dVector(tmp)

    display = [pcdScene]
    
    pcdCenter = o3d.geometry.PointCloud()
    pcdCenter.points = o3d.utility.Vector3dVector(centerArea)

    #  Get the asset's bounding box
    centerBox = pcdCenter.get_oriented_bounding_box()
    centerBox.color = (0.1, 0.2, 0.2)
    display.append(centerBox)

    for instance in seenInst:
        if instance != 0:
            instancePoints = pcdArr[labelInstance == instance]

            if (np.shape(instancePoints)[0] > 20):
                pcdItem = o3d.geometry.PointCloud()
                pcdItem.points = o3d.utility.Vector3dVector(instancePoints)
                hull, _ = pcdItem.compute_convex_hull()
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
                hull_ls.paint_uniform_color((0, 0, 1))


                maskInst = (labelInstance != instance) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
                pcdWithoutInstance = pcdArr[maskInst, :]

                boxToVel, valid = assetIsValid(instancePoints, pcdWithoutInstance)
                

                display.append(hull_ls)
                if (valid):
                    display.append(removeLidarShadowLines(instancePoints))
                    display.append(boxToVel)
                    # display.append(hullToVelLines(instancePoints, pcdWithoutInstance))

                # get_oriented_bounding_box
                # get_axis_aligned_bounding_box
                obb = pcdItem.get_oriented_bounding_box()
                obb.color = (0.7, 0, 1)
                display.append(obb)



    o3d.visualization.draw_geometries(display)



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
    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    addX = destination[X_AXIS] - point[X_AXIS]
    addY = destination[Y_AXIS] - point[Y_AXIS]

    points[:, X_AXIS] = points[:, X_AXIS] + addX
    points[:, Y_AXIS] = points[:, Y_AXIS] + addY

    return points


"""
Rotate a point counterclockwise by a given angle around a given origin.
https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
"""
def rotateOnePoint(origin, point, angle):    

    radians = (angle * math.pi) / 180

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

    # Rotate the points relative to their center 
    points = rotatePointsLocalized(points, angle)

    pcdPoints = o3d.geometry.PointCloud()
    pcdPoints.points = o3d.utility.Vector3dVector(points)
    obb = pcdPoints.get_oriented_bounding_box()
    centerOfPoints = obb.get_center()

    point = (centerOfPoints[0], centerOfPoints[1])

    newLocation = rotateOnePoint((0, 0), point, angle)

    points = translatePointsXY(points, newLocation)

    return points


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
Removes the LiDAR shadow by casting lines based on the hull of the asset
Then deleteing if the points are found within the polygon

https://math.stackexchange.com/questions/83404/finding-a-point-along-a-line-in-three-dimensions-given-two-points
"""
def removeLidarShadow(asset, scene, intensity, semantics, labelsInstance):

    # Prepare asset and scene point clouds
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(scene)

    #  Get the asset's hull mesh
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    
    centerCamPoint = np.array([0, 0, -0.5])
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

    hull2.scale(0.5, hull2.get_center())
    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdCut = o3d.geometry.PointCloud()
    pcdCut.points = o3d.utility.Vector3dVector(combinedVertices)
    cutPointsHull, _ = pcdCut.compute_convex_hull()

    mask = checkInclusionBasedOnTriangleMesh(scene, cutPointsHull)
    mask = np.logical_not(mask)

    scene = scene[mask, :]
    intensity = intensity[mask]
    semantics = semantics[mask]
    labelsInstance = labelsInstance[mask]

    return (scene, intensity, semantics, labelsInstance)