
import numpy as np
import open3d as o3d


import math
import random




def voxCheck(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcdVox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    print(pcdVox)

    o3d.visualization.draw_geometries([pcdVox])



def hiddenPointRem(points):
    
    print(np.shape(points))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print(diameter)

    # diameter = 20
    camera = [0, 0, diameter]
    radius = diameter * 100

    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw_geometries([pcd])

    print(np.shape(pcd.points))


def alignZdim(asset, scene, intensity, semantics, labelsInstance):

    asset[:, 2] = asset[:, 2] + 10

    maskGround = (semantics == 40) | (semantics == 44) | (semantics == 48) | (semantics == 49) | (semantics == 60) | (semantics == 72)
    pcdArrGround = scene[maskGround, :]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    aabb = pcdAsset.get_axis_aligned_bounding_box()
    aabb.color = (0, 0, 1)
    boxPoints = np.asarray(aabb.get_box_points())
    print(boxPoints)

    boxMinZ = np.min(boxPoints.T[2])


    bP1 = boxPoints[boxPoints[:, 2] == boxMinZ]
    bP2 = boxPoints[boxPoints[:, 2] != boxMinZ]
    bP1[:, 2] = bP1[:, 2] + 20
    bP2[:, 2] = bP2[:, 2] - 20

    boxPointsZLarger = np.vstack((bP1, bP2))

    print(boxPointsZLarger)

    largerBox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(boxPointsZLarger))
    largerBox.color = (1, 0, 1)



    mask = largerBox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pcdArrGround))
    onlyByCar = pcdArrGround[mask]
    groundMin = np.min(onlyByCar.T[2])

    print(boxMinZ)
    print(groundMin)

    boxMinZ = round(boxMinZ, 2)
    groundMin = round(groundMin, 2)

    change = groundMin - boxMinZ

    asset[:, 2] = asset[:, 2] + change

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.get_axis_aligned_bounding_box()
    aabb2 = pcdAsset.get_axis_aligned_bounding_box()
    aabb2.color = (1, 1, 0)

    o3d.visualization.draw_geometries([pcd, aabb, aabb2, largerBox])

    print(np.shape(pcd.points))


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


# http://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
def deform(asset, scene, intensity, semantics, labelsInstance):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.paint_uniform_color([0.5, 0.5, 0.5])

    # Select random point
    pointIndex = np.random.choice(asset.shape[0], 1, replace=False)
    print(pointIndex)
    pcdAsset.colors[pointIndex] = [1, 0, 0]

    # Nearest k points
    assetNumPoints = np.shape(asset)[0]
    percentDeform = random.uniform(0.05, 0.12)
    k = int(assetNumPoints * percentDeform)

    print("Find its 200 nearest neighbors, and paint them blue.")
    pcd_tree = o3d.geometry.KDTreeFlann(pcdAsset)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcdAsset.points[pointIndex], k)
    # np.asarray(pcdAsset.colors)[idx[1:], :] = [0, 0, 1]

    mu, sigma = 0.05, 0.04
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma, (k))
    # noise = np.lexsort((noise[:,0], noise[:,1]))
    noise = np.sort(noise)[::-1]
    print(np.shape(noise))
    print(noise)
    print(assetNumPoints)
    print(k)
    print(percentDeform)

    for index in range(0, len(idx)):
        asset[idx[index]] = translatePointFromCenter(asset[idx[index]], noise[index])
        if index != 0:
            pcdAsset.colors[idx[index]] = [0, 0, 1 - (index * .002)]

    pcdAsset.points = o3d.utility.Vector3dVector(asset)

    print(idx)

    # o3d.visualization.draw_geometries([pcdAsset])
    o3d.visualization.draw_geometries([pcdAsset, pcd])


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

        ba = 0 - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen

        pt2 = 0 + ((-100) * ba2)

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


def meshAsset(asset, scene, intensity, semantics, labelsInstance):
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.paint_uniform_color([0.5, 0.5, 0.7])
    pcdAsset.estimate_normals()
    print(pcdAsset.has_normals())
    pcdAsset.orient_normals_towards_camera_location()

    # mesh, _ = pcdAsset.compute_convex_hull()
    radii = [0.15, 0.15, 0.15, 0.15]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))

    mesh.scale(1.2, center=mesh.get_center())


    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    sceneRays = o3d.t.geometry.RaycastingScene()
    sceneRays.add_triangles(legacyMesh)

    shadow = getLidarShadowMesh(np.array(mesh.vertices))
    sceneMask = checkInclusionBasedOnTriangleMesh(scene, shadow)
    sceneIncluded = scene[sceneMask]
    sceneMaskNot = np.logical_not(sceneMask)
    sceneNotIncluded = scene[sceneMaskNot]

    raysVectors = []
    for point in sceneIncluded:
        raysVectors.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectors, dtype=o3d.core.Dtype.Float32)

    ans = sceneRays.cast_rays(rays)

    print(ans.keys())
    # print(ans["t_hit"][0])
    # print(ans["primitive_ids"][0])
    # print(ans["primitive_uvs"][0])

    # print(ans["t_hit"])

    # for hit in ans["t_hit"]:
    #     if (hit != math.inf):
    #         print(hit.numpy())

    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    print(np.shape(pointsOnMesh))

    # print(sceneRays.count_intersections(rays))

    newPosition = []
    for vector in pointsOnMesh:
        # print(vector.numpy())
        newPosition.append(vector.numpy())
    # print(newPosition)

    pcdScene2 = o3d.geometry.PointCloud()
    pcdScene2.points = o3d.utility.Vector3dVector(sceneIncluded)
    pcdScene3 = o3d.geometry.PointCloud()
    pcdScene3.points = o3d.utility.Vector3dVector(sceneNotIncluded)
    pcdScene4 = o3d.geometry.PointCloud()
    pcdScene4.points = o3d.utility.Vector3dVector(pointsOnMesh.numpy())
    
    numPoints = np.sum(sceneMask)
    

    # tmpPcd = mesh.sample_points_uniformly(number_of_points = numPoints)

    # o3d.visualization.draw_geometries([pcdAsset])
    # o3d.visualization.draw_geometries([pcdScene2, rec_mesh])
    o3d.visualization.draw_geometries([pcdScene3, mesh, pcdAsset])
    o3d.visualization.draw_geometries([pcdScene3, pcdScene4, pcdAsset])


def meshAssetUniformSample(asset, scene, intensity, semantics, labelsInstance):
    
    scene = scene[semantics != 212]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.paint_uniform_color([0.5, 0.5, 0.7])
    pcdAsset.estimate_normals()
    print(pcdAsset.has_normals())
    pcdAsset.orient_normals_towards_camera_location()

    # radii = [0.07, 0.07, 0.07, 0.07]
    radii = [0.1, 0.1, 0.1, 0.1]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))

    # alpha = 0.7
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcdAsset, alpha)
    # mesh.compute_vertex_normals()


    # hull, _ = pcdAsset.compute_convex_hull()

    rec_mesh.scale(1.2, center=rec_mesh.get_center())


    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(rec_mesh)
    sceneRays = o3d.t.geometry.RaycastingScene()
    # sceneRays.add_triangles(legacyMesh)

    shadow = getLidarShadowMesh(np.array(rec_mesh.vertices))
    sceneMask = checkInclusionBasedOnTriangleMesh(scene, shadow)
    sceneIncluded = scene[sceneMask]
    sceneMaskNot = np.logical_not(sceneMask)
    sceneNotIncluded = scene[sceneMaskNot]

    raysVectors = []
    for point in sceneIncluded:
        raysVectors.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectors, dtype=o3d.core.Dtype.Float32)

    ans = sceneRays.cast_rays(rays)

    # print(ans.keys())
    # print(ans["t_hit"][0])
    # print(ans["primitive_ids"][0])
    # print(ans["primitive_uvs"][0])

    for hit in ans["t_hit"]:
        if (hit != math.inf):
            print(hit)

    pcdScene2 = o3d.geometry.PointCloud()
    pcdScene2.points = o3d.utility.Vector3dVector(sceneIncluded)
    pcdScene3 = o3d.geometry.PointCloud()
    pcdScene3.points = o3d.utility.Vector3dVector(sceneNotIncluded)
    
    numPoints = np.sum(sceneMask)

    tmpPcd = rec_mesh.sample_points_uniformly(number_of_points = numPoints)

    # o3d.visualization.draw_geometries([pcdAsset])
    # o3d.visualization.draw_geometries([pcdScene2, rec_mesh])
    o3d.visualization.draw_geometries([tmpPcd, pcdScene3])




def meshAssetV1(asset, scene, intensity, semantics, labelsInstance):
    
    scene = scene[semantics != 212]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.paint_uniform_color([0.5, 0.5, 0.7])
    pcdAsset.estimate_normals()
    print(pcdAsset.has_normals())
    pcdAsset.orient_normals_towards_camera_location()

    # radii = [0.07, 0.07, 0.07, 0.07]
    radii = [0.1, 0.1, 0.1, 0.1]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))


    # alpha = 0.7
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcdAsset, alpha)
    # mesh.compute_vertex_normals()


    # hull, _ = pcdAsset.compute_convex_hull()


    # o3d.visualization.draw_geometries([pcdAsset])
    o3d.visualization.draw_geometries([pcd, rec_mesh])


def meshWalls(asset, scene, intensity, semantics, labelsInstance):
    
    scene = scene[semantics != 212]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)
    pcd.estimate_normals()
    print(pcd.has_normals())
    pcd.orient_normals_towards_camera_location()

    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.paint_uniform_color([0.5, 0.5, 0.7])

    radii = [0.1, 0.1, 0.1, 0.1]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    alpha = 0.7
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    # o3d.visualization.draw_geometries([pcdAsset])
    o3d.visualization.draw_geometries([pcdAsset, rec_mesh])




"""
Main Method
"""
def main():
    labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000001.label"
    binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000001.bin"
    # labelsFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/labels/000001.label"
    # binFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/velodyne/000001.bin"

    # ------

    pcdArr = np.fromfile(binFileName, dtype=np.float32)

    label_arr = np.fromfile(labelsFileName, dtype=np.int32)

    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))

    intensity = pcdArr[:, 3]

    pcdArr = np.delete(pcdArr, 3, 1)    
    
    semantics = label_arr & 0xFFFF
    labelsInstance = label_arr >> 16 

    # ------

    asset = pcdArr[labelsInstance == 213]

    # alignZdim(asset, pcdArr, intensity, semantics, labelsInstance)

    # meshWalls(asset, pcdArr, intensity, semantics, labelsInstance)
    meshAsset(asset, pcdArr, intensity, semantics, labelsInstance)

    # hiddenPointRem(pcdArr)

    
    # saveToBin()



if __name__ == '__main__':
    main()




















