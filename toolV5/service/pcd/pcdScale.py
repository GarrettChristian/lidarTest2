
import copy
import numpy as np
import open3d as o3d
import random

import service.pcd.pcdCommon as pcdCommon


# --------------------------------------------------------------------------
# Scale

def scaleVehicle(asset, intensityAsset, semanticsAsset, instancesAsset, 
                scene, intensity, semantics, instances, details, 
                scaleLimit, scaleAmount):


    # Prepare to create the mesh estimating normals
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    pcdAsset.estimate_normals()
    pcdAsset.orient_normals_towards_camera_location()

    
    # Check if count of points are greater than allowed to use ball pivoting on
    if (np.shape(asset)[0] > scaleLimit):
        # print("Point count {} exceeds scale point limit {}".format(np.shape(asset)[0], scaleLimit))
        details["issue"] = "Point count {} exceeds scale point limit {}".format(np.shape(asset)[0], scaleLimit)
        return False, None, None, None, None, None, None, None, None, details
    
    # Create a mesh using the ball pivoting method
    radii = [0.15]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcdAsset, o3d.utility.DoubleVector(radii))

    # o3d.visualization.draw_geometries([mesh])

    # Check that the mesh is valid
    if (np.shape(np.array(mesh.vertices))[0] < 1 or np.shape(np.array(mesh.triangles))[0] < 1):
        # print("MESH NOT SUFFICENT: Vertices {} Triangles {}".format(np.shape(np.array(mesh.vertices))[0], np.shape(np.array(mesh.triangles))[0]))
        details["issue"] = "MESH NOT SUFFICENT: Vertices {} Triangles {}".format(np.shape(np.array(mesh.vertices))[0], np.shape(np.array(mesh.triangles))[0])
        return False, None, None, None, None, None, None, None, None, details
    
    # Smooth the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    mesh.compute_vertex_normals()

    # Scale the vehicle mesh
    scale = scaleAmount
    if (not scale):
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
        # print("SCENE or ASSET PROVIDED EMPTY: SCENE {}, ASSET {}".format(np.shape(scene)[0], np.shape(asset)[0]))
        details["issue"] = "SCENE or ASSET PROVIDED EMPTY: SCENE {}, ASSET {}".format(np.shape(scene)[0], np.shape(asset)[0])
        return False, None, None, None, None, None, None, None, None, details

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
    nonHitAsset = np.logical_not(hit.numpy())

    # print(len(newAsset))
    # print(len(newAssetScene))

    if len(newAsset) == 0 or len(newAssetScene) == 0:
        # print("GOT NONE OF THE OG ASSET {} OR NONE OF SCENE {}".format(len(newAsset), len(newAssetScene)))
        details["issue"] = "GOT NONE OF THE OG ASSET {} OR NONE OF SCENE {}".format(len(newAsset), len(newAssetScene))
        return False, None, None, None, None, None, None, None, None, details

    # Fix the intensity of each of the points in the scene that were pulled into the asset by using the closest scaled asset point
    pcd_tree = o3d.geometry.KDTreeFlann(scaledPoints)
    for pointIndex in range(0, len(newAssetScene)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(newAssetScene[pointIndex], 1)
        intensityIntersect[pointIndex] = intensityAsset[idx]
        semanticsIntersect[pointIndex] = semanticsAsset[idx]
        instancesIntersect[pointIndex] = instancesAsset[idx]

    newAsset, intensityAsset, semanticsAsset, instancesAsset = pcdCommon.combine(newAsset, newIntensityAsset, newSemanticsAsset, newInstancesAsset, 
                                                                    newAssetScene, intensityIntersect, semanticsIntersect, instancesIntersect)


    # print(np.shape(newAsset))
    if (np.shape(newAsset)[0] < 20):
        # print("New asset too little points {}".format(np.shape(newAsset)[0]))
        details["issue"] = "New asset too little points {}".format(np.shape(newAsset)[0])
        return False, None, None, None, None, None, None, None, None, details

    details["pointsRemoved"] = int(np.sum(nonHitAsset))
    details["pointsAffected"] = int(np.shape(newAsset)[0])

    # Return revised scene with scaled vehicle 
    return True, newAsset, intensityAsset, semanticsAsset, instancesAsset, sceneNonIntersect, intensityNonIntersect, semanticsNonIntersect, instancesNonIntersect, details
















