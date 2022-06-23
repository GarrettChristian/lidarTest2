

# NOTE
# STORING THIS ALTERNATIVE REMOVE THE HERE, THE IDEA WAS THAT IT WOULD PLACE MASKS IN THE SPACES TO FILL
# THEN ANY POINTS THAT GET ROTATED WOULD GET PULLED TO THAT MASK




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
    # shadowVerticesRaised[:, 2] = shadowVerticesRaised[:, 2] + 1
    # pcdShadowRaised = o3d.geometry.PointCloud()
    # pcdShadowRaised.points = o3d.utility.Vector3dVector(shadowVerticesRaised)
    # hullShadowRaised, _ = pcdShadowRaised.compute_convex_hull()
    # maskNonGround = (semantics != 40) | (semantics != 44) | (semantics != 48) | (semantics != 49) | (semantics != 60) | (semantics != 72)
    # sceneWithoutGround = scene[maskNonGround]
    # maskAbove = checkInclusionBasedOnTriangleMesh(sceneWithoutGround, hullShadowRaised)
    # print("Above: ", np.sum(maskAbove))

    # if (np.sum(maskAbove) > 30):
    #     print("TOO MANY ABOVE")
    #     return False, None, None, None, None,

    # Remove

    #  Get the asset's convex hull
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    hull, _ = pcdAsset.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)
    assetBox = pcdAsset.get_oriented_bounding_box()

    

    # Find the min and max for the asset    
    # minAsset, maxAsset = closestBoundingTwo(assetBox.get_min_bound(), assetBox.get_max_bound(), asset)

    # Find the min and max for that hull
    minX = sys.maxsize
    minY = sys.maxsize
    minZ = sys.maxsize
    maxX = sys.maxsize * -1
    maxY = sys.maxsize * -1
    maxZ = sys.maxsize * -1
    for pt in np.asarray(hullVertices):
        maxX = max(maxX, pt[0])
        maxY = max(maxY, pt[1])
        maxZ = max(maxZ, pt[2])
        minX = min(minX, pt[0])
        minY = min(minY, pt[1])
        minZ = min(minZ, pt[2])

    # Get best angle
    minAsset = [0, 0, 0]
    maxAsset = [0, 0, 0]
    bestAngle = 0
    options = [[maxX, maxY, minZ], [maxX, minY, minZ], [minX, maxY, minZ], [minX, minY, minZ]]
    for pt1 in options:
        for pt2 in options:
            angle = getAngleRadians(centerCamPoint, pt1, pt2)
            angle = abs(angle)
            if (bestAngle < angle):
                bestAngle = angle
                minAsset = pt1
                maxAsset = pt2

    print("{} {} {}".format(minAsset, maxAsset, bestAngle))

    # minAsset, maxAsset = closestBoundingTwo(minAsset, maxAsset, asset)

    # Get the mid point between the two
    midPointX = (minAsset[0] + maxAsset[0]) / 2
    midPointY = (minAsset[1] + maxAsset[1]) / 2
    midPoint = np.array([midPointX, midPointY, minZ])
    midPointMaxZ = np.array([midPointX, midPointY, maxZ])

    # Get relative "left and right" of the bounds
    left = minAsset
    right = maxAsset
    if (isLeft(centerCamPoint, midPoint, maxAsset)):
        left = maxAsset
        right = minAsset


    # CREATE A MESH TO CAST POINTS TO 

    # Scale the shadow to find points on the edges
    scaledHull = copy.deepcopy(hull).scale(1.2, center=hull.get_center())
    scaledShadow = getLidarShadowMesh(np.asarray(scaledHull.vertices))

    # VIS TMP
    pcdCast2 = o3d.geometry.PointCloud()
    pcdCast2.points = o3d.utility.Vector3dVector(np.asarray([minAsset, maxAsset, midPoint]))
    pcdCast2.paint_uniform_color((.6, 0, .6))
    pcdScene = o3d.geometry.PointCloud()
    pcdScene.points = o3d.utility.Vector3dVector(scene)
    hull_ls21 = o3d.geometry.LineSet.create_from_triangle_mesh(shadow)
    hull_ls21.paint_uniform_color((0, 1, 0.5))
    hull_ls221 = o3d.geometry.LineSet.create_from_triangle_mesh(scaledShadow)
    hull_ls221.paint_uniform_color((0, 0.5, 1))
    o3d.visualization.draw_geometries([pcdScene, hull_ls21, hull_ls221, pcdCast2])

    # Get ground and walls
    maskNonGround = (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    sceneWithoutGround = scene[maskNonGround]
    maskGround = np.logical_not(maskNonGround)
    sceneGround = scene[maskGround]
    
    # Get points that are close to the hole
    maskGroundEdges = checkInclusionBasedOnTriangleMesh(sceneGround, scaledShadow)
    groundIncluded = sceneGround[maskGroundEdges]
    maskNonGroundEdges = checkInclusionBasedOnTriangleMesh(sceneWithoutGround, scaledShadow)
    wallsIncluded = sceneWithoutGround[maskNonGroundEdges]

    # Create a mask to fill the hole
    maskHole = None
    if (np.shape(groundIncluded)[0] > 4):
        pcdEdgeGround = o3d.geometry.PointCloud()
        pcdEdgeGround.points = o3d.utility.Vector3dVector(groundIncluded)
        hullGround, _ = pcdEdgeGround.compute_convex_hull()
        maskHole = hullGround
    if (np.shape(wallsIncluded)[0] > 4):
        pcdEdgeNonGround = o3d.geometry.PointCloud()
        pcdEdgeNonGround.points = o3d.utility.Vector3dVector(wallsIncluded)
        hullWalls, _ = pcdEdgeNonGround.compute_convex_hull()
        if (maskHole == None):
            maskHole = hullWalls
        else:
            maskHole += hullWalls

    if (maskHole == None):
        return False, None, None, None, None

    o3d.visualization.draw_geometries([pcdScene, maskHole])

    # Print sem data
    semSetInval = set()
    for sem in maskNonGroundEdges:
        if (sem in globals.instancesVehicle.keys()
            or sem in globals.instancesWalls.keys()):
            semSetInval.add(sem)
    for sem in semSetInval:
        print(globals.name_label_mapping[sem])


    # GET HALVES TO REPLACE WITH 
    

    # Split shadow in half
    replaceLeftShadow = [midPoint, midPointMaxZ]
    replaceRightShadow = [midPoint, midPointMaxZ]
    for point in shadowVertices:
        if (not isLeft(centerCamPoint, midPoint, point)):
            replaceLeftShadow.append(point)
        else:
            replaceRightShadow.append(point)

    # Validate that there are enough points to recreate a hull
    if (len(replaceLeftShadow) < 4 or len(replaceLeftShadow) < 4):
        print("Left shadow {}, Right shadow {} not enough points".format(len(replaceLeftShadow), len(replaceRightShadow)))
        return False, None, None, None, None

    # Rotate Shadow Halves
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
    
    replaceLeftShadow = rotatePoints(replaceLeftShadow, angleRight)
    replaceRightShadow = rotatePoints(replaceRightShadow, angleLeft)

    # Recreate hulls
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

    # Rotate points
    if (len(pcdIncluded) > 0):
        pcdIncluded = rotatePoints(pcdIncluded, angleLeft)
    else:
        print("left points empty")
    
    if (len(pcdIncluded2) > 0):
        pcdIncluded2 = rotatePoints(pcdIncluded2, angleRight)
    else:
        print("right points empty")

    # Combine left & right
    pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded = combine(pcdIncluded, intensityIncluded, semanticsIncluded, instancesIncluded,
        pcdIncluded2, intensityIncluded2, semanticsIncluded2, instancesIncluded2)

    # Calulate intersection for scene to mesh points
    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(maskHole)
    sceneRays = o3d.t.geometry.RaycastingScene()
    sceneRays.add_triangles(legacyMesh)

    if (np.shape(pcdIncluded)[0] < 1):
        print("No points found to fill hole with {}".format(np.shape(pcdIncluded)[0]))
        return False, None, None, None, None

    raysVectorsScene = []
    for point in pcdIncluded:
        raysVectorsScene.append([0, 0, 0, point[0], point[1], point[2]])

    rays = o3d.core.Tensor(raysVectorsScene, dtype=o3d.core.Dtype.Float32)
    ans = sceneRays.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    pointsOnMesh = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))

    newHole = []
    for vector in pointsOnMesh:
        newHole.append(vector.numpy())
    
    intensityHole = intensityIncluded[hit.numpy()] 
    semanticsHole = semanticsIncluded[hit.numpy()]
    instancesHole = instancesIncluded[hit.numpy()]
    


    hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated)
    hull_ls2.paint_uniform_color((0, 1, 0.5))
    hull_ls22 = o3d.geometry.LineSet.create_from_triangle_mesh(shadowRotated2)
    hull_ls22.paint_uniform_color((1, 0, 0.5))

    pcdNewAddition = o3d.geometry.PointCloud()
    pcdNewAddition.points = o3d.utility.Vector3dVector(newHole)
    # pcdScene = o3d.geometry.PointCloud()
    # pcdScene.points = o3d.utility.Vector3dVector(scene)


    

    # o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, hull_ls44, pcdNewAddition, pcdScene, pcdCast2])
    o3d.visualization.draw_geometries([hull_ls, hull_ls2, hull_ls22, pcdNewAddition, pcdScene, pcdCast2])
    # o3d.visualization.draw_geometries([hull, pcdCastHull22, pcdCast2])
    


    sceneReplace, intensityReplace, semanticsReplace, instancesReplace = combine(newHole, intensityHole, semanticsHole, instancesHole,
                                                                            scene, intensity, semantics, instances)


    
    return True, sceneReplace, intensityReplace, semanticsReplace, instancesReplace