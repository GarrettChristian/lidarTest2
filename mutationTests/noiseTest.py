
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

    o3d.visualization.draw_geometries([pcdAsset])

    # Select random point 1102
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
    # o3d.visualization.draw_geometries([pcdAsset], 
    #                                     zoom=1,
    #                                     front=[1, 1, 0],
    #                                     lookat=[0, 0, 0],
    #                                     up=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcdAsset, pcd])





"""
Main Method
"""
def main():
    # labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000001.label"
    # binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000001.bin"
    # labelsFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/labels/000001.label"
    # binFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/velodyne/000001.bin"
    labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000003.label"
    binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000003.bin"

    # ------

    pcdArr = np.fromfile(binFileName, dtype=np.float32)

    label_arr = np.fromfile(labelsFileName, dtype=np.int32)

    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))

    intensity = pcdArr[:, 3]

    pcdArr = np.delete(pcdArr, 3, 1)    
    
    semantics = label_arr & 0xFFFF
    labelsInstance = label_arr >> 16 

    # ------

    # asset = pcdArr[labelsInstance == 212]
    asset = pcdArr[labelsInstance == 213]

    # alignZdim(asset, pcdArr, intensity, semantics, labelsInstance)

    deform(asset, pcdArr, intensity, semantics, labelsInstance)

    # hiddenPointRem(pcdArr)

    
    # saveToBin()



if __name__ == '__main__':
    main()




















