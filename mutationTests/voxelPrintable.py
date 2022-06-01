
import numpy as np
import open3d as o3d






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
Main Method
"""
def main():
    # labelsFileName = "/home/garrett/Documents/data/dataset/sequences/00/labels/000001.label"
    # binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000001.bin"
    labelsFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/labels/000001.label"
    binFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/velodyne/000001.bin"

    # ------

    pcdArr = np.fromfile(binFileName, dtype=np.float32)

    label_arr = np.fromfile(labelsFileName, dtype=np.int32)

    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))

    intensity = pcdArr[:, 3]

    pcdArr = np.delete(pcdArr, 3, 1)    
    
    semantics = label_arr & 0xFFFF
    labelsInstance = label_arr >> 16 

    # ------

    asset = pcdArr[labelsInstance == 212]

    alignZdim(asset, pcdArr, intensity, semantics, labelsInstance)


    # hiddenPointRem(pcdArr)

    
    # saveToBin()



if __name__ == '__main__':
    main()




















