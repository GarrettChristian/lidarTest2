
import numpy as np
import open3d as o3d


import math
import random

from scipy.interpolate import interp1d



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
        pt = (point[0], point[1])
        newLocation = rotateOnePoint((0, 0), pt, angle)
        point[0] = newLocation[0]
        point[1] = newLocation[1]

    return pointsRotated




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

    evalLine = np.zeros((count + 1, 3), dtype=np.float)
    evalLine[:, 0] = xVals
    evalLine[:, 1] = yVals

    return evalLine


def voxWalls(asset, scene, intensity, semantics, instances):
    
    maskNotGround = (semantics != 0) & (semantics != 1) & (semantics != 40) & (semantics != 44) & (semantics != 48) & (semantics != 49) & (semantics != 60) & (semantics != 72)
    scene = scene[maskNotGround]
    scene[:, 2] = 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene)
    pcd.estimate_normals()
    print(pcd.has_normals())
    pcd.orient_normals_towards_camera_location()

    #  Get the asset's bounding box
    pcdAsset = o3d.geometry.PointCloud()
    pcdAsset.points = o3d.utility.Vector3dVector(asset)
    obb = pcdAsset.get_oriented_bounding_box()
    assetCenter = obb.get_center() 

    voxelGridWalls = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    display = [voxelGridWalls, pcdAsset]

    linePoints = getPointsOnLine((0, 0), assetCenter, 40)
    degrees = []


    # for deg in range(0, 360, 5):
    for deg in range(0, 360):
        lineRotated = rotatePoints(linePoints, deg)

        pcdLine = o3d.geometry.PointCloud()
        pcdLine.points = o3d.utility.Vector3dVector(lineRotated)
        pcdLine.paint_uniform_color((1, 0.2, 0.2))
        
        included = voxelGridWalls.check_if_included(o3d.utility.Vector3dVector(lineRotated))
        included = np.logical_or.reduce(included, axis=0)
        if (not included): 
            degrees.append(deg)
            pcdLine.paint_uniform_color((0.2, 1, 0.2))
            
    
        display.append(pcdLine)



    print(degrees)

    

    # o3d.visualization.draw_geometries([pcdAsset])
    o3d.visualization.draw_geometries(display)


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
    instances = label_arr >> 16 

    # ------

    asset = pcdArr[instances == 213]
    
    pcdArr = pcdArr[instances != 213]
    intensity = intensity[instances != 213]
    semantics = semantics[instances != 213]
    instances = instances[instances != 213]

    # alignZdim(asset, pcdArr, intensity, semantics, instances)

    voxWalls(asset, pcdArr, intensity, semantics, instances)
    # meshAsset(asset, pcdArr, intensity, semantics, instances)

    # hiddenPointRem(pcdArr)

    
    # saveToBin()



if __name__ == '__main__':
    main()




















