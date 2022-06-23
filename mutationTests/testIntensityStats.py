


from glob import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
import csv


instancesVehicle = {
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    18: 'truck',
    20: 'other-vehicle',
    # 31: 'bicyclist',
    # 32: 'motorcyclist',
    252: 'moving-car',
    # 253: 'moving-bicyclist',
    # 255: 'moving-motorcyclist',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

instances = {
    10: 'car',
    11: 'bicycle',
    # 13: 'bus', 
    15: 'motorcycle',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    # 31: 'bicyclist',
    32: 'motorcyclist',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

vehicles = set(instancesVehicle.keys())


maxAll = []
minAll = []
avgAll = []
pointsAll = []
maxSelect = []
minSelect = []
avgSelect = []
percentSelect = []
stdSelect = []


def createCsv(cols):
    csvFile = open("type_intensity_data.csv", "w")
    csvWriter = csv.writer(csvFile)

    rows = zip(*cols)

    for row in rows:
        csvWriter.writerow(row)
    

def createKey():
    col = []

    col.append("")
    col.append("All")
    col.append("Points All")
    col.append("Min of Points")
    col.append("Max of Points")
    col.append("Avg of Points")
    col.append("Std of Points")
    col.append("")
    col.append("Min Intensity All")
    col.append("Min of Min")
    col.append("Max of Min")
    col.append("Avg of Min")
    col.append("Std of Min")
    col.append("")
    col.append("Max Intensity All")
    col.append("Min of Max")
    col.append("Max of Max")
    col.append("Avg of Max")
    col.append("Std of Max")
    col.append("")
    col.append("Avg Intensity All")
    col.append("Min of Avg")
    col.append("Max of Avg")
    col.append("Avg of Avg")
    col.append("Std of Avg")
    col.append("")
    col.append("Select (based on Intensity)")
    col.append("Percent of Points Selected")
    col.append("Min of Percent")
    col.append("Max of Percent")
    col.append("Avg of Percent")
    col.append("Std of Percent")
    col.append("")
    col.append("Min Intensity Select")
    col.append("Min of Min")
    col.append("Max of Min")
    col.append("Avg of Min")
    col.append("Std of Min")
    col.append("")
    col.append("Max Intensity Select")
    col.append("Min of Max")
    col.append("Max of Max")
    col.append("Avg of Max")
    col.append("Std of Max")
    col.append("")
    col.append("Avg Intensity Select")
    col.append("Min of Avg")
    col.append("Max of Avg")
    col.append("Avg of Avg")
    col.append("Std of Avg")
    col.append("")
    col.append("Std Intensity Select")
    col.append("Min of Std")
    col.append("Max of Std")
    col.append("Avg of Std")
    col.append("Std of Std")
    col.append("")

    return col


def createCol(typeName):
    global maxAll
    global minAll
    global avgAll
    global maxSelect
    global minSelect
    global avgSelect
    global percentSelect
    global pointsAll
    global stdSelect

    col = []
    
    minAllNp = np.array(minAll)
    maxAllNp = np.array(maxAll)
    avgAllNp = np.array(avgAll)
    minSelectNp = np.array(minSelect)
    maxSelectNp = np.array(maxSelect)
    avgSelectNp = np.array(avgSelect)
    percentSelectNp = np.array(percentSelect)
    pointsAllNp = np.array(pointsAll)
    stdSelectNp = np.array(stdSelect)


    print("\n-----------------------\n\n{}".format(typeName))

    # All

    print("\Points ALL")
    minPoints = np.amin(pointsAllNp)
    maxPoints = np.amax(pointsAllNp)
    avgPoints = np.average(pointsAllNp)
    stdPoints = np.std(pointsAllNp)
    print("Min of Point Counts {}".format(minPoints))
    print("Max of Point Counts {}".format(maxPoints))
    print("Avg of Point Counts {}".format(avgPoints))
    print("Std of Point Counts {}".format(stdPoints))
    col.append(typeName)
    col.append("")
    col.append("")
    col.append(minPoints)
    col.append(maxPoints)
    col.append(avgPoints)
    col.append(stdPoints)
    
    print("\nMIN ALL")
    minMin = np.amin(minAllNp)
    maxMin = np.amax(minAllNp)
    avgMin = np.average(minAllNp)
    stdMin = np.std(minAllNp)
    print("Min of min values {}".format(minMin))
    print("Max of min values {}".format(maxMin))
    print("Avg of min values {}".format(avgMin))
    print("Std of min values {}".format(stdMin))
    col.append("")
    col.append("")
    col.append(minMin)
    col.append(maxMin)
    col.append(avgMin)
    col.append(stdMin)

    print("\nMAX ALL")
    minMax = np.amin(maxAllNp)
    maxMax = np.amax(maxAllNp)
    avgMax = np.average(maxAllNp)
    stdMax = np.std(maxAllNp)
    print("Min of max values {}".format(minMax))
    print("Max of max values {}".format(maxMax))
    print("Avg of max values {}".format(avgMax))
    print("Std of max values {}".format(stdMax))
    col.append("")
    col.append("")
    col.append(minMax)
    col.append(maxMax)
    col.append(avgMax)
    col.append(stdMax)

    print("\nAVG ALL")
    minAvg = np.amin(avgAllNp)
    maxAvg = np.amax(avgAllNp)
    avgAvg = np.average(avgAllNp)
    stdAvg = np.std(avgAllNp)
    print("Min of avg values {}".format(minAvg))
    print("Max of avg values {}".format(maxAvg))
    print("Avg of avg values {}".format(avgAvg))
    print("Std of avg values {}".format(stdAvg))
    col.append("")
    col.append("")
    col.append(minAvg)
    col.append(maxAvg)
    col.append(avgAvg)
    col.append(stdAvg)

    # Select
    print("\nPercent Select")

    minPercentSelect = np.amin(percentSelectNp)
    maxPercentSelect = np.amax(percentSelectNp)
    avgPercentSelect = np.average(percentSelectNp)
    stdPercentSelect = np.std(percentSelectNp)
    print("Min of min values {}".format(minPercentSelect))
    print("Max of min values {}".format(maxPercentSelect))
    print("Avg of min values {}".format(avgPercentSelect))
    print("Std of min values {}".format(stdPercentSelect))
    col.append("")
    col.append("")
    col.append("")
    col.append(minPercentSelect)
    col.append(maxPercentSelect)
    col.append(avgPercentSelect)
    col.append(stdPercentSelect)

    print("\nMIN Select")
    minMinSelect = np.amin(minSelectNp)
    maxMinSelect = np.amax(minSelectNp)
    avgMinSelect = np.average(minSelectNp)
    stdMinSelect = np.std(minSelectNp)
    print("Min of min values {}".format(minMinSelect))
    print("Max of min values {}".format(maxMinSelect))
    print("Avg of min values {}".format(avgMinSelect))
    print("Std of min values {}".format(stdMinSelect))
    col.append("")
    col.append("")
    col.append(minMinSelect)
    col.append(maxMinSelect)
    col.append(avgMinSelect)
    col.append(stdMinSelect)

    print("\nMAX Select")
    minMaxSelect = np.amin(maxSelectNp)
    maxMaxSelect = np.amax(maxSelectNp)
    avgMaxSelect = np.average(maxSelectNp)
    stdMaxSelect = np.std(maxSelectNp)
    print("Min of max values {}".format(minMaxSelect))
    print("Max of max values {}".format(maxMaxSelect))
    print("Avg of max values {}".format(avgMaxSelect))
    print("Std of max values {}".format(stdMaxSelect))
    col.append("")
    col.append("")
    col.append(minMaxSelect)
    col.append(maxMaxSelect)
    col.append(avgMaxSelect)
    col.append(stdMaxSelect)

    print("\nAVG Select")
    minAvgSelect = np.amin(avgSelectNp)
    maxAvgSelect = np.amax(avgSelectNp)
    avgAvgSelect = np.average(avgSelectNp)
    stdAvgSelect = np.std(avgSelectNp)
    print("Min of avg values {}".format(minAvgSelect))
    print("Max of avg values {}".format(maxAvgSelect))
    print("Avg of avg values {}".format(avgAvgSelect))
    print("Std of avg values {}".format(stdAvgSelect))
    col.append("")
    col.append("")
    col.append(minAvgSelect)
    col.append(maxAvgSelect)
    col.append(avgAvgSelect)
    col.append(stdAvgSelect)

    print("\nSTD Select")
    minStdSelect = np.amin(stdSelectNp)
    maxStdSelect = np.amax(stdSelectNp)
    avgStdSelect = np.average(stdSelectNp)
    stdStdSelect = np.std(stdSelectNp)
    print("Min of std values {}".format(minStdSelect))
    print("Max of std values {}".format(maxStdSelect))
    print("Avg of std values {}".format(avgStdSelect))
    print("Std of std values {}".format(stdStdSelect))
    col.append("")
    col.append("")
    col.append(minStdSelect)
    col.append(maxStdSelect)
    col.append(avgStdSelect)
    col.append(stdStdSelect)

    col.append("")
    return col





def intensityData(intensityAsset, typeNum):
    global maxAll
    global minAll
    global avgAll
    global maxSelect
    global minSelect
    global avgSelect
    global percentSelect
    global pointsAll
    global stdSelect

    mask = np.ones(np.shape(intensityAsset), dtype=bool)

    if (typeNum in vehicles):    
        dists = nearestNeighbors(intensityAsset, 2)
        class0 = intensityAsset[dists[:, 1] == 0]
        class1 = intensityAsset[dists[:, 1] == 1]

        mask = dists[:, 1] == 0
        if np.shape(class0)[0] < np.shape(class1)[0]:
            mask = dists[:, 1] == 1

    maxAll.append(np.amax(intensityAsset))
    minAll.append(np.amin(intensityAsset))
    avgAll.append(np.average(intensityAsset))
    selectIntensity = intensityAsset[mask]
    maxSelect.append(np.amax(selectIntensity))
    minSelect.append(np.amin(selectIntensity))
    avgSelect.append(np.average(selectIntensity))
    stdSelect.append(np.std(selectIntensity))

    selectCount = np.sum(selectIntensity)
    pointsNum = np.shape(intensityAsset)[0]
    percentSelect.append(selectCount / pointsNum)
    pointsAll.append(pointsNum)
    


# https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian
def nearestNeighbors(values, nbr_neighbors):

    zeroCol = np.zeros((np.shape(values)[0],), dtype=bool)
    valuesResized = np.c_[values, zeroCol]
    # np.append(np.array(values), np.array(zeroCol), axis=1)
    # valuesResized = np.hstack((np.array(values), np.array(zeroCol)))

    nn = NearestNeighbors(n_neighbors=nbr_neighbors, metric='cosine', algorithm='brute').fit(valuesResized)
    dists, idxs = nn.kneighbors(valuesResized)

    return dists




"""
Opens a bin and label file splitting between xyz, intensity, semantics, instances 
"""
def openLabelBinFiles(binFile, labelFile):
    # Label
    label_arr = np.fromfile(labelFile, dtype=np.int32)
    semantics = label_arr & 0xFFFF
    instances = label_arr >> 16 

    # Bin File
    pcdArr = np.fromfile(binFile, dtype=np.float32)
    pcdArr = pcdArr.reshape((int(np.shape(pcdArr)[0]) // 4, 4))
    
    intensity = pcdArr[:, 3]
    pcdArr = np.delete(pcdArr, 3, 1)

    return pcdArr, intensity, semantics, instances



"""
Opens a bin and label file splitting between xyz, intensity, semantics, instances 
"""
def openLabelBin(pathVel, pathLabel, sequence, scene):

    folderNum = str(sequence).rjust(2, '0')
    currPathVel = pathVel + folderNum
    currPathLbl = pathLabel + folderNum

    binFile = currPathVel + "/velodyne/" + scene + ".bin"
    labelFile = currPathLbl + "/labels/" + scene + ".label"

    return openLabelBinFiles(binFile, labelFile)


"""
Gets the data from a given asset Record 
"""
def getInstanceFromAssetRecord(assetRecord, pathVel, pathLbl):

    instance = assetRecord["instance"]
    sequence = assetRecord["sequence"]
    scene = assetRecord["scene"]

    _, intensity, _, labelInstance = openLabelBin(pathVel, pathLbl, sequence, scene)

    maskOnlyInst = (labelInstance == instance)

    intensity = intensity[maskOnlyInst]

    return intensity





"""
Connect to mongodb 
"""
def mongoConnect():
    global mutationCollection

    configFile = open("../mongoconnect.txt", "r")
    mongoUrl = configFile.readline()
    print("Connecting to mongodb")
    configFile.close()
    
    client = MongoClient(mongoUrl)
    db = client["lidar_data"]
    
    return db["assets3"]






def main():
    global maxAll
    global minAll
    global avgAll
    global maxSelect
    global minSelect
    global avgSelect
    global percentSelect
    global pointsAll
    global stdSelect

    print("\n\n------------------------------")
    print("\n\nStarting Mutation CSV Generator\n\n")

    # typeName = "car"
    # typeNames = ["bicycle"]
    typeNums = instances.keys()
    # typeNums = [252]

    velPath="/home/garrett/Documents/data/dataset/sequences/"
    lblPath="/home/garrett/Documents/data/dataset2/sequences/"
            
    assetCollection = mongoConnect()
    

    cols = [createKey()]

    for typeNum in typeNums:

        # Reset accumulators
        pointsAll = []
        maxAll = []
        minAll = []
        avgAll = []
        percentSelect = []
        maxSelect = []
        minSelect = []
        avgSelect = []
        stdSelect = []

        assetRecords = assetCollection.find({"typeNum": typeNum})

        for asset in assetRecords:
            print("{}".format(asset["_id"]))
            intensity = getInstanceFromAssetRecord(asset, velPath, lblPath)
            intensityData(intensity, asset["typeNum"])

        col = createCol(instances[typeNum])
        cols.append(col)

        

    createCsv(cols)
    


    
    

if __name__ == '__main__':
    main()



