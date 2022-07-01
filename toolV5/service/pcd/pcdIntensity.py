
import numpy as np
import random

from sklearn.neighbors import NearestNeighbors

import domain.semanticMapping as semanticMapping

# --------------------------------------------------------------------------
# Intensity

def intensityChange(intensityAsset, type, details, intensityMod):

    # Create a mask that represents the portion to change the intensity for
    mask = np.ones(np.shape(intensityAsset), dtype=bool)

    if (type in semanticMapping.instancesVehicle.keys()):    
        dists = nearestNeighbors(intensityAsset, 2)
        class0 = intensityAsset[dists[:, 1] == 0]
        class1 = intensityAsset[dists[:, 1] == 1]

        # Take majority class
        mask = dists[:, 1] == 0
        if np.shape(class0)[0] < np.shape(class1)[0]:
            mask = dists[:, 1] == 1

        # Threshold for license intensity if NN didn't catch it
        mask = np.where(intensityAsset >= 0.8, False, mask)

        averageC0 = 0 
        maxC0 = 0
        averageC1 =0
        maxC1 = 0
        if (np.shape(class0)[0] > 0):
            averageC0 = np.average(class0)
            maxC0 = np.amax(class0)
        if (np.shape(class1)[0] > 0):
            averageC1 = np.average(class1)        
            maxC1 = np.amax(class1)
        print("Class 0 {} avg {} max {}, Class 1 {} avg {} max {}".format(np.shape(class0)[0], averageC0, maxC0, np.shape(class1)[0], averageC1, maxC1))

    average = np.average(intensityAsset[mask])
    
    mod = random.uniform(.1, .3)
    if average > .1:
        mod = random.uniform(-.1, -.3)

    # For given recreation
    if (intensityMod != None):
        mod = intensityMod

    details["intensity"] = mod
    
    print("Intensity {}".format(mod))
    
    print(intensityAsset)
    intensityAsset = np.where(mask, intensityAsset + mod, intensityAsset)
    intensityAsset = np.where(intensityAsset < 0, 0, intensityAsset)
    intensityAsset = np.where(intensityAsset > 1, 1, intensityAsset)
    print(intensityAsset)
    print(average)

    return intensityAsset, details


"""
nearestNeighbors
Seperates the values into k groups using nearest neighbors

https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian
"""
def nearestNeighbors(values, nbr_neighbors):

    zeroCol = np.zeros((np.shape(values)[0],), dtype=bool)
    valuesResized = np.c_[values, zeroCol]
    # np.append(np.array(values), np.array(zeroCol), axis=1)
    # valuesResized = np.hstack((np.array(values), np.array(zeroCol)))

    nn = NearestNeighbors(n_neighbors=nbr_neighbors, metric='cosine', algorithm='brute').fit(valuesResized)
    dists, idxs = nn.kneighbors(valuesResized)

    return dists


