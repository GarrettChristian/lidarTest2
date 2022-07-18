#!/bin/bash


# -------------------------------------------------------------------------------------------------------------------

# Options

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
# toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
toolData="/media/garrett/ExtraDrive1/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
# saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"
saveAt="/media/garrett/ExtraDrive1/output"

# toolData=/home/garrett/Documents/lidarTest2/mutationTests/toolV3/data/

# -------------------------------------------------------------------------------------------------------------------

# Run command 

python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt

# -------------------------------------------------------------------------------------------------------------------




