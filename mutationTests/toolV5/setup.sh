#!/bin/bash

# ------------------------------------------------------------------------

echo 
echo "Running Environment Setup"
echo 

# ------------------------------------------------------------------------

# Python Path, should be the root of the tool (where this script is)
toolPath=$(pwd)
echo "Setting PYTHONPATH to $toolPath"
export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo $PYTHONPATH
echo 

# ------------------------------------------------------------------------

# Semantic Labels, should match your bins 
# Should match with the instances extracted
# "/home/../data/dataset2/sequences/"
labelPath="/home/garrett/Documents/data/dataset2/sequences/"
echo "Setting Label Path: $labelPath"
export LABELSPATH=$labelPath
echo $LABELSPATH
echo 

# ------------------------------------------------------------------------

# Bins, the LiDAR scans
# "/home/../data/dataset/sequences/"
binPath="/home/garrett/Documents/data/dataset/sequences/"
echo "Setting Bin Path: $binPath"
export BINSPATH=$binPath
echo $BINSPATH
echo 

# ------------------------------------------------------------------------

# A text file with the mongodb connection string
# https://www.mongodb.com/docs/manual/reference/connection-string/
mdbConnectPath="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
echo "Setting Mongodb Connection Path: $mdbConnectPath"
export MONGOCONNECT=$mdbConnectPath
echo $MONGOCONNECT
echo 

# ------------------------------------------------------------------------




