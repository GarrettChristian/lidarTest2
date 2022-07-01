#!/bin/bash


# -------------------------------------------------------------------------------------------------------------------

# Options

# Where the final data json will come from
# Directory that contains
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
# or Mongodb id
dataId="XBR6VmeWt4bLE3Ja6JMQLA"

mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"

# -------------------------------------------------------------------------------------------------------------------

# Run command 

python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
# python produceCsv.py -id "$dataId" -mdb "$mongoconnect" -saveAt $saveAt

# -------------------------------------------------------------------------------------------------------------------




