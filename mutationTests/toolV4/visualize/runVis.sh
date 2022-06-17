



# SET DIRS
# ./visualizeCustom.py -m -p /home/garrett/Documents/lidarTest2/toolV2/data/done/labels/cyl -v /home/garrett/Documents/lidarTest2/toolV2/data/done/velodyne
# ./visualizeCustom.py -p /home/garrett/Documents/lidarTest2/toolV2/data/done/labels/cyl -v /home/garrett/Documents/lidarTest2/toolV2/data/done/velodyne
# ./visualizeCustom.py -m -p /home/garrett/Documents/lidarTest2/toolV2/data/done/labels/cyl -v /home/garrett/Documents/lidarTest2/toolV2/data/done/velodyne

lbls="/home/garrett/Documents/lidarTest2/toolV3/data/done/labels/cyl"
vels="/home/garrett/Documents/lidarTest2/toolV3/data/done/velodyne"
# ./visualizeCustom.py -m -p $lbls -v $vels


# ------------------------------------------------------------------------------------------------

# SPECIFIC SCAN

# scan=/home/garrett/Documents/lidarTest2/toolV2/data/done/labels/actual/SWkv7xQDJc7Q845X7uiTFc-ADD_ROTATE.label
# scan=/home/garrett/Documents/lidarTest2/toolV2/data/done/labels/spv/ZEc4TVVh3Med3725tTA6mT-ADD_ROTATE.label
# scan=/home/garrett/Documents/lidarTest2/toolV2/data/done/labels/actual/ZEc4TVVh3Med3725tTA6mT-ADD_ROTATE.label
scan="/home/garrett/Documents/data/dataset/sequences/05/labels/001139.label" 
vel="/home/garrett/Documents/data/dataset/sequences/05/velodyne"
# vel=/home/garrett/Documents/lidarTest2/toolV2/data/done/velodyne

./visualizeCustom.py -ps $scan -v $vel

# ./visualizeCustom.py -ps 

