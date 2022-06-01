


from laserVisRangeImage import LaserScan
from laserVisRangeImage import LaserScanVis

# create a visualizer
scan = LaserScan(project=True)
# binFileName = "/home/garrett/Documents/data/dataset/sequences/00/velodyne/000000.bin"
# binFileName = "/home/garrett/Documents/lidarTest2/mutationTests/test1.bin"
# binFileName = "/Volumes/Extreme SSD/semKitti/dataset/sequences/00/velodyne/000000.bin"
binFileName = "0.bin"

vis = LaserScanVis(scan=scan, scan_name=binFileName)

vis.run()