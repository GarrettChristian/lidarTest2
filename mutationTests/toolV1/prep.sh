



rm /Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/dataset/sequences/00/labels/*
rm /Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/dataset/sequences/00/velodyne/*

mv *.label /Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/dataset/sequences/00/labels/
mv *.bin /Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/dataset/sequences/00/velodyne/




cd /Users/garrettchristian/DocumentsDesktop/uva21/summerProject/semantic-kitti-api
./visualize.py --sequence 00 --dataset /Users/garrettchristian/DocumentsDesktop/uva21/summerProject/lidarTest2/dataset/





