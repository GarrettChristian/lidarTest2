



echo "testing Cyl"

python modelRunner.py -model cyl

cp /home/garrett/Documents/data/results/cyl/000000.label /home/garrett/Documents/data/tmp/dataset/sequences/00/labels

echo "run visualizer"

cd /home/garrett/Documents/semantic-kitti-api/

# ./visualize.py --sequence 00 --dataset /home/garrett/Documents/data/tmp/dataset/



