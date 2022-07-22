

modelDirName="JS3C-Net"
modelsBaseDir="/home/garrett/Documents"
modelDir="$modelsBaseDir/$modelDirName"
dataDirRoot="/home/garrett/Documents/data/tmp/dataset"
dataDir="/home/garrett/Documents/data/tmp/dataset/sequences/00/velodyne"
predDir="/home/garrett/Documents/data/out"

container="js3c-net"
image=$container"_image"


# modelRunCommand="python3 test_kitti_segment.py " 
modelRunCommand="python3 -m trace --trace test_kitti_segment.py " 
modelRunCommand+=" --labels $dataDirRoot"
modelRunCommand+=" --log_dir $predDir"
modelRunCommand+=" --gpu 0 --dataset test"




# Run
# container name
# Access to GPUs
# User (otherwise we won't have permission to modify files created by bind mount)
# Mount model dir
# Mount data (bins) dir
# Mount predictions (bins) dir
# image
# bash (command) 


echo 
echo $modelRunCommand
echo 
echo Running Docker $container, $image 
echo 

docker run \
--name $container \
--gpus all \
-u root \
--mount type=bind,source="$modelDir",target="$modelDir" \
--mount type=bind,source="$dataDir",target="$dataDir" \
--mount type=bind,source="$predDir",target="$predDir" \
$image \
bash -c "cd $modelDir/lib/pointgroup_ops && python3 setup.py develop && cd $modelDir && $modelRunCommand"
# bash -c "cd $modelDir/lib/pointgroup_ops && python3 setup.py develop && pip3 list"


# Clean up container
docker container stop $container && docker container rm $container




