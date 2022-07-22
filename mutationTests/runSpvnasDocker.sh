

modelDirName="spvnas"
modelsBaseDir="/home/garrett/Documents"
modelDir="$modelsBaseDir/$modelDirName"
dataDirRoot="/home/garrett/Documents/data/tmp/dataset"
dataDir="/home/garrett/Documents/data/tmp/dataset/sequences/00/velodyne"
predDir="/home/garrett/Documents/data/out"

container="spvnas"
image=$container"_image"


modelRunCommand="torchpack dist-run " 
modelRunCommand+="-np 1 python3 evaluate.py configs/semantic_kitti/default.yaml "
modelRunCommand+="--name SemanticKITTI_val_SPVNAS@65GMACs "
modelRunCommand+="--data-dir $dataDirRoot/sequences "
modelRunCommand+="--save-dir $predDir"

# docker run \
# --mount type=bind,source="/home/garrett/Documents/SalsaNext",target="/home/garrett/Documents/SalsaNext" \
# --mount type=bind,source="/home/garrett/Documents/lidarTest2/toolV5/output",target="/home/garrett/Documents/lidarTest2/toolV5/output" \
# --name salsa_next \
# --gpus all \
# salsa-test


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
--user "$(id -u)" \
--mount type=bind,source="$modelDir",target="$modelDir" \
--mount type=bind,source="$dataDir",target="$dataDir" \
--mount type=bind,source="$predDir",target="$predDir" \
$image \
bash -c "cd $modelDir && $modelRunCommand"



# Clean up container
docker container stop $container && docker container rm $container




