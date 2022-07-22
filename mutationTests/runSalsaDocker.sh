

modelDirName="SalsaNext"
modelsBaseDir="/home/garrett/Documents"
modelDir="$modelsBaseDir/$modelDirName"
dataDir="/home/garrett/Documents/data/tmp/dataset/sequences/00/velodyne"
predDir="/home/garrett/Documents/data/out"


modelRunCommand="python3 infer.py"
modelRunCommand+=" -d /home/garrett/Documents/data/tmp/dataset"
modelRunCommand+=" -l /home/garrett/Documents/data/out"
modelRunCommand+=" -m $modelDir/pretrained"
modelRunCommand+=" -s test -c 1"

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
echo Running Docker
echo 

docker run \
--name salsa_next \
--gpus all \
--user "$(id -u)" \
--mount type=bind,source="$modelDir",target="$modelDir" \
--mount type=bind,source="$dataDir",target="$dataDir" \
--mount type=bind,source="$predDir",target="$predDir" \
salsa_next_image \
bash -c "cd $modelDir/train/tasks/semantic && $modelRunCommand"



# Clean up container
docker container stop salsa_next && docker container rm salsa_next




