
saveDir="/home/garrett/Documents/savingRuns/runs6_23"


# -------------------------------------------------------------------------------------------------------------------


# echo "Running Tool"

# mut="SCENE_DEFORM"
# mut="SCENE_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SIGN_REPLACE"
# mut="SCENE_INTENSITY"
# mut="SCENE_SCALE"

velPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset2/sequences/"
count=2000
batch=100


# Run command 
python semFuzzLidar.py -path "$velPath" -lbls $lblPath -m $mut -count $count -b $batch

# -------------------------------------------------------------------------------------------------------------------

echo "Running Visulization"

data=/home/garrett/Documents/data/dataset/sequences/
labels=/home/garrett/Documents/data/resultsBase/
toolData="/home/garrett/Documents/lidarTest2/toolV5/data/"

cd finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" 
cd ..

# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/data/"
dataId="XBR6VmeWt4bLE3Ja6JMQLA"

cd analytics
python produceCsv.py -data "$toolData" 
cd ..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir


mv analytics/*.csv $newSaveDir
cp -r finalVisualize/finalvis $newSaveDir
rm -r finalVisualize/finalvis
cp -r data $newSaveDir
rm -r data

# -------------------------------------------------------------------------------------------------------------------

