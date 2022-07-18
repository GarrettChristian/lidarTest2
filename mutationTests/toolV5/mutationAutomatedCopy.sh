#!/bin/bash

# -------------------------------------------------------------------------------------------------------------------


saveDir="/media/garrett/ExtraDrive1/savingRuns7_7"


# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
# mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output

# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

# mut="ADD_ROTATE"
mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
# mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output


# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output



# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
# mut="VEHICLE_DEFORM"
mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output



# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
# mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output


# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
# mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output



# -------------------------------------------------------------------------------------------------------------------


echo "Running Tool"

# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
mut="SIGN_REPLACE"
# mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset4/sequences/"
predPath="/home/garrett/Documents/data/resultsBase/"
count=2000
batch=400


# Run command 
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -predPath $predPath -m $mut -count $count -b $batch -asyncEval -saveAll


# -------------------------------------------------------------------------------------------------------------------


echo "Running Visulization"

data="/home/garrett/Documents/data/dataset/sequences/"
labels="/home/garrett/Documents/data/resultsBase/"
toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/finalVisualize
python finalVisualization.py -pdata "$data" -plabels "$labels" -ptool "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..


# -------------------------------------------------------------------------------------------------------------------

echo "Running Analytics"

toolData="/home/garrett/Documents/lidarTest2/toolV5/output/"
mongoconnect="/home/garrett/Documents/lidarTest2/mongoconnect.txt"
saveAt="/home/garrett/Documents/lidarTest2/toolV5/output"


cd controllers/analytics
python produceCsv.py -data "$toolData" -mdb "$mongoconnect" -saveAt $saveAt
cd ../..

# -------------------------------------------------------------------------------------------------------------------

echo "Moving data"

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
echo "Current Time: $current_time"
 
newDir=$mut"_"$current_time
echo "Dir name: $newDir"

newSaveDir=$saveDir/$newDir
echo "Save at: $newSaveDir"

mkdir $newSaveDir

cp -r output $newSaveDir
rm -r output


# -------------------------------------------------------------------------------------------------------------------



