#!/bin/bash



# -------------------------------------------------------------------------------------------------------------------

# Options

# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="SIGN_REPLACE"
mut="VEHICLE_DEFORM"
# mut="VEHICLE_INTENSITY"
# mut="VEHICLE_SCALE"

binPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset2/sequences/"
seq="00"
scene="000000"
# seq="04"
# scene="000244"
seq="08"
scene="000701"
# assetId="00-000000-213"
# assetId="00-002452-132-car"
# assetId="00-000000-1"
# assetId="00-000156-6"
# assetId="04-000050-6-moving-car"
assetId="08-000368-12-other-vehicle"
count=1
batch=100


# -------------------------------------------------------------------------------------------------------------------

# Run command 

# python semFuzzLidar.py -path "$velPath" -lbls $lblPath
python semFuzzLidar.py -binPath "$binPath" -labelPath $lblPath -m $mut -count $count -b $batch
# python semFuzzLidar.py -path "$basePath" -m $mut -vis -ns
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -ns -vis
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -ns
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -vis -assetId $assetId -rotate 323
# python semFuzzLidar.py -path "$velPath" -lbls $lblPath -m $mut -count $count -b $batch -assetId $assetId -rotate 116 -seq $seq -scene $scene

# -------------------------------------------------------------------------------------------------------------------




