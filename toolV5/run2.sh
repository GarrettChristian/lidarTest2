






# -------------------------------------------------------------------------------------------------------------------

# Options

# mut="SCENE_DEFORM"
# mut="SCENE_MIRROR_ROTATE"
# mut="SCENE_REMOVE"
# mut="ADD_ROTATE"
# mut="ADD_MIRROR_ROTATE"
mut="SIGN_REPLACE"
# mut="SCENE_INTENSITY"
#mut="SCENE_SCALE"

velPath="/home/garrett/Documents/data/dataset/sequences/"
lblPath="/home/garrett/Documents/data/dataset2/sequences/"
seq="00"
scene="000000"
# seq="04"
# scene="000244"
assetId="00-000000-213"
# assetId="00-000000-1"
# assetId="00-000156-6"
iter=20
batch=100


# -------------------------------------------------------------------------------------------------------------------

# Run command 

# python semFuzzLidar.py -path "$basePath" 
python semFuzzLidar.py -path "$velPath" -lbls $lblPath -m $mut -i $iter -b $batch
# python semFuzzLidar.py -path "$velPath" -lbls $lblPath -m $mut -i $iter -b $batch -vis
# python semFuzzLidar.py -path "$basePath" -m $mut -vis -ns
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -ns -vis
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -ns
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -vis -assetId $assetId -rotate 323
# python semFuzzLidar.py -path "$basePath" -m $mut -seq $seq -scene $scene -ns -assetId $assetId -rotate 0

# -------------------------------------------------------------------------------------------------------------------




