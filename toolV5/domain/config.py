"""

CONFIGURATION FILE FOR SEMANTIC LiDAR FUZZER


"""


# ----------------------------------------------------


# The models in use for this run
# ["cyl", "spv", "sal"]
models = ["cyl", "spv", "sal", "sq3", "pol", "ran"]


# ----------------------------------------------------
# Paths


# The LiDAR scans in bin format
# "/home/../data/dataset/sequences"
SCAN_PATH = "/home/garrett/Documents/data/dataset/sequences"


# Ground Truth Semantic Labels, should correspond to your bins 
# Should correspond with the instances extracted
# "/home/../data/dataset2/sequences"
LABEL_PATH = "/home/garrett/Documents/data/dataset2/sequences"


# Semantic Predictions, should have a base prediction for each model and scan
# "/home/../data/dataset2/sequences"
MODEL_INITIAL_PREDICTION_PATH = "/home/garrett/Documents/data/resultsBase"


# Directory where models have been cloned
# "/home/../dir"
BASE_MODEL_DIRECTORY = "/home/garrett/Documents"


# A text file with the mongodb connection string
# Should contain the asset metadata corrisponding to your label instances
# https://www.mongodb.com/docs/manual/reference/connection-string/
mdbConnectPath = "/home/garrett/Documents/lidarTest2/mongoconnect.txt"



# ----------------------------------------------------
# Configurable mutation parameters

# TODO


# ----------------------------------------------------





