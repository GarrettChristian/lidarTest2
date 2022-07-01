"""
semFuzzLidar 
Main runner for the mutation tool

@Author Garrett Christian
@Date 6/23/22
"""

import argparse
import os

from domain.toolSessionManager import SessionManager

import controllers.mutationTool.mutationRunner as mutationRunner


# -------------------------------------------------------------
# Arguments


def parse_args():
    p = argparse.ArgumentParser(
        description='Model Runner')

    # Required params
    p.add_argument("-binPath", 
        help="Path to the sequences folder of LiDAR scan bins", 
        nargs='?', const="/home/garrett/Documents/data/dataset/sequences/", 
        default="/home/garrett/Documents/data/dataset/sequences/")

    p.add_argument("-labelPath", 
        help="Path to the sequences label files should corrispond with velodyne", 
        nargs='?', const="/home/garrett/Documents/data/dataset2/sequences/", 
        default="/home/garrett/Documents/data/dataset2/sequences/")

    p.add_argument('-m', 
        help='Transformations to perform comma seperated example: ADD_ROTATE,ADD_MIRROR_ROTATE or ADD_ROTATE defaults to ADD_ROTATE')

    p.add_argument("-b", 
        help="Batch to create before evaluating", 
        nargs='?', const=100, default=100)

    p.add_argument("-count", 
        help="The total number of valid mutations you would like to create", 
        nargs='?', const=1, default=1)

    p.add_argument("-mdb", 
        help="Path to the connection string for mongo", 
        nargs='?', const="/home/garrett/Documents/lidarTest2/mongoconnect.txt", 
        default="/home/garrett/Documents/lidarTest2/mongoconnect.txt")

    # Tool configurable params
    p.add_argument("-saveAt", 
        help="Location to save the tool output", 
        nargs='?', const=os.getcwd(), 
        default=os.getcwd())

    p.add_argument("-t", 
        help="Thread number default to 1", 
        nargs='?', const=2, default=2)

    p.add_argument("-scaleLimit", 
        help="Limit to the number of points for scale", 
        nargs='?', const=10000, default=10000)

    p.add_argument('-asyncEval', 
        help='Seperates the evaluation into its own process',
        action='store_true', default=False)

    # Optional Flags
    p.add_argument('-vis', 
        help='Visualize with Open3D',
        action='store_true', default=False)

    p.add_argument('-verbose', help='Enables verbose logging',
        action='store_true', default=False)

    p.add_argument('-ne', help='Disables Evaluation',
        action='store_false', default=True)

    p.add_argument('-ns', help='Disables Saving',
        action='store_false', default=True)

    # Debug options
    # Asset / scene
    p.add_argument("-assetId", 
        help="Asset Identifier, optional forces the tool to choose one specific asset", 
        nargs='?', const=None, default=None)

    p.add_argument("-seq", 
        help="Sequences number, provide as 1 rather than 01 (default all labeled 0-10) CANNOT BE USED WITHOUT assetId & scene", 
        nargs='?', const=0, default=range(0, 11))

    p.add_argument( "-scene", 
        help="Specific scene number provide full ie 002732, CANNOT BE USED WITHOUT assetId & seq",
        nargs='?', const=None, default=None)
    
    # Mutation debug / recreation options
    # Rotate
    p.add_argument('-rotate', 
        help='Value to rotate', 
        default=None,
        required=False)
    # Mirror
    p.add_argument('-mirror', 
        help='Value to mirror', 
        default=None,
        required=False)
    # Intensity
    p.add_argument('-intensity', 
        help='Value to change intensity', 
        default=None,
        required=False)
    # Scale
    p.add_argument('-scale', 
        help='Value to scale by', 
        default=None,
        required=False)
    # Sign
    p.add_argument('-sign', 
        help='Sign to replace with', 
        default=None,
        required=False)
    # Deform
    p.add_argument('-deformPercent', 
        help='Amount of the asset to deform', 
        default=None,
        required=False)
    p.add_argument('-deformPoint', 
        help='Point to deform from', 
        default=None,
        required=False)
    p.add_argument('-deformMu', 
        help='Mean for random deformation', 
        default=None,
        required=False)
    p.add_argument('-deformSigma', 
        help='Sigma for random deformation', 
        default=None,
        required=False)
    p.add_argument('-deformSeed', 
        help='Sigma for random deformation', 
        default=None,
        required=False)

    
    return p.parse_args()

    


# ----------------------------------------------------------

def main():

    print("\n\n------------------------------")
    print("\n\nStarting Semantic LiDAR Fuzzer\n\n")
    
    # Get arguments 
    args = parse_args()
    
    # Perform the setup creating a session manager
    sessionManager = SessionManager(args)

    # Start the mutation tool
    print("Starting Mutation")
    try:
        mutationRunner.runMutations(sessionManager)

    except KeyboardInterrupt:
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Concluding\n")


   


if __name__ == '__main__':
    main()



