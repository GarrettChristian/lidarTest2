#!/bin/bash


# ------------------------------------------
# Params


data="/home/garrett/Documents/data/tmp/dataset"
pred="/home/garrett/Documents/data/out"
model="cyl"


# ------------------------------------------
# Command


python modelPredTester.py -data "$data" -pred "$predBasePath" -model "$model"


# ------------------------------------------



