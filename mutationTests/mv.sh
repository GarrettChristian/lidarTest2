#!/bin/bash


# ------------------------------------------


# for model in "pol" "sq3"
for model in "dar"
do
    for folder in "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10"
    do
        mv /home/garrett/Documents/data/resultsBase/$folder/$model/sequences/00/predictions/*.label /home/garrett/Documents/data/resultsBase/$folder/$model
        rm -rf /home/garrett/Documents/data/resultsBase/$folder/$model/sequences
    done
done



