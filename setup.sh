#! /bin/bash

DATASET_PATH=/home/cesare/Projects/datasets
DATA_PATH=./data/Assembly101
METADATA_PATH=$DATA_PATH/metadata


conda create -n loras python=3.10
mkdir -p $DATA_PATH
ln -s $DATASET_PATH/Assembly101/ $DATA_PATH



echo Request access to the dataset and 
echo Download dataset manually from Google drive at:
echo https://drive.google.com/drive/folders/1QoT-hIiKUrSHMxYBKHvWpW9Z9aCznJB7?usp=sharing
echo When ready, hit a key

pause 


wget https://raw.githubusercontent.com/assembly-101/assembly101-action-recognition/main/MS-G3D-action-recognition/CSVs/actions.csv -O $METADATA_PATH/actions.csv
wget https://raw.githubusercontent.com/assembly-101/assembly101-action-recognition/main/MS-G3D-action-recognition/CSVs/train.csv -O $METADATA_PATH/train.csv
wget https://raw.githubusercontent.com/assembly-101/assembly101-action-recognition/main/MS-G3D-action-recognition/CSVs/validation.csv -O $METADATA_PATH/validation.csv

python -m loras.datasets.assembly
