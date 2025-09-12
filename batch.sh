#!/bin/bash

# script=0.read.evoked.py
script=mvpa.1.1.rawEpochs.megAreas.py


python $script -s ./rawdata/S01_20220119 &
python $script -s ./rawdata/S02_20220315 &
python $script -s ./rawdata/S03_20220322 &
python $script -s ./rawdata/S04_20220329 &
python $script -s ./rawdata/S05_20220426 &
python $script -s ./rawdata/S06_20230627 &
python $script -s ./rawdata/S07_20231220 &
python $script -s ./rawdata/S08_20231221 &
python $script -s ./rawdata/S09_20240110 &
python $script -s ./rawdata/S10_20240111 &
