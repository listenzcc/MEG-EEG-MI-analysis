#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

script=mvpa.4.FBCSP.vote.py
# script=mvpa.4.FBCSP.vote-accumulate-dev.py

python $script -s ./rawdata/S01_20220119 -t 13 -m meg &
python $script -s ./rawdata/S02_20220315 -t 13 -m meg 
python $script -s ./rawdata/S03_20220322 -t 13 -m meg &
python $script -s ./rawdata/S04_20220329 -t 13 -m meg 
python $script -s ./rawdata/S05_20220426 -t 13 -m meg &
python $script -s ./rawdata/S06_20230627 -t 13 -m meg 
python $script -s ./rawdata/S07_20231220 -t 13 -m meg &
python $script -s ./rawdata/S08_20231221 -t 13 -m meg 
python $script -s ./rawdata/S09_20240110 -t 13 -m meg &
python $script -s ./rawdata/S10_20240111 -t 13 -m meg 

