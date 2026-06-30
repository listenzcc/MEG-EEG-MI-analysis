#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

script=mvpa.4.FBCSP.vote.py

python $script -s ./rawdata/S01_20220119 -m eeg -t 1234 &
python $script -s ./rawdata/S02_20220315 -m eeg -t 1234 &
python $script -s ./rawdata/S03_20220322 -m eeg -t 1234 &
python $script -s ./rawdata/S04_20220329 -m eeg -t 1234 &
python $script -s ./rawdata/S05_20220426 -m eeg -t 1234 &
python $script -s ./rawdata/S06_20230627 -m eeg -t 1234 &
python $script -s ./rawdata/S07_20231220 -m eeg -t 1234 &
python $script -s ./rawdata/S08_20231221 -m eeg -t 1234 &
python $script -s ./rawdata/S09_20240110 -m eeg -t 1234 &
python $script -s ./rawdata/S10_20240111 -m eeg -t 1234
