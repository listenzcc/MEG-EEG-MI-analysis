#!/bin/bash

script=erd.1st.compute.py
mode=eeg

python $script -s S01_20220119 -m $mode &
python $script -s S02_20220315 -m $mode &
python $script -s S03_20220322 -m $mode &
python $script -s S04_20220329 -m $mode &
python $script -s S05_20220426 -m $mode &
python $script -s S06_20230627 -m $mode &
python $script -s S07_20231220 -m $mode &
python $script -s S08_20231221 -m $mode &
python $script -s S09_20240110 -m $mode &
python $script -s S10_20240111 -m $mode &
