#!/bin/bash
source ./.bashrc
source activate /home/sli/ycwang24/anaconda3/envs/tf1.12/
which python
#export model_name="mus-lstm-n"
python evaluate_ad.py
