#! /bin/bash

for i in dhsnet amulet nldf srm picanet dss basnet cpd poolnet egnet scrn gcpa itsd minet
do python3 test_fps.py $i --gpus=$1 --backbone=vgg
python3 test_fps.py $i --gpus=$1
done