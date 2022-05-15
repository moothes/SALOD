#! /bin/bash

#for i in dhsnet amulet nldf srm picanet dss basnet cpd poolnet egnet scrn gcpa itsd minet
#for i in f3net ldf gatenet pfsnet ctdnet
for i in dhsnet itsd scrn
do python3 test_fps.py $i --gpus=$1 --backbone=vgg
python3 test_fps.py $i --gpus=$1
done