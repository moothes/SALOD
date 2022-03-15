#! /bin/bash

for i in 10 30 50 100 300 500 1000 3000 5000 10000
do python3 train.py $1 --gpus=$2 --train_split=$i
done