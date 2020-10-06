#!/usr/bin/env bash

data='he'
ngpus=2
epochs=30
exp_name='nas_balloon'
for exp_name in 'fib' 'nas_stea' 'nas_lob' 'nas_balloon'
do
for exp_num in 'baseline_'${data}'_1' 'baseline_'${data}'_2' 'baseline_'${data}'_3'
do
num_class=3
if [ "$exp_name" = "nas_stea" ]; then
    echo "steat. class num 2"
    num_class=2
else
    echo "not steat. class num 3"
    num_class=3
fi
python train_patho.py --random-seed -1 --epochs ${epochs}  --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4 \
  --save-dir ./experiments/${exp_name}/${exp_num} --data-type ${data} --gpus ${ngpus}
python test_patho.py --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4  \
  --model-path ./experiments/${exp_name}/${exp_num}/checkpoint_best.pth.tar  \
  --save-dir ./experiments/${exp_name}/${exp_num}/best --data-type ${data} --gpus ${ngpus}
done
done
