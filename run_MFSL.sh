#!/usr/bin/env bash

exp_name='fib'
loss_type='single'
fusion_type='mid'
batch_size=4
ngpus=2
random_seed=-1
epochs=30
for exp_name in 'fib' 'nas_stea' 'nas_lob' 'nas_balloon'
do
for exp_num in 'MFSL_1' 'MFSL_2' 'MFSL_3'
do
num_class=3
if [ "$exp_name" = "nas_stea" ]; then
    echo "steat. class num 2"
    num_class=2
else
    echo "not steat. class num 3"
    num_class=3
fi
python train_fusion.py --random-seed ${random_seed} --epochs ${epochs} --exp ${exp_name} --exp-num ${exp_num} --fusion-type ${fusion_type} --loss-type ${loss_type} \
  --model Classifier --num-class ${num_class} --batch-size ${batch_size} \
  --save-dir ./experiments/${exp_name}/${exp_num} --gpus ${ngpus}
python test_fusion.py --exp ${exp_name} --exp-num ${exp_num}  --exp-num ${exp_num} --fusion-type ${fusion_type} \
  --model Classifier --num-class ${num_class} --batch-size ${batch_size}  \
  --model-path ./experiments/${exp_name}/${exp_num}/checkpoint_best.pth.tar  \
  --save-dir ./experiments/${exp_name}/${exp_num}/best --gpus ${ngpus}
done
done
