
# MIAU results with 5% data to be forgotten
for num_forget in 50; do
        python3 -u main_MIAU.py  --model logistic --dataset mnist  --epochs 15 --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 1 --seed 124
        python3 -u main_MIAU.py  --model cnn --dataset mnist --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 1  --see 42
done 

# impact of deletion rate （Seed 42）
for num_forget in 10 50 100 150 200 250 300; do
        python3 -u main_MIAU.py  --model logistic --dataset mnist  --epochs 15 --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 1 --seed 124
        python3 -u main_MIAU.py  --model cnn --dataset mnist --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 1  --see 42
done 

# impact of deletion rate （Different Seeds）
for seed in 42 930 124 114514 3407 5 1; do
for num_forget in 10 50 100 150 200 250 300; do
        python3 -u main_MIAU.py  --model logistic --dataset mnist  --epochs 15 --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 1 --seed $seed
        python3 -u main_MIAU.py  --model cnn --dataset mnist --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 1  --see $seed
done 
done

## Before running, please comment out all the code related to IJ/NU in main_MIAU.py.
# python3 -u main_MIAU.py --model resnet18 --method sorted_concat  --dataset lfw --epochs 50 --num_dataset 984 --batch_size 40 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 1 --seed 930
# python3 -u main_MIAU.py --model resnet18 --method sorted_concat  --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50  --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 0  --seed 930
# python3 -u main_MIAU.py --model resnet18 --method sorted_concat  --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42


