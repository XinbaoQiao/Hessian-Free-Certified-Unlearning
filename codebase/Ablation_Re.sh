# # Stepsize
for num_forget in 50; do
    for lr in  0.001 0.005 0.01 0.015 0.02 0.025   ; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
    done
done

for num_forget in 50; do
    for lr in  0.1 0.15 0.2 0.25 0.3; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
    done
done


for num_forget in 50; do
    for lr in 0.26 0.27 0.28 0.29; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
    done
done


for num_forget in 50; do
    for lr in   0.03 0.035 0.04 0.045 0.05; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
    done
done
