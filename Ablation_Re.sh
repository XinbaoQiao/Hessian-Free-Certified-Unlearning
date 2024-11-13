# # Stepsize
for num_forget in 10 50  200 300; do
    for lr in  0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
    done
done

