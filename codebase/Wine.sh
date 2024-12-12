for algorithm in main_proposed.py main_retrain.py main_NU.py main_IJ.py main_eva.py; do
for forget in 7 14; do
for seed in 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset wine --epochs 40 --num_dataset 142 --batch_size 64 --num_forget $forget --lr 0.01 --regularization 0.005 --lr_decay 0.9995 --clip 10 --gpu 0  --seed $seed
done
done
done