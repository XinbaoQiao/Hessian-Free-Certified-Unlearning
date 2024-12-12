for algorithm in main_proposed.py main_retrain.py main_NU.py main_IJ.py main_eva.py; do
for seed in 42 124 3407 114514 5 1 930; do
    python3 -u $algorithm  --model mlp --dataset obesity --epochs 100 --num_dataset 2111 --batch_size 64 --num_forget 106 --lr 0.05 --regularization 0.001 --lr_decay 0.9995 --clip 10 --gpu 1  --seed $seed
done
done
