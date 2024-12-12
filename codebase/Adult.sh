for algorithm in main_proposed.py main_retrain.py main_NU.py main_IJ.py main_eva.py; do
for seed in 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset adult --epochs 80 --num_dataset 10000 --batch_size 5000 --num_forget 500 --lr 0.05  --regularization 0.05 --lr_decay 0.9995 --clip 10 --gpu 0  --seed $seed
done
done


for algorithm in main_proposed.py ; do
for seed in 2 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset adult --epochs 80 --num_dataset 10000 --batch_size 5000 --num_forget 500 --lr 0.05  --regularization 0.05 --lr_decay 0.9995 --clip 10 --gpu 0  --seed $seed
done
done

for algorithm in main_retrain.py; do
for seed in 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset adult --epochs 80 --num_dataset 10000 --batch_size 5000 --num_forget 500 --lr 0.05  --regularization 0.05 --lr_decay 0.9995 --clip 10 --gpu 1  --seed $seed
done
done


for algorithm in  main_eva.py; do
for seed in 42 124 3407 114514 5 1; do
    python3 -u $algorithm  --model mlp --dataset adult --epochs 80 --num_dataset 10000 --batch_size 5000 --num_forget 500 --lr 0.05  --regularization 0.05 --lr_decay 0.9995 --clip 10 --gpu 2  --seed $seed
done
done