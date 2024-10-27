for seed in 42 124 3407 114514 5 1; do
        python3 -u main_MIAL.py --model logistic --dataset mnist --epochs 15 --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 1 --seed $seed
        python3 -u main_MIAL.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 1  --seed $seed
done

## Before running, please comment out all the code related to IJ/NU in main_MIAL.py.
# python3 -u main_MIAL.py --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50  --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 0  --seed 930
# python3 -u main_MIAL.py --model resnet18 --dataset lfw --epochs 50 --num_dataset 984 --batch_size 40 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 1 --seed 42
# python3 -u main_MIAL.py --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 7 --seed 42
