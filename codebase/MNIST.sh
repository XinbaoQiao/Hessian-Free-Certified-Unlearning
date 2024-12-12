# ############# Verification #############
# Convex Setting
for seed in 42 930 124 114514 3407 5 1; do
    for num_forget in 50  10 50 100 150 200 250 300; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed $seed
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed $seed
        python3 -u main_NU.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed $seed
        python3 -u main_IJ.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed $seed
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05 --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed $seed
    done
done

# Non-Convex Setting
for seed in 42 930 124 114514 3407 5 1; do
    for num_forget in 50 10 50 100 150 200 250 300; do
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed $seed
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed $seed
        python3 -u main_NU.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed $seed
        python3 -u main_IJ.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed $seed
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed $seed
    done
done  

