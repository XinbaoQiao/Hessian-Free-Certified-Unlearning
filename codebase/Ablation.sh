# # Stepsize
for num_forget in 10 50  200 300; do
    for lr in  0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr $lr  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
    done
done

for num_forget in 10 50  200 300; do
    for lr in 0.2 0.3 0.4 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr $lr --clip 10  --gpu 0  --seed 42
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr $lr --clip 10  --gpu 0  --seed 42
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr $lr --clip 10  --gpu 0  --seed 42
    done
done

# Stepsize with more epoch
for epochs in 15 150 300 1000; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs  $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42

        python3 -u main_proposed.py --model logistic --dataset mnist --epochs $epochs --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.005  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.005  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.005  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
done

for epochs in 20 200 400 800; do
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 0  --seed 42
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05  --clip 10  --gpu 0  --seed 42
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget 200 --lr 0.05 --clip 10  --gpu 0  --seed 42
       
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --gpu 0  --seed 42
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05  --clip 10  --gpu 0  --seed 42
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget 200 --lr 0.05 --clip 10  --gpu 0  --seed 42

done


# # Epoch
for num_forget in 10 50  200 300; do
    for epochs in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 50 60 70 80 90 100 110 130 140 160 180 200; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget $num_forget --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 42
done
done

for num_forget in 10 50  200 300; do
    for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 30 35 40 45 50 55 60 70 80 90 100; do
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed 42
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed 42
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  $num_forget --lr 0.05 --clip 10  --gpu 0  --seed 42
done
done




# # lr_decay
for lr_decay in 1 0.995; do
    for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay $lr_decay --clip 5 --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay $lr_decay --clip 5 --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs $epochs  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay $lr_decay --clip 5 --gpu 0  --seed 42
done
done

for lr_decay in 1 0.995; do
    for epochs in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10  --lr_decay $lr_decay --gpu 0  --seed 42
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10 --lr_decay $lr_decay --gpu 0  --seed 42
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs $epochs --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip 10 --lr_decay $lr_decay --gpu 0  --seed 42
done
done



# # Clip
for clip in 0.3 0.5 0.8 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 50 100 10000; do
        python3 -u main_proposed.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip $clip --gpu 0  --seed 42
        python3 -u main_retrain.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip $clip --gpu 0  --seed 42
        python3 -u main_eva.py --model logistic --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip $clip --gpu 0  --seed 42
done

for clip in 0.3 0.5 0.8 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 50 100 10000; do
        python3 -u main_proposed.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05 --clip $clip  --gpu 0  --seed 42
        python3 -u main_retrain.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget  200 --lr 0.05  --clip $clip  --gpu 0  --seed 42
        python3 -u main_eva.py --model cnn --dataset mnist  --epochs 20 --num_dataset 1000 --batch_size 64 --num_forget 200 --lr 0.05 --clip $clip  --gpu 0  --seed 42
done