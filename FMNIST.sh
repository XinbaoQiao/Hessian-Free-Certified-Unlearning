for num_forget in 200 40 400 600 800 1000 1200; do
    python3 -u main_proposed.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_NU.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_IJ.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_retrain.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_eva.py --model cnn --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
done

for num_forget in 200 40  400 600 800 1000 1200; do
    python3 -u main_proposed.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_NU.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_IJ.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_retrain.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
    python3 -u main_eva.py --model lenet --dataset fashion-mnist --epochs 30 --num_dataset 4000 --batch_size 256 --num_forget $num_forget --lr 0.5 --clip 0.5 --gpu 1  --seed 42
done


