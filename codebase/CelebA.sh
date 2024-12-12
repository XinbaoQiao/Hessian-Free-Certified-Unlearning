for algorithm in main_proposedresnet.py main_retrain.py main_finetune.py main_neggrad.py main_eva.py; do
python3 -u $algorithm --model resnet18 --dataset celeba --epochs 5 --num_dataset 10000 --batch_size 64 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.01 --clip 10 --gpu 4 --seed 42
done

python3 -u main_proposed.py --model logistic --application --dataset mnist --epochs 15  --num_dataset 1000 --batch_size 32 --num_forget 200 --lr 0.05  --regularization 0.5 --lr_decay 0.995 --clip 5 --gpu 0  --seed 124