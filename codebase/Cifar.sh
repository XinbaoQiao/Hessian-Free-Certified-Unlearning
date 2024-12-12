for algorithm in main_proposedresnet.py main_retrain.py main_finetune.py main_neggrad.py main_eva.py; do
python3 -u $algorithm --model resnet18 --dataset cifar  --epochs 40 --num_dataset 50000 --batch_size 256 --num_forget  50 --lr 0.001 --regularization 1e-2 --lr_decay 0.99995 --clip 10  --gpu 0  --seed 930
done
