for algorithm in main_proposedresnet.py main_retrain.py main_finetune.py main_neggrad.py; do
python3 -u $algorithm --model resnet18 --dataset lfw --epochs 50 --num_dataset 984 --batch_size 40 --num_forget 50 --regularization 1e-2 --lr_decay 0.9995 --lr 0.004 --clip 5.5 --gpu 1 --seed 42
done

