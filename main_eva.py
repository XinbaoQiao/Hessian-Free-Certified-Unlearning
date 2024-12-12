import random
import time
import matplotlib
matplotlib.use('Agg')
from utils.options import args_parser
from utils.Evaluate_Euclidean import Evaluate_Euclidean
from utils.Evaluate_Euclidean import Evaluate_Euclidean_ResNet

args = args_parser()
if args.model in ['resnet18']:
    Evaluate_Euclidean_ResNet(args)
else:
    Evaluate_Euclidean(args)
