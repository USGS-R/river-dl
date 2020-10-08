import os
import argparse
from river_dl.train import train_model


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should'
                                           'be written')
parser.add_argument("-i", "--in_data", help='the input data file')
parser.add_argument("-u", "--hidden_units", help='num of hidden units',
                    type=int, default=20)
parser.add_argument("-m", "--metrics_file", help='file that contains the'
                                                 'evaluation metrics')
parser.add_argument("-s", "--random_seed", help='random seed', default=None,
                    type=int)
parser.add_argument("-p", "--pretrain_epochs", help='number of pretrain'
                    'epochs', type=int)
parser.add_argument("-f", "--finetune_epochs", help='number of finetune'
                    'epochs', type=int)
parser.add_argument("-q", "--flow-in-temp", help='whether or not to do flow\
                    in temp', action='store_true') 
parser.add_argument("--pt_temp_wgt", help='weight for temp rmse in pretraining',
                    type=float, default=0.5) 
parser.add_argument("--ft_temp_wgt", help='weight for temp rmse in finetuning',
                    type=float, default=0.5)
parser.add_argument("--pt_learn_rate", help='learning rate for pretraining',
                    type=float, default=0.005)
parser.add_argument("--ft_learn_rate", help='learning rate for finetuning',
                    type=float, default=0.01)
parser.add_argument("--model", help="type of model to train",
                    choices=['lstm', 'rgcn'], default='rgcn')


args = parser.parse_args()
flow_in_temp = args.flow_in_temp
in_data_file = args.in_data
hidden_units = args.hidden_units
out_dir = args.outdir
pt_epochs = args.pretrain_epochs
ft_epochs = args.finetune_epochs
pt_temp_wgt = args.pt_temp_wgt
ft_temp_wgt = args.ft_temp_wgt

# -------- train ------
model = train_model(in_data_file, pt_epochs, ft_epochs, hidden_units,
                    out_dir=out_dir, flow_in_temp=flow_in_temp,
                    finetune_temp_rmse_weight=ft_temp_wgt,
                    pretrain_temp_rmse_weight=pt_temp_wgt,
                    seed=args.random_seed, learning_rate_ft=args.ft_learn_rate,
                    learning_rate_pre=args.pt_learn_rate, model=args.model)

