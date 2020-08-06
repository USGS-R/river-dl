import os
import argparse
from river_dl.train import train_model
from river_dl.postproc_utils import predict, all_overall


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should'
                                           'be written')
parser.add_argument("-i", "--in_data", help='the input data file')
parser.add_argument("-u", "--hidden_units", help='num of hidden units',
                    type=int, default=20)
parser.add_argument("-m", "--metrics_file", help='file that contains the'
                                                 'evaluation metrics')
parser.add_argument("-p", "--pretrain_epochs", help='number of pretrain'
                    'epochs', type=int)
parser.add_argument("-f", "--finetune_epochs", help='number of finetune'
                    'epochs', type=int)
parser.add_argument("-q", "--flow-in-temp", help='whether or not to do flow\
                    in temp', action='store_true') 

args = parser.parse_args()
flow_in_temp = args.flow_in_temp
in_data_file = args.in_data
hidden_units = args.hidden_units
out_dir = args.outdir
pt_epochs = args.pretrain_epochs
ft_epochs = args.finetune_epochs

# -------- train ------
model = train_model(in_data_file, pt_epochs, ft_epochs, hidden_units,
                    out_dir=out_dir, flow_in_temp=flow_in_temp)

