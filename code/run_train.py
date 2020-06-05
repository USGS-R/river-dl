import argparse
import datetime
import tensorflow as tf
from train import train_model

start_time = datetime.datetime.now()

# Declare constants ######
learning_rate_pre = 0.005
learning_rate_ft = 0.01


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument('-i', "--input_data_file", help='data file [something].npz')
parser.add_argument("-t", "--tag", help='tag to append to end of file',
                    default='')
parser.add_argument("-f", "--finetune_epochs", help='num finetuning epochs',
                    default=100, type=int)
parser.add_argument("-p", "--pretrain_epochs", help='num pretraining epochs',
                    default=200, type=int)
parser.add_argument("-s", "--seed", help='random seed', type=int)
parser.add_argument("-u", "--hidden_units", help='number of hidden units',
                    type=int)


args = parser.parse_args()
epochs_pre = args.pretrain_epochs
epochs_finetune = args.finetune_epochs
if args.seed:
    seed = args.seed
    tf.random.set_seed(seed)
else:
    seed = None

in_data_file = args.input_data_file
out_dir = args.outdir
tag = args.tag
if tag != '':
    tag = f'_{tag}'
hidden_size = args.hidden_units
train_model()
