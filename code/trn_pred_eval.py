import os
import argparse
from train import train_model
from postproc_utils import predict, all_overall


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should'
                                           'be written')
parser.add_argument("-i", "--in_data", help='the input data file')
parser.add_argument("-h", "--hidden_units", help='num of hidden units',
                    type=int, default=20)
parser.add_argument("-m", "--metrics_file", help='file that contains the'
                                                 'evaluation metrics')

args = parser.parse_args()
in_data_file = args.in_data
hidden_units = args.hidden_units
out_dir = args.outdir
pt_epochs = 200
ft_epochs = 100

# -------- train ------
model = train_model(in_data_file, pt_epochs, ft_epochs, hidden_units,
                    out_dir=out_dir)

# -------- predict ------
partitions = ['trn', 'tst']
variables = ['temp', 'flow']
preds_trn = predict(model, in_data_file, 'trn', 'preds_trn.feather')
preds_tst = predict(model, in_data_file, 'tst', 'preds_tst.feather')

# --------- evaluate -----
eval_file = args.metrics_file
obs_dir = "../data/in"
obs_temp = os.path.join(obs_dir, 'obs_temp_full')
obs_flow = os.path.join(obs_dir, 'obs_flow_full')
all_metrics = all_overall(preds_trn, preds_tst, obs_temp, obs_flow,
                          outfile=eval_file)
