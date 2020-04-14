import argparse
from RGCN_tf2 import RGCNModel
import numpy as np
from postproc_utils import predict_evaluate


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument("-t", "--tag", help='tag to append to end of output files',
                    default='')
parser.add_argument('-i', "--input_data_file", help='data file [something].npz')
parser.add_argument("-w", "--weights_dir", help='directory where\
                    trained_weights_{tag}/ is')
args = parser.parse_args()

hidden_size = 20
in_data_file = args.input_data_file
outdir = args.outdir
weights_dir = args.weights_dir
run_tag = args.tag
if run_tag != '':
    run_tag = f'_{run_tag}'

data = np.load(in_data_file)
num_segs = data['dist_matrix'].shape[0]
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'])

model.load_weights(f'{weights_dir}/trained_weights{run_tag}/')

predict_evaluate(model, data, 'tst', num_segs, run_tag, outdir)
predict_evaluate(model, data, 'trn', num_segs, run_tag, outdir)
