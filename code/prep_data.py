import os
import argparse
from preproc_utils import prep_data

# read in arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outfile", help='file where the output should be'
                                         'written')
parser.add_argument("-f", "--finetune_vars", help='whether to finetune on\
                    temp, flow, or both', choices=['temp', 'flow', 'both'],
                    default='both')
parser.add_argument("-p", "--pretrain_vars", help='whether to pretrain on\
                    temp, flow, or both', choices=['temp', 'flow', 'both'],
                    default='both')
parser.add_argument("-x", "--exclude_segs_file", help='yml file that contains\
                    segments to exclude', default=None)
parser.add_argument("-l", "--log_q", help='whether or not to log discharge for\
                    training', action='store_true')
args = parser.parse_args()

out_file = args.outfile
pt_vars = args.pretrain_vars
ft_vars = args.finetune_vars
log_q = args.log_q
exclude_file = args.exclude_segs_file
data_dir = '../data/in/'
obs_temper = os.path.join(data_dir, 'obs_temp_full')
obs_flow = os.path.join(data_dir, 'obs_flow_full')
sntemp = os.path.join(data_dir, 'uncal_sntemp_input_output')
distfile = os.path.join(data_dir, 'distance_matrix.npz')
x_vars =['seg_rain', 'seg_tave_air', 'seginc_swrad', 'seg_length',
         'seginc_potet', 'seg_slope', 'seg_humid', 'seg_elev']
test_start = '2004-09-30'
test_yrs = 12

# set up model/read in data
data = prep_data(obs_temper_file=obs_temper, obs_flow_file=obs_flow,
                 pretrain_file=sntemp, distfile=distfile, x_vars=x_vars,
                 pretrain_vars=pt_vars, finetune_vars=ft_vars,
                 n_test_yr=test_yrs, test_start_date=test_start,
                 exclude_file=exclude_file, log_q=log_q, out_file=out_file)

