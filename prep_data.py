import argparse
import numpy as np
from data_utils import read_exclude_segs_file, read_process_data

# read in arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--network", help='network - "full" or "subset"',
                    choices=['full', 'subset'], default='full')
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument("-f", "--finetune_vars", help='whether to finetune on\
                    temp, flow, or both', choices=['temp', 'flow', 'both'])
parser.add_argument("-p", "--pretrain_vars", help='whether to pretrain on\
                    temp, flow, or both', choices=['temp', 'flow', 'both'])
parser.add_argument('-s', '--start_tst_date', help='date when test period \
                    starts', default='2004-09-30')
parser.add_argument('-y', '--y_tst_year', help='number of years in test set',
                    default=12, type=int)
parser.add_argument('-i', '--input_data_dir', help='directory where input data\
                    are located')
parser.add_argument("-d", "--dist_matrix", help='which type of distance matrix\
                    to use', choices=['upstream', 'downstream', 'updown'],
                    default='upstream')
parser.add_argument("-t", "--tag", help='tag to append to end of file',
                    default='')
parser.add_argument("-x", "--exclude_segs_file", help='yml file that contains\
                    segments to exclude', default=None)
args = parser.parse_args()

in_data_dir = args.input_data_dir
network = args.network
out_dir = args.outdir
pt_vars = args.pretrain_vars
ft_vars = args.finetune_vars
dist_mat = args.dist_matrix
if args.exclude_segs_file:
    exclude_segs = read_exclude_segs_file(args.exclude_segs_file)
else:
    exclude_segs = []
tag = args.tag
if tag != '':
    tag = f'_{tag}'


if network == "full":
    subset = False
elif network == "subset":
    subset = True

# set up model/read in data
data = read_process_data(in_data_dir,
                         subset=subset,
                         pretrain_out_vars=pt_vars,
                         finetune_out_vars=ft_vars,
                         dist_type=dist_mat,
                         test_start_date=args.y_tst_year,
                         exclude_segs=exclude_segs)

np.savez_compressed(f'{out_dir}processed_data.npz')
