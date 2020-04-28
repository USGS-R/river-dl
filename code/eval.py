import argparse
import os
from postproc_utils import calc_metrics


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument("-t", "--tag", help='tag to append to end of output files',
                    default='')
parser.add_argument('-p', "--pred_data_file", help='file with predictions')
parser.add_argument('-d', "--data_obs_file", help='observation file')
parser.add_argument('-s', "--section_data", help='the section of the data\
                    (test or train)', choices=['tst', 'trn'])
args = parser.parse_args()


outdir = args.outdir
pred_file = args.pred_data_file
obs_file = args.data_obs_file
run_tag = args.tag
partition = args.section_data
if run_tag != '':
    run_tag = f'_{run_tag}'

calc_metrics(pred_file, obs_file, outdir, partition, run_tag)
