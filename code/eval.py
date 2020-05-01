import argparse
from postproc_utils import calc_metrics, reach_specific_metrics


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument("-t", "--tag", help='tag to append to end of output files',
                    default='')
parser.add_argument('-p', "--pred_data_file", help='file with predictions')
parser.add_argument('-T', "--temp_obs_file",
                    help='temperature observation file')
parser.add_argument('-Q', "--flow_obs_file", help='flow observation file')
parser.add_argument('-s', "--section_data", help='the section of the data\
                    (test or train)', choices=['tst', 'trn'])
args = parser.parse_args()


outdir = args.outdir
pred_file = args.pred_data_file
temp_obs_file = args.temp_obs_file
flow_obs_file = args.flow_obs_file
run_tag = args.tag
partition = args.section_data
if run_tag != '':
    run_tag = f'_{run_tag}'

calc_metrics(pred_file, temp_obs_file, flow_obs_file, outdir, partition, run_tag)
reach_specific_metrics(pred_file, temp_obs_file, flow_obs_file, outdir, partition, run_tag)
