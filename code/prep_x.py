import os
import argparse
import numpy as np
from preproc_utils import prep_x

# read in arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outfile", help='file where the output should be'
                                            'written')
parser.add_argument('-s', '--start_tst_date', help='date when test period \
                    starts', default='2004-09-30')
parser.add_argument('-v', '--variables', help='input variables')
parser.add_argument('-y', '--y_tst_year', help='number of years in test set',
                    default=12, type=int)
parser.add_argument('-i', '--infile', help='input data file are located')
args = parser.parse_args()

in_data_dir = args.infile
network = args.network
outfile = args.outfile


# set up model/read in data
data = prep_x(in_data_dir, n_test_yr=args.y_tst_year, test_start_date=args.start_tst_date, out_file=outfile)

