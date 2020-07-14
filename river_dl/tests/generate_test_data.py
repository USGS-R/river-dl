import pandas as pd


def select_data(df, col, selection):
    df = df.set_index(col)
    df = df.loc[selection]
    df.reset_index(inplace=True)
    return df


def sel_date_segs(df, segs, start_date, end_date):
    df = select_data(df, 'date', slice(start_date, end_date))
    df = select_data(df, 'seg_id_nat', segs)
    return df


# need to subset this data so it's just two years and two sites. I think
# such a dataset should be representative enough to run tests against
dfs = pd.read_feather('../../data/in/uncal_sntemp_input_output_subset.feather')
dfs['date'] = pd.to_datetime(dfs['date'])
dft = pd.read_csv('../../data/in/obs_flow_subset.csv', parse_dates=['date'],
                  infer_datetime_format=True)
dfq = pd.read_csv('../../data/in/obs_temp_subset.csv', parse_dates=['date'],
                  infer_datetime_format=True)

start_date = '2004-09-15'
end_date = '2006-10-15'
segs = ['2012', '2007']

dft = sel_date_segs(dft, segs, start_date, end_date)
dfq = sel_date_segs(dfq, segs, start_date, end_date)
dfs = sel_date_segs(dfs, segs, start_date, end_date)

dft.to_csv('test_data/obs_temp_full.csv', index=False)
dfq.to_csv('test_data/obs_flow_full.csv', index=False)
dfs.to_feather('test_data/uncal_sntemp_input_output.feather')
