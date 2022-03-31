import pytest
from river_dl import preproc_utils
from river_dl.postproc_utils import prepped_array_to_df

# segments in test dataset
segs = [2012, 2007, 2014, 2037]
# date range in test dataset
min_date = "2003-09-15"
max_date = "2006-10-15"

obs_flow = "test_data/obs_flow.csv"
obs_temp = "test_data/obs_temp.csv"
sntemp = "test_data/test_data"


def test_read_exclude():
    exclude_file = "test_data/exclude_2007.yml"
    ex0 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex0 == [{"seg_id_nats_ex": [2007]}]
    exclude_file = "test_data/exclude1.yml"
    ex1 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex1 == [
        {"seg_id_nats_ex": [2007], "start_date": "2005-03-15"},
        {"seg_id_nats_ex": [2012], "end_date": "2005-03-15"},
    ]


def test_prep_data():
    prepped_data = preproc_utils.prep_all_data(
        x_data_file="test_data/test_data",
        y_data_file="test_data/obs_temp_flow",
        train_start_date="2003-09-15",
        train_end_date="2004-09-16",
        val_start_date="2004-09-17",
        val_end_date="2005-09-18",
        test_start_date="2005-09-19",
        test_end_date="2006-09-20",
        spatial_idx_name="segs_test",
        time_idx_name="times_test",
        x_vars=["seg_rain", "seg_tave_air"],
        y_vars_finetune=["temp_c", "discharge_cms"],
    )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()
    assert "y_obs_vars" in prepped_data.keys()
    assert "y_mean" in prepped_data.keys()
    assert "y_std" in prepped_data.keys()
    assert "x_vars" in prepped_data.keys()
    assert "x_mean" in prepped_data.keys()
    assert "x_std" in prepped_data.keys()


def test_prep_data_w_pretrain():
    prepped_data = preproc_utils.prep_all_data(
        x_data_file="test_data/test_data",
        y_data_file="test_data/obs_temp_flow",
        pretrain_file="test_data/test_data",
        train_start_date="2003-09-15",
        train_end_date="2004-09-16",
        val_start_date="2004-09-17",
        val_end_date="2005-09-18",
        test_start_date="2005-09-19",
        test_end_date="2006-09-20",
        spatial_idx_name="segs_test",
        time_idx_name="times_test",
        x_vars=["seg_rain", "seg_tave_air"],
        y_vars_finetune=["temp_c", "discharge_cms"],
        y_vars_pretrain=["seg_tave_water", "seg_outflow"],
    )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()
    assert "y_obs_vars" in prepped_data.keys()
    assert "y_pre_vars" in prepped_data.keys()
    assert "y_mean" in prepped_data.keys()
    assert "y_std" in prepped_data.keys()
    assert "x_vars" in prepped_data.keys()
    assert "x_mean" in prepped_data.keys()
    assert "x_std" in prepped_data.keys()


def test_prep_data_w_pretrain_file_no_y_pretrain():
    with pytest.raises(ValueError):
        preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/obs_temp_flow",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
        )


def test_prep_data_no_validation_dates():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert prepped_data["x_trn"] is not None
    assert prepped_data["x_val"] is None
    assert prepped_data["x_tst"] is not None
    assert prepped_data["ids_trn"] is not None
    assert prepped_data["ids_val"] is None
    assert prepped_data["ids_tst"] is not None
    assert prepped_data["times_trn"] is not None
    assert prepped_data["times_val"] is None
    assert prepped_data["times_tst"] is not None
    assert prepped_data["y_pre_trn"] is not None
    assert prepped_data["y_obs_trn"] is not None
    assert prepped_data["y_obs_val"] is None
    assert prepped_data["y_obs_tst"] is not None

def test_prep_data_no_test_dates():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert prepped_data["x_trn"] is not None
    assert prepped_data["x_val"] is not None
    assert prepped_data["x_tst"] is None
    assert prepped_data["ids_trn"] is not None
    assert prepped_data["ids_val"] is not None
    assert prepped_data["ids_tst"] is None
    assert prepped_data["times_trn"] is not None
    assert prepped_data["times_val"] is not None
    assert prepped_data["times_tst"] is None
    assert prepped_data["y_pre_trn"] is not None
    assert prepped_data["y_obs_trn"] is not None
    assert prepped_data["y_obs_val"] is not None
    assert prepped_data["y_obs_tst"] is None

def test_prep_data_no_val_end():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                val_start_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
                y_vars_pretrain=["seg_tave_water", "seg_outflow"],
            )

def test_prep_data_no_val_start():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                val_end_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
            )

def test_prep_data_no_test_end():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                test_start_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
            )

def test_prep_data_no_test_start():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                test_end_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
                y_vars_pretrain=["seg_tave_water", "seg_outflow"],
            )

def test_prep_data_no_train():
    with pytest.raises(TypeError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                test_start_date="2003-09-15",
                test_end_date="2004-09-16",
                val_start_date="2003-09-15",
                val_end_date="2004-09-16",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
            )


def assert_segs_in_ids(data):
    assert 2007 in data
    assert 2012 in data
    assert 2014 in data
    assert 2037 in data


def df_from_array(data, partition):
    df = prepped_array_to_df(data[f'y_obs_{partition}'],
                             data[f'times_{partition}'],
                             data[f'ids_{partition}'],
                             col_names = data['y_obs_vars'],
                             spatial_idx_name='segs_test',
                             time_idx_name='times_test',
                           )
    return df


def get_num_non_nans(df, segment):
    df_seg = df.query(f"segs_test == {segment}") 
    df_seg = df_seg.set_index(['segs_test', 'times_test'])
    print(df_seg)
    return df_seg.notna().sum().sum()



def test_prep_data_val_test_sites():
    no_sites_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            val_sites=[2007],
            test_sites=[2012],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert data["ids_tst"] is None
    assert no_sites_data["ids_tst"] is None

    assert_segs_in_ids(no_sites_data['ids_trn'])
    assert_segs_in_ids(no_sites_data['ids_val'])
    assert_segs_in_ids(data['ids_trn'])
    assert_segs_in_ids(data['ids_val'])

    df_sites_trn = df_from_array(data, 'trn')
    df_no_sites_trn = df_from_array(no_sites_data, 'trn')

    assert get_num_non_nans(df_no_sites_trn, 2007) > 0
    assert get_num_non_nans(df_no_sites_trn, 2012) > 0
    assert get_num_non_nans(df_no_sites_trn, 2014) > 0
    assert get_num_non_nans(df_no_sites_trn, 2037) > 0

    assert get_num_non_nans(df_sites_trn, 2007) == 0
    assert get_num_non_nans(df_sites_trn, 2012) == 0
    assert get_num_non_nans(df_sites_trn, 2014) > 0
    assert get_num_non_nans(df_sites_trn, 2037) > 0

    df_sites_val = df_from_array(data, 'val')
    df_no_sites_val = df_from_array(no_sites_data, 'val')

    assert get_num_non_nans(df_no_sites_val, 2007) > 0
    assert get_num_non_nans(df_no_sites_val, 2012) > 0
    assert get_num_non_nans(df_no_sites_val, 2014) > 0
    assert get_num_non_nans(df_no_sites_val, 2037) > 0

    assert get_num_non_nans(df_sites_val, 2007) > 0
    assert get_num_non_nans(df_sites_val, 2012) == 0
    assert get_num_non_nans(df_sites_val, 2014) > 0
    assert get_num_non_nans(df_sites_val, 2037) > 0


def test_prep_data_val_test_sites_test_dates():
    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            val_sites=[2007],
            test_sites=[2012],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )


    assert_segs_in_ids(data['ids_trn'])
    assert_segs_in_ids(data['ids_val'])
    assert_segs_in_ids(data['ids_tst'])

    df_trn = df_from_array(data, 'trn')
    df_val = df_from_array(data, 'val')
    df_tst = df_from_array(data, 'tst')

    assert get_num_non_nans(df_trn, 2007) == 0
    assert get_num_non_nans(df_trn, 2012) == 0
    assert get_num_non_nans(df_trn, 2014) > 0
    assert get_num_non_nans(df_trn, 2037) > 0

    assert get_num_non_nans(df_val, 2007) > 0
    assert get_num_non_nans(df_val, 2012) == 0
    assert get_num_non_nans(df_val, 2014) > 0
    assert get_num_non_nans(df_val, 2037) > 0

    assert get_num_non_nans(df_tst, 2007) > 0
    assert get_num_non_nans(df_tst, 2012) > 0
    assert get_num_non_nans(df_tst, 2014) > 0
    assert get_num_non_nans(df_tst, 2037) > 0


def test_prep_data_just_test_sites():
    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            test_sites=[2012],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )


    assert_segs_in_ids(data['ids_trn'])
    assert_segs_in_ids(data['ids_val'])
    assert_segs_in_ids(data['ids_tst'])

    df_trn = df_from_array(data, 'trn')
    df_val = df_from_array(data, 'val')
    df_tst = df_from_array(data, 'tst')

    assert get_num_non_nans(df_trn, 2007) > 0
    assert get_num_non_nans(df_trn, 2012) == 0
    assert get_num_non_nans(df_trn, 2014) > 0
    assert get_num_non_nans(df_trn, 2037) > 0

    assert get_num_non_nans(df_val, 2007) > 0
    assert get_num_non_nans(df_val, 2012) == 0
    assert get_num_non_nans(df_val, 2014) > 0
    assert get_num_non_nans(df_val, 2037) > 0

    assert get_num_non_nans(df_tst, 2007) > 0
    assert get_num_non_nans(df_tst, 2012) > 0
    assert get_num_non_nans(df_tst, 2014) > 0
    assert get_num_non_nans(df_tst, 2037) > 0


def test_prep_data_just_val_site():
    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            val_sites=[2012],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )


    assert_segs_in_ids(data['ids_trn'])
    assert_segs_in_ids(data['ids_val'])
    assert_segs_in_ids(data['ids_tst'])

    df_trn = df_from_array(data, 'trn')
    df_val = df_from_array(data, 'val')
    df_tst = df_from_array(data, 'tst')

    assert get_num_non_nans(df_trn, 2007) > 0
    assert get_num_non_nans(df_trn, 2012) == 0
    assert get_num_non_nans(df_trn, 2014) > 0
    assert get_num_non_nans(df_trn, 2037) > 0

    assert get_num_non_nans(df_val, 2007) > 0
    assert get_num_non_nans(df_val, 2012) > 0
    assert get_num_non_nans(df_val, 2014) > 0
    assert get_num_non_nans(df_val, 2037) > 0

    assert get_num_non_nans(df_tst, 2007) > 0
    assert get_num_non_nans(df_tst, 2012) > 0
    assert get_num_non_nans(df_tst, 2014) > 0
    assert get_num_non_nans(df_tst, 2037) > 0


def test_prep_data_multi_val_site():
    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            val_sites=[2012, 2037],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )


    assert_segs_in_ids(data['ids_trn'])
    assert_segs_in_ids(data['ids_val'])
    assert_segs_in_ids(data['ids_tst'])

    df_trn = df_from_array(data, 'trn')
    df_val = df_from_array(data, 'val')
    df_tst = df_from_array(data, 'tst')

    assert get_num_non_nans(df_trn, 2007) > 0
    assert get_num_non_nans(df_trn, 2012) == 0
    assert get_num_non_nans(df_trn, 2014) > 0
    assert get_num_non_nans(df_trn, 2037) == 0

    assert get_num_non_nans(df_val, 2007) > 0
    assert get_num_non_nans(df_val, 2012) > 0
    assert get_num_non_nans(df_val, 2014) > 0
    assert get_num_non_nans(df_val, 2037) > 0

    assert get_num_non_nans(df_tst, 2007) > 0
    assert get_num_non_nans(df_tst, 2012) > 0
    assert get_num_non_nans(df_tst, 2014) > 0
    assert get_num_non_nans(df_tst, 2037) > 0


def test_prep_data_multi_test_site():
    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            test_sites=[2012, 2037],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )


    assert_segs_in_ids(data['ids_trn'])
    assert_segs_in_ids(data['ids_val'])
    assert_segs_in_ids(data['ids_tst'])

    df_trn = df_from_array(data, 'trn')
    df_val = df_from_array(data, 'val')
    df_tst = df_from_array(data, 'tst')

    assert get_num_non_nans(df_trn, 2007) > 0
    assert get_num_non_nans(df_trn, 2012) == 0
    assert get_num_non_nans(df_trn, 2014) > 0
    assert get_num_non_nans(df_trn, 2037) == 0

    assert get_num_non_nans(df_val, 2007) > 0
    assert get_num_non_nans(df_val, 2012) == 0
    assert get_num_non_nans(df_val, 2014) > 0
    assert get_num_non_nans(df_val, 2037) == 0

    assert get_num_non_nans(df_tst, 2007) > 0
    assert get_num_non_nans(df_tst, 2012) > 0
    assert get_num_non_nans(df_tst, 2014) > 0
    assert get_num_non_nans(df_tst, 2037) > 0


def test_prep_data_no_scale_y():
    data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            test_sites=[2012, 2037],
            spatial_idx_name="segs_test",
            normalize_y=False,
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    # make sure all the std's are 1 and the means are 0
    assert ((data['y_std'] - 1).sum() == 0)
    assert (data['y_mean'].sum() == 0)
