import pytest
import os
import shutil

from river_dl import preproc_utils
from river_dl import train
from river_dl import loss_functions


def test_finetune_rgcn():
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
        segs=[2007, 2012],
        distfile="../../../drb-dl-model/data/in/distance_matrix.npz",
        x_vars=["seg_rain", "seg_tave_air"],
        y_vars=["temp_c"],
    )

    test_out_dir = 'test_data/test_training_out'
    if os.path.exists(test_out_dir):
        shutil.rmtree(test_out_dir)

    os.mkdir(test_out_dir)

    model = train.train_model(
        io_data=prepped_data,
        finetune_epochs=2,
        pretrain_epochs=0,
        hidden_units=10,
        out_dir='test_data/test_training_out',
        model_type="rgcn",
        seed=2,
        dropout=0.12,
        loss_func=loss_functions.rmse
    )


def test_pretrain_fail():
    prepped_data = preproc_utils.prep_all_data(
        x_data_file="test_data/test_data",
        y_data_file="test_data/obs_temp_flow",
        # pretrain_file="test_data/obs_temp_flow",
        train_start_date="2003-09-15",
        train_end_date="2004-09-16",
        val_start_date="2004-09-17",
        val_end_date="2005-09-18",
        test_start_date="2005-09-19",
        test_end_date="2006-09-20",
        spatial_idx_name="segs_test",
        time_idx_name="times_test",
        x_vars=["seg_rain", "seg_tave_air"],
        y_vars=["temp_c"],
    )

    with pytest.raises(KeyError):
        model = train.train_model(
            io_data=prepped_data,
            pretrain_epochs=2,
            finetune_epochs=2,
            hidden_units=10,
            out_dir='test_data/test_training_out',
            model_type="lstm",
            seed=2,
            dropout=0.12,
            loss_func=loss_functions.rmse_masked_one_var
        )

