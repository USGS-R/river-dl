import argparse
from river_dl.train import train_model
import river_dl.loss_functions as lf


def get_loss_func_from_str(loss_func_str, lambdas=None):
    if loss_func_str == "rmse":
        return lf.rmse
    elif loss_func_str == "nse":
        return lf.nse
    elif loss_func_str == "kge":
        return lf.kge
    elif loss_func_str == "multitask_rmse":
        return lf.multitask_rmse(lambdas)
    elif loss_func_str == "multitask_nse":
        return lf.multitask_nse(lambdas)
    elif loss_func_str == "multitask_kge":
        return lf.multitask_kge(lambdas)
    else:
        raise ValueError(f"loss function {loss_func_str} not supported")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-o", "--outdir", help="directory where the output should" "be written"
)
parser.add_argument("-i", "--in_data", help="the input data file")
parser.add_argument(
    "-u", "--hidden_units", help="num of hidden units", type=int, default=20
)
parser.add_argument(
    "-m", "--metrics_file", help="file that contains the" "evaluation metrics"
)
parser.add_argument(
    "-s", "--random_seed", help="random seed", default=None, type=int
)
parser.add_argument(
    "-p", "--pretrain_epochs", help="number of pretrain" "epochs", type=int
)
parser.add_argument(
    "-f", "--finetune_epochs", help="number of finetune" "epochs", type=int
)
parser.add_argument(
    "--pt_learn_rate",
    help="learning rate for pretraining",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--ft_learn_rate",
    help="learning rate for finetuning",
    type=float,
    default=0.01,
)
parser.add_argument(
    "--model",
    help="type of model to train",
    choices=["lstm", "rgcn", "gru"],
    default="rgcn",
)
parser.add_argument(
    "--num_tasks",
    help="number of tasks (variables to be predicted)",
    default=1,
    type=int,
)
parser.add_argument(
    "--loss_func",
    help="loss function",
    default="rmse",
    type=str,
    choices=[
        "rmse",
        "nse",
        "kge",
        "multitask_rmse",
        "multitask_kge",
        "multitask_nse",
    ],
)
parser.add_argument("--dropout", help="dropout rate", default=0, type=float)
parser.add_argument(
    "--recurrent_dropout", help="recurrent dropout", default=0, type=float
)
parser.add_argument(
    "--lambdas",
    help="lambdas for weighting variable losses",
    default=[1, 1],
    type=list,
)

args = parser.parse_args()

loss_func = get_loss_func_from_str(args.loss_func)


# -------- train ------
model = train_model(
    args.in_data_file,
    args.pretrain_epochs,
    args.finetune_epochs,
    args.hidden_units,
    out_dir=args.out_dir,
    num_tasks=args.num_tasks,
    loss_func=loss_func,
    dropout=args.dropout,
    recurrent_dropout=args.recurrent_dropout,
    seed=args.random_seed,
    learning_rate_ft=args.ft_learn_rate,
    learning_rate_pre=args.pt_learn_rate,
    model_type=args.model,
)
