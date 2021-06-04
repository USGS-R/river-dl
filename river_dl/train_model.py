import os
import argparse
from river_dl.train import train_model


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
    "--num_tasks", help="number of tasks (outputs to be predicted)", default=1, type=int
)
parser.add_argument(
    "--lambdas", help="lambdas for weighting variable losses", default=[1, 1], type=list
)


args = parser.parse_args()

# -------- train ------
model = train_model(
    args.in_data_file,
    args.pretrain_epochs,
    args.finetune_epochs,
    args.hidden_units,
    out_dir=args.out_dir,
    num_tasks=args.num_tasks,
    lambdas=args.lambdas,
    seed=args.random_seed,
    learning_rate_ft=args.ft_learn_rate,
    learning_rate_pre=args.pt_learn_rate,
    model_type=args.model,
)
