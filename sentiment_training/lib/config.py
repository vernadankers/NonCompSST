import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Roberta SST-7 implementation.")

    # data arguments
    parser.add_argument("--save", default="checkpoints/",
                        help="directory to save checkpoints in")

    # model arguments
    parser.add_argument("--model_type", type=str, default="roberta-base")

    # training arguments
    parser.add_argument("--epochs", default=5, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batchsize", default=32, type=int,
                        help="batchsize for optimizer updates")
    parser.add_argument("--lr", default=5e-6, type=float, metavar="LR",
                        help="initial learning rate")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="Whether to use TQDM in command line or not.")
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup", type=float, default=0.1)

    # miscellaneous options
    parser.add_argument("--seed", default=1, type=int,
                        help="random seed (default: 1)")

    args = parser.parse_args()
    return args
