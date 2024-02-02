import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames",
        type=str,
        default="raw",
        choices=["raw", "vec"],
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=["none", "all", "rare", "slc"],
        help="Depending what set of training data you want.",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="random",
        choices=["random", "separate"],
        help="Depending on the mode of the experiment.",
    )
    parser.add_argument(
        "--model", type=str, default="alexnet", choices=["alexnet", "resnet", "vgg", "vit"]
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--step_size",
        type=int,
        default=4,
        help="How many epochs before decreasing learning rate (if using a step-down policy).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor for learning rate step-down.",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=100,
        help="How often the loss is displayed.",
    )

    parser.add_argument(
        "--trial",
        action="store_true",
        help="If set, the experiment will be limited to 0.2 of the dataset.",
    )
    # Build options dict
    opt = vars(parser.parse_args())

    return opt
