import torch
import random
import logging
import numpy as np
from typing import Dict


def set_seed(seed: int):
    """
    Set random seed.
    Args:
        - seed (int)
    """
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def report(logger: logging.Logger, dataset: str, metrics: Dict,
           epoch: int = -1):
    """
    Report performance in standard format.
    Args:
        - logger: customised logger object
        - dataset (str): name of fold currently reporting for
        - metrics (dict): contains acc and f1
        - epoch (int): epoch we're reporting for
    """
    metric_messages = []
    if epoch != -1:
        message = f"Epoch {epoch}, {dataset}, "
    else:
        message = f"Testing... {dataset}, "

    for metric in metrics:
        metric_messages.append(f"{metric}: {metrics[metric]:.4f}")

    message += "\t".join(metric_messages)
    logger.info(message)
