import os
import pickle
import logging
import numpy as np
import torch
from typing import Tuple
from lib import Roberta, SentimentDataset, Trainer, set_seed, report, \
    parse_args


def load_data(batchsize: int) -> Tuple[SentimentDataset, SentimentDataset,
                                       SentimentDataset, SentimentDataset]:
    """
    Prepare a Dataset object, that loads SST-7 sentences & labels from file.

    Args:
        - batchsize (int)
    Returns:
        - tuple of 4 SentimentDatasets
    """
    train_dataset = SentimentDataset("train", batchsize)
    dev_dataset = SentimentDataset("validation", batchsize)
    test_dataset = SentimentDataset("test", batchsize)
    stimuli = SentimentDataset("stimuli", batchsize)
    return train_dataset, dev_dataset, test_dataset, stimuli


def main():
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    # Preload data, report statistics
    train_dataset, dev_dataset, test_dataset, stimuli = load_data(
        args.batchsize)
    logger.info(f"Train/dev/test sizes = {len(train_dataset)}/"
                + f"{len(dev_dataset)}/{len(test_dataset)}/{len(stimuli)}")

    # Prepare model, optimiser, loss function
    model = Roberta(**vars(args)).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"{params} trainable parameters.")
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Create trainer object for training and testing & train
    trainer = Trainer(args, model, criterion, optimizer,
                      device, train_dataset, logger)
    best_checkpoint = trainer.train(
        train_dataset, dev_dataset, args.disable_tqdm)

    # Store and load the best state dict encountered during training
    model.load_state_dict(best_checkpoint)
    #torch.save(model.state_dict(), args.save + "/" + "model.pt")

    # Final evaluation on SST-7 test set, save to file
    for dataset, name in [(test_dataset, "test"), (stimuli, "noncompsst")]:
        fn = os.path.join(args.save, name)
        test_performance, predictions, labels = trainer.test(
            dataset, args.disable_tqdm)
        data = (dataset.sentences, labels, predictions)
        report(logger, name, test_performance)
        with open(fn + ".txt", 'w', encoding="utf-8") as f:
            for p in predictions:
                f.write(f"{p}\n")
        pickle.dump(data, open(fn + ".pickle", 'wb'))


if __name__ == "__main__":
    main()
