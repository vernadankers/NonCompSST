import torch
import copy
import numpy as np
import sklearn.metrics
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from typing import Tuple, List, Dict
from .utils import report
from .dataset import SentimentDataset


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device,
                 dataset, logger):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.criterion = criterion
        self.max_epochs = args.epochs
        self.logger = logger
        if args.scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(
                    dataset.num_batches * args.epochs * args.warmup),
                num_training_steps=dataset.num_batches * args.epochs)
        else:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(
                    dataset.num_batches * args.epochs * args.warmup),
                num_training_steps=dataset.num_batches * args.epochs)

    def step(self, batch: Tuple[List[str], torch.LongTensor]) -> \
            Tuple[float, torch.FloatTensor]:
        """
        Process one batch with the model.
        Args:
            - batch (tuple of inputs and labels)
        Returns:
            - loss (float)
            - logits (torch.FloatTensor)
        """
        sentence, target = batch
        output = self.model(sentence)
        target = target.to(self.device)
        loss = self.criterion(output, target)
        return loss, output

    def test(self, dataset: SentimentDataset, disable_tqdm: bool) -> \
            Tuple[Dict, List[int], List[int]]:
        """
        Evaluate model on given dataset (no gradients).
        Args:
            - dataset (SentimentDataset)
            - disable_tqdm (bool): whether to print progression with tqdm
        Returns:
            - dict with performance metrics
            - predicted outputs
            - gold labels of examples considered
        """
        outputs, losses, labels = [], [], []
        self.model.eval()
        with torch.no_grad():
            desc = 'Testing epoch  ' + str(self.epoch + 1) + ''
            for idx in tqdm(range(dataset.num_batches), desc=desc,
                            disable=disable_tqdm):
                batch = dataset.get_batch(idx)
                loss, output = self.step(batch)
                outputs.extend(torch.argmax(output, dim=-1).tolist())
                losses.append(loss.item())
                labels.extend(batch[1].to('cpu').tolist())

        metrics = {
            "loss": np.mean(losses),
            "accuracy": sklearn.metrics.accuracy_score(labels, outputs),
            "f1": sklearn.metrics.f1_score(labels, outputs, average="macro")
        }
        return metrics, outputs, labels

    def train(self, train_dataset: SentimentDataset,
              dev_dataset: SentimentDataset, disable_tqdm: bool) -> Dict:
        """
        Train SST-7 train (unrolled) while selecting the best checkpoint
        on the validation dataset.
        Args:
            - train_dataset (SentimentDataset)
            - dev_dataset (SentimentDataset)
            - disable_tqdm (bool): whether to print progression with tqdm
        Returns:
            - state dict of best checkpoint (dict)
        """
        best_score = 0
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            train_dataset.shuffle()
            self.model.train()
            self.optimizer.zero_grad()
            indices = torch.randperm(train_dataset.num_batches, device='cpu')

            desc = 'Training epoch ' + str(self.epoch + 1) + ''
            losses = []
            for idx in tqdm(indices, desc=desc, disable=disable_tqdm):
                loss, output = self.step(train_dataset.get_batch(idx))
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            self.logger.info(
                f"Epoch {epoch + 1}, training loss: {np.mean(losses):.2f}")
            with torch.no_grad():
                dev_performance = self.test(dev_dataset, disable_tqdm)[0]
                report(self.logger, "dev", dev_performance, epoch + 1)
                # Save the checkpoint and update the best performing model
                if dev_performance["accuracy"] > best_score:
                    best_score = dev_performance["accuracy"]
                    best_checkpoint = copy.deepcopy(self.model.state_dict())
                train_performance = self.test(train_dataset, disable_tqdm)[0]
                report(self.logger, "train", train_performance, epoch + 1)
        return best_checkpoint
