import math
import torch
import random
import stanza
import torch.utils.data as data
from copy import deepcopy
from typing import Tuple, List


class SentimentDataset(data.Dataset):
    def __init__(self, dataset_name, batchsize):
        super().__init__()
        if dataset_name != "stimuli":
            # We preprocessed SST-7 and wrote it to file prior to training
            self.sentences, self.labels = \
                zip(*[line.split("\t")
                      for line in open(f"../sst7/{dataset_name}.tsv",
                                       encoding="utf-8")])
            self.labels = [int(label) for label in self.labels]
        else:
            self.pipeline = stanza.Pipeline(lang='en', processors='tokenize')

            stimuli = [
                self.preprocess(line).lower()
                for line in open("../noncompsst/stimuli_flat.txt",
                                 encoding="utf-8")]
            self.sentences = stimuli
            # Dummy labels, the stimuli are a test-only dataset
            self.labels = [0 for _ in self.sentences]

        self.num_batches = math.ceil(len(self.labels) / batchsize)
        self.size = len(self.labels)
        self.batchsize = batchsize
        self.order = list(range(len(self.labels)))

    def preprocess(self, s: str) -> str:
        """
        SST was tokenised, tokenise stimuli to match that formatting
        Args:
            - s (string): phrase to tokenise
        Returns:
            - string that separates tokens with whitespace
        """
        doc = self.pipeline(s)
        return ' '.join([
            word.text for sent in doc.sentences for word in sent.words])

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Get a single input/output pair.
        Args:
            - index (int)
        Returns:
            - tuple of a sentence (str) and a sentiment label (int)
        """
        sentence = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        return sentence, label

    def get_batch(self, index: int) -> Tuple[List[str], torch.LongTensor]:
        """
        Collect examples based on the batch number
        Args:
            - index (int): index of *batch* not example
        Returns:
            - tuple of a list of sentences and tensor with sentiment labels
        """
        index = index * self.batchsize
        batch = []
        for i in range(index, index + self.batchsize):
            if i < self.size:
                batch.append(self.__getitem__(self.order[i]))
        sentences, labels = zip(*batch)
        labels = torch.LongTensor(labels)
        return sentences, labels

    def __len__(self):
        return self.size

    def shuffle(self):
        random.shuffle(self.order)
