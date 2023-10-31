# NonCompSST
Repository to accompany the EMNLP 2023 findings short paper that proposes NonCompSST:

```
@inproceedings{dankers2023noncomp,
    title = "Non-Compositionality in Sentiment: New Data and Analyses",
    author = "Dankers, Verna and Lucas, Christopher G",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
}
```

### Data
- The `sst7` folder contains train, validation and test data for the 7-point scale version of the Stanford Sentiment Treebank, but with sentences that match our stimuli removed.
- The `noncomposst` folder contains the stimuli used for Study 1 and 2 (under `stimuli_studyX.tsv`), and the `maxabs_ranking.tsv`, the variant of the ranking that we recommend future work uses when working with our non-compositionality ratings (descriptions of other variants can be found in the paper).

### Model training
- The `sentiment_training` folder contains the implementation used to train models on SST-7, and, afterwards, test on the same stimuli presented to humans in Study 2.
- Run `sentiment_training/scripts/train.sh <model_name>` to train a sentiment model and to use it to predict the sentiment for our NonCompSST stimuli.
- Afterwards, use `python create_ranking.py --folder <folder_name>` to generate a non-compositionality ranking from the models' predictions, where the folder name is the parent folder in which the test set predictions for multiple seeds are located.
