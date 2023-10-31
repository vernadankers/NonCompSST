import argparse
import numpy as np
import pathlib
from collections import defaultdict
import pickle
import math


def create_ranking(folder, outfile):
    sentiment_prds = defaultdict(list)
    ranking_max = dict()
    ranking_max_abs = dict()
    ranking_separate = dict()
    ranking_separate_abs = dict()
    ranking_clean = dict()
    stimuli = open("../noncompsst/stimuli_flat.txt",
                   encoding='utf-8').readlines()
    print(len(stimuli))
    desktop = pathlib.Path(folder)
    for fn in list(desktop.rglob("*")):
        if "noncompsst.txt" in str(fn):
            prds = [float(l.strip())
                    for l in open(str(fn)).readlines()]
            print(fn)
            for s, p in zip(stimuli, prds):
                sentiment_prds[s.strip()].append(p)

    for k, v in sentiment_prds.items():
        sentiment_prds[k] = np.mean(v)
    sentiment_prds = dict(sentiment_prds)

    with open("../noncompsst/stimuli.tsv", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            [_, p1, _, p2, rp1, rp2, rp3] = lines[i].strip().split('\t')
            natural = sentiment_prds[f"{p1} {p2}"]
            synth1 = sentiment_prds[f"{rp1} {p2}"]
            synth2 = sentiment_prds[f"{rp2} {p2}"]
            synth3 = sentiment_prds[f"{rp3} {p2}"]
            diff1 = natural - np.mean([synth1, synth2, synth3])

            [_, p1, _, p2, rp1, rp2, rp3] = lines[i+1].strip().split('\t')
            synth1 = sentiment_prds[f"{p1} {rp1}"]
            synth2 = sentiment_prds[f"{p1} {rp2}"]
            synth3 = sentiment_prds[f"{p1} {rp3}"]
            diff2 = natural - np.mean([synth1, synth2, synth3])
            ranking_separate[f"<left> {p1} {p2}"] = diff1
            ranking_separate[f"<right> {p1} {p2}"] = diff2
            ranking_separate_abs[f"<left> {p1} {p2}"] = abs(diff1)
            ranking_separate_abs[f"<right> {p1} {p2}"] = abs(diff2)
            ranking_max[f"{p1} {p2}"] = diff1 if abs(
                diff1) > abs(diff2) else diff2
            ranking_max_abs[f"{p1} {p2}"] = max([abs(diff1), abs(diff2)])

    ungrammaticality = pickle.load(
        open("../noncompsst/ungrammaticality.pickle", 'rb'))
    sentiment_prds2 = dict()
    for k, v in sentiment_prds.items():
        if ungrammaticality[k]:
            sentiment_prds2[k] = math.nan
        else:
            sentiment_prds2[k] = sentiment_prds[k]
    sentiment_prds2 = dict(sentiment_prds2)

    with open("../noncompsst/stimuli.tsv", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            [_, p1, _, p2, rp1, rp2, rp3] = lines[i].strip().split('\t')
            if f"{p1} {p2}" not in sentiment_prds2:
                continue
            natural = sentiment_prds2[f"{p1} {p2}"]
            synth1 = sentiment_prds2[f"{rp1} {p2}"]
            synth2 = sentiment_prds2[f"{rp2} {p2}"]
            synth3 = sentiment_prds2[f"{rp3} {p2}"]
            if math.isnan(natural) or (math.isnan(synth1) and math.isnan(synth2) and math.isnan(synth3)):
                continue
            diff1 = natural - np.nanmean([synth1, synth2, synth3])

            [_, p1, _, p2, rp1, rp2, rp3] = lines[i+1].strip().split('\t')
            synth1 = sentiment_prds2[f"{p1} {rp1}"]
            synth2 = sentiment_prds2[f"{p1} {rp2}"]
            synth3 = sentiment_prds2[f"{p1} {rp3}"]
            if math.isnan(synth1) and math.isnan(synth2) and math.isnan(synth3):
                continue
            diff2 = natural - np.nanmean([synth1, synth2, synth3])
            ranking_clean[f"<left> {p1} {p2}"] = diff1
            ranking_clean[f"<right> {p1} {p2}"] = diff2

    pickle.dump((ranking_max, ranking_max_abs, ranking_separate, \
        ranking_separate_abs, ranking_clean, sentiment_prds),
        open(outfile, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--outfile")
    args = parser.parse_args()

    create_ranking(args.folder, args.outfile)
