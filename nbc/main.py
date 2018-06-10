import argparse
import numpy as np
import operator
import pandas as pd
import torch
from torch.distributions import Multinomial
import typing
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("-filename")

args = parser.parse_args()

dat = pd.read_csv(
        args.filename,
        header=None,
        delim_whitespace=True,
        names=[
            "mpg", "cylinders", "displacement",
            "horsepower", "weight", "acceleration",
            "model year", "origin", "car name"]
        )


def normalize_column(column_name: str, df: pd.DataFrame) -> typing.Tuple[
        float, float]:
    if df[column_name].dtype == "object":
        df[column_name] = pd.to_numeric(df[column_name])
    stats = df[column_name].astype(float).describe()
    mean = stats["mean"]
    std = stats["std"]
    df[column_name] = (df[column_name] - mean) / std
    return (mean, std)


def remove_rows_with_val_in_col(val: object, col: str, df: pd.DataFrame):
    rows = df.loc[df[col] == val]
    df.drop(labels=rows.index, inplace=True)


def remove_cols(cols: typing.List[str], df: pd.DataFrame):
    df.drop(columns=cols, inplace=True)


mpg_stats = normalize_column("mpg", dat)
displacement_stats = normalize_column("displacement", dat)
remove_rows_with_val_in_col('?', "horsepower", dat)
horsepower_stats = normalize_column("horsepower", dat)
weight_stats = normalize_column("weight", dat)
acceleration_stats = normalize_column("acceleration", dat)

# Goal: Predict horsepower from the rest of the data by naively assuming they
# are independent

remove_cols(["model year", "origin", "car name"], dat)

train = dat.sample(frac=0.80, random_state=np.random.RandomState(seed=0))
test = dat.drop(labels=train.index)

cyl_categories = sorted(dat["cylinders"].unique().tolist())

print(f"Cylinder categories: {cyl_categories}")

print(f"Training on {train.shape[0]} points")
print(f"Testing on {test.shape[0]} points")

dists: typing.Dict[int, typing.Dict[str, torch.distributions.Distribution]] = \
        {}
for cyl in cyl_categories:
    dists[cyl] = {}
    mpg_cyl_stats = train[train["cylinders"] == cyl]["mpg"].describe()
    disp_cyl_stats = train[train["cylinders"] == cyl]["displacement"] \
        .describe()
    hp_cyl_stats = train[train["cylinders"] == cyl]["horsepower"].describe()
    weight_cyl_stats = train[train["cylinders"] == cyl]["weight"].describe()
    acc_cyl_stats = train[train["cylinders"] == cyl]["acceleration"].describe()
    dists[cyl]["mpg"] = torch.distributions.Normal(
            mpg_cyl_stats["mean"],
            mpg_cyl_stats["std"])
    dists[cyl]["displacement"] = torch.distributions.Normal(
            disp_cyl_stats["mean"],
            disp_cyl_stats["std"])
    dists[cyl]["horsepower"] = torch.distributions.Normal(
            hp_cyl_stats["mean"],
            hp_cyl_stats["std"])
    dists[cyl]["weight"] = torch.distributions.Normal(
            weight_cyl_stats["mean"],
            weight_cyl_stats["std"])
    dists[cyl]["acceleration"] = torch.distributions.Normal(
            acc_cyl_stats["mean"],
            acc_cyl_stats["std"])

probs: typing.Dict[
            int,
            typing.Dict[int, float]] = {}
for entry in test.itertuples():
    for cyl in cyl_categories:
        prob = dists[cyl]["mpg"].log_prob(entry[1]) * \
                dists[cyl]["displacement"].log_prob(entry[3]) * \
                dists[cyl]["horsepower"].log_prob(entry[4]) * \
                dists[cyl]["weight"].log_prob(entry[5]) * \
                dists[cyl]["acceleration"].log_prob(entry[6])
        if entry[0] not in probs:
            probs[entry[0]] = {}
        probs[entry[0]][cyl] = torch.exp(prob)

count = 0
correct = 0
for entry in probs:
    prediction = max(probs[entry].items(), key=operator.itemgetter(1))[0]
    actual = test.loc[entry]["cylinders"]
    if np.isclose(prediction, actual):
        correct += 1
    count += 1
pct = correct / count * 100
print(f"percent of correct predictions: {pct:.2f}%")
