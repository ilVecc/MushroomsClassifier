from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

value_ranges = {
    "class": {
        "e": "edible",
        "p": "poisonous"},
    "cap-shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "f": "flat",
        "k": " knobbed",
        "s": "sunken"},
    "cap-surface": {
        "f": "fibrous",
        "g": "grooves",
        "y": "scaly",
        "s": "smooth"},
    "cap-color": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "r": "green",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"},
    "bruises": {
        "t": "bruises",
        "f": "no"},
    "odor": {
        "a": "almond",
        "l": "anise",
        "c": "creosote",
        "y": "fishy",
        "f": "foul",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy"},
    "gill-attachment": {
        "a": "attached",
        "d": "descending",
        "f": "free",
        "n": "notched"},
    "gill-spacing": {
        "c": "close",
        "w": "crowded",
        "d": "distant"},
    "gill-size": {
        "b": "broad",
        "n": "narrow"},
    "gill-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "g": "gray",
        "r": " green",
        "o": "orange",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"},
    "stalk-shape": {
        "e": "enlarging",
        "t": "tapering"},
    "stalk-root": {
        "b": "bulbous",
        "c": "club",
        "u": "cup",
        "e": "equal",
        "z": "rhizomorphs",
        "r": "rooted",
        "?": "missing"},
    "stalk-surface-above-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"},
    "stalk-surface-below-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"},
    "stalk-color-above-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"},
    "stalk-color-below-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"},
    "veil-type": {
        "p": "partial",
        "u": "universal"},
    "veil-color": {
        "n": "brown",
        "o": "orange",
        "w": "white",
        "y": "yellow"},
    "ring-number": {
        "n": "none",
        "o": "one",
        "t": "two"},
    "ring-type": {
        "c": "cobwebby",
        "e": "evanescent",
        "f": "flaring",
        "l": "large",
        "n": "none",
        "p": "pendant",
        "s": "sheathing",
        "z": "zone"},
    "spore-print-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "r": "green",
        "o": "orange",
        "u": "purple",
        "w": "white",
        "y": "yellow"},
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary"},
    "habitat": {
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
        "d": "woods"}
}


def analyze_data():
    df = pd.read_csv("../mushrooms.csv")
    df = df.drop_duplicates()
    df = df.replace(value_ranges)
    
    # plot class balance
    sns.countplot(x=df["class"], data=df, saturation=1)
    plt.savefig("images/class_balance.png")
    plt.clf()
    
    # plot feature balance between classes
    data = df.drop("class", axis=1)
    classes = df["class"]
    for i, col in enumerate(data.columns):
        sns.countplot(x=data[col], data=data, hue=classes, saturation=1)
        plt.savefig("images/feature_{}_balance.png".format(col))
        plt.clf()


if __name__ == '__main__':
    Path("images").mkdir(parents=True, exist_ok=True)
    analyze_data()
