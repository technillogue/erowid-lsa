#!/usr/bin/env -S /usr/bin/python3.8 -m poetry run python
# pylint: disable=import-outside-toplevel, wrong-import-order
from collections import Counter
from typing import List, Iterator, Optional
from pathlib import Path
from timeit import default_timer
import pandas as pd
import numpy as np
from pdb import pm, set_trace


class Timer:
    def __init__(self) -> None:
        self.checkpoints = [default_timer()]
        print("started timer")

    def checkpoint(self, msg: str) -> None:
        self.checkpoints.append(default_timer())
        diff = round(self.checkpoints[-1] - self.checkpoints[-2], 3)
        print(f"{msg} (elapsed time: {diff}s)")


timer = Timer()


class ReportFinder:
    def __init__(
        self,
        root: str = "experiences",
        limit: Optional[int] = None,
        common: Optional[int] = 15,
    ) -> None:
        self.root = root
        self.labels: List[str] = []
        self.common = common
        self.limit = limit

    def search(self) -> Iterator[str]:
        categories = Counter(
            category
            for category in Path(self.root).iterdir()
            for item in category.iterdir()
        ).most_common(self.common)
        import random

        random.shuffle(categories)
        for category, _ in categories:
            for report in category.iterdir():
                if report.is_file() and report.stat().st_size > 0:
                    self.labels.append(category.name)
                    yield report.absolute().as_posix()
                if self.limit and len(self.labels) == self.limit:
                    return


def vectorize(words: List[str], method: str = "manual") -> np.matrix:
    if method == "zeugma":
        from zeugma.embeddings import EmbeddingTransformer

        return EmbeddingTransformer().transform(words)
    with open(
        "/home/sylv/gensim-data/glove-850b-300d/glove.840B.300d.txt",
        encoding="utf-8",
    ) as glove:
        matrix = np.zeros((len(words), 300))
        w2v = {}
        for i, word in enumerate(words):
            while word not in w2v:
                line = glove.readline()
                if not line:
                    break
                token, *vec = line.split(" ")
                w2v[token] = np.array([float(n) for n in vec])
            if word in w2v:
                matrix[i] = w2v[word]
    return matrix


def vectorize_reports(reporter: ReportFinder) -> pd.DataFrame:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    counter = CountVectorizer(
        input="filename", decode_error="replace", strip_accents="ascii", min_df=3
    )

    weigher = TfidfTransformer()

    timer.checkpoint("loaded vectorizer")
    counts = counter.fit_transform(
        reporter.search()
    )  # shape: (n_reports, n_words)
    timer.checkpoint(
        "counted {} experience reports. vocabulary: {} words".format(
            *counts.shape
        )
    )
   # sort_by_freq_indices = counts.sum(axis=0).argsort()
  #  counts = counts[
  #      :,sort_by_freq_indices[::-1]
   # ]  # sort in descending order of overall frequency
    # in principle, if both the w2v model and vectorized words are sorted in the same order
    # you don't have to load the model in memory at all....
    word_weights = weigher.fit_transform(counts)  # shape: (n_reports, n_words)
    timer.checkpoint("TF-IDF transformed")
    total_report_weights = word_weights.sum(axis=1)  # shape: (n_reports)
    words = counter.get_feature_names()
    # words = words[   sort_by_freq_indices[::-1]
    #]  # shape: (n_words)
    word_vectors: List = vectorize(words)
    # shape: (n_words, n_dimensions)
    timer.checkpoint(f"vectorized words into {len(word_vectors[0])} dimensions")
    sum_report_vectors = word_weights * word_vectors
    # shape: (n_reports, n_dimensions)
    avg_report_vectors = sum_report_vectors / total_report_weights
    # same shape; not 100% certain that division is necessary
    timer.checkpoint(
        "overall report vector mean: {}. overall SD: {}".format(
            avg_report_vectors.mean(), avg_report_vectors.std()
        )
    )
    df = pd.DataFrame(avg_report_vectors)
    df["labels"] = reporter.labels
    return df


def scale(vectors: pd.DataFrame, cosine=False) -> pd.DataFrame:
    if "labels" in vectors:
        labels = vectors["labels"]
        vectors = vectors.loc[:, vectors.columns != "labels"]
    from sklearn.manifold import MDS
    if cosine:
    # i thought word2vec was optimized for cosine similarity but that doesn't seem be as useful
        from sklearn.metrics.pairwise import cosine_similarity
        dissimilarity = cosine_similarity(vectors)
        scaler = MDS(n_components=2, dissimilarity="precomputed")
        scaled = scaler.fit_transform(dissimilarity)
    else:
        scaler = MDS(n_components=2)
        scaled = scaler.fit_transform(vectors)
    timer.checkpoint("scaled into 2d")
    df = pd.DataFrame(scaled, columns=["x", "y"])
    df["labels"] = labels
    return df


def plot(df: pd.DataFrame) -> None:
    from plotnine import ggplot, aes, geom_point
    from matplotlib import use

    fig = ggplot(df, aes(x="x", y="y", color="collapsed_labels")) + geom_point()
    fig.save(
        "experience_report_dissimilarity_by_category.png", width=10, height=10
    )
    use("TkAgg")
    fig.draw()
    timer.checkpoint("plotted")
    return fig


try:
    report_vectors_2d = pd.read_csv("report_vectors_2d.csv")
except FileNotFoundError:
    try:
        report_vectors = pd.read_csv("report_vectors.csv")
    except FileNotFoundError:
        report_vectors = vectorize_reports(ReportFinder(limit=None, common=None))
        report_vectors.to_csv("report_vectors.csv")
    report_vectors_2d = scale(report_vectors)
    report_vectors_2d.to_csv("report_vectors_2d.csv")


collapse = True # necessary only if common=None
if collapse:
     label_counts = Counter(report_vectors_2d["labels"])
     kept_labels = next(zip(*label_counts.most_common(15)))
     report_vectors_2d["collapsed_labels"] = [
        label if label in kept_labels else "other"
        for label in report_vectors_2d["labels"]
     ]
     timer.checkpoint("collapsed categories")
else:
    report_vectors_2d["collapsed_labels"] = report_vectors_2d["labels"]
plot(report_vectors_2d)
