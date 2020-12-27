#!/usr/bin/env -S /usr/bin/python3.8 -m poetry run python
from collections import Counter
from typing import List, Tuple, Iterator, Optional
from pathlib import Path
from timeit import default_timer
import pandas as pd

class Timer:
    def __init__(self) -> None:
        self.checkpoints = [default_timer()]
        print("started timer")

    def checkpoint(self, msg: str) -> None:
        self.checkpoints.append(default_timer())
        diff = round(self.checkpoints[-1] - self.checkpoints[-2], 3)
        print(f"{msg} (elapsed time: {diff}s")


timer = Timer()

class ReportFinder:
    def __init__(self, root: str = "experiences") -> None:
        self.root = root
        self.labels: List[str] = []

    def search(self, limit: Optional[int] = None) -> Iterator[str]:
        for category in Path(self.root).iterdir():
            for report in category.iterdir():
                if report.is_file() and report.stat().st_size > 0:
                    self.labels.append(category.name)
                    yield report.absolute().as_posix()
                if limit and len(self.labels) > limit:
                    return

def vectorize_reports(finder: ReportFinder, limit=None) -> Tuple["np.matrix", List[str]]:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from zeugma.embeddings import EmbeddingTransformer

    counter = CountVectorizer(
        input="filename", decode_error="replace", strip_accents="ascii"
    )

    reporter = ReportFinder("experiences")
    weigher = TfidfTransformer()
    vectorizer = EmbeddingTransformer()

    timer.checkpoint("loaded vectorizer")
    counts = counter.fit_transform(reporter.search(limit))  # shape: (n_reports, n_words)
    timer.checkpoint(
        "counted {} experience reports. vocabulary: {} words".format(*counts.shape)
    )
    word_weights = weigher.fit_transform(counts)  # shape: (n_reports, n_words)
    timer.checkpoint("TF-IDF transformed")
    total_report_weights = word_weights.sum(axis=1)  # shape: (n_reports)
    words = counter.get_feature_names()  # shape: (n_words)

    word_vectors: List = vectorizer.transform(words)
    # shape: (n_words, n_dimensions)
    timer.checkpoint(f"vectorized words into {len(word_vectors[0])} dimensions")
    sum_report_vectors = word_weights * word_vectors
    # shape: (n_reports, n_dimensions)
    report_vectors = sum_report_vectors / total_report_weights
    # same shape; not 100% certain that division is necessary
    timer.checkpoint(
        "overall report vector mean: {}. overall SD: {}".format(
            report_vectors.mean(), report_vectors.std()
        )
    )
    df = pd.DataFrame(report_vectors)
    df["labels"] = reporter.labels
    return df

def scale(report_vectors):
    if "labels" in report_vectors:
        labels = report_vectors["labels"]
        report_vectors = report_vectors.loc[:, report_vectors.columns != "labels"]
    from sklearn.manifold import MDS
    scaler = MDS(n_components=2)
    scaled = scaler.fit_transform(report_vectors)
    timer.checkpoint("scaled into 2d")
    df = pd.DataFrame(scaled, columns = ["x", "y"])
    df["labels"] = labels
    return df

def plot(df: pd.DataFrame) -> None:
    from plotnine import ggplot, aes, geom_point
    from matplotlib import use

    fig = (
        ggplot(df, aes(x="x", y="y", color="collapsed_labels"))
        + geom_point()
    )
    fig.save("experience_report_dissimilarity_by_category.png", width=10, height=10)
    use("TkAgg")
    fig.draw()
    timer.checkpoint("plotted")


try:
    report_vectors_2d = pd.read_csv("report_vectors_2d.csv")
except FileNotFoundError:
    try:
        report_vectors = pd.read_csv("report_vectors.csv")
    except FileExistsError:
        report_vectors = vectorize_reports(ReportFinder())
        report_vectors.to_csv("report_vectors.csv")
    report_vectors_2d = scale(report_vectors)
    report_vectors_2d.to_csv("report_vectors_2d.csv")

label_counts = Counter(report_vectors_2d["labels"])
kept_labels = next(zip(*label_counts.most_common(30)))
report_vectors_2d["collapsed_labels"] = [
    label if label in kept_labels else "other" for label in report_vectors_2d["labels"]
]
timer.checkpoint("collapsed categories")
plot(report_vectors_2d)
