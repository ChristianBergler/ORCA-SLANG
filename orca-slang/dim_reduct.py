"""
Module: knn_classifier.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import pickle
import argparse
import numpy as np
import seaborn as sns

import torch

import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from collections import OrderedDict

from utils.logging import Logger

parser = argparse.ArgumentParser()

torch.manual_seed(42)


parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

parser.add_argument(
    "--deep_features",
    type=str,
    default=None,
    help="Deep Feature Input File (.p) Including Latent Embeddings and Other Information from previous Clustering.",
)

parser.add_argument(
    "--cluster_data",
    type=str,
    default=None,
    help="Clustering Input File (.p) Including Cluster Centers und Cluster Predictions with respect to the given deep features from the previous Clustering.",
)

parser.add_argument(
    "--label_info",
    type=str,
    default=None,
    help="Mapping cluster numbers to real label names according to any previous applied (multi-class) classification system/model"
)

parser.add_argument(
    "--principle_comp",
    type=int,
    default=5,
    help="Number of Principle Components"
)

parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="The directory to store the logs."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to Store the Final Output."
)

parser.add_argument(
    "--tsne_comp",
    type=str,
    default=None,
    help="Number of t-SNE dimensions."
)

ARGS = parser.parse_args()

log = Logger("DIM-REDUCT", ARGS.debug, ARGS.log_dir)

class DataLoader:
    def __init__(self, file_name, keys, display_names=None):
        self.file_name = file_name
        self.keys = keys
        self.display_names = keys
        if display_names is not None:
            assert len(self.display_names) == len(keys), "If using separate names for displaying, length of names must match length of keys"
            self.display_names = self.display_names

        self.values = self._load_values()
        self.flattened_values = self.flatten_values()
        print("")

    def _load_pickle(self):
        with open(self.file_name, "rb") as f:
            data = pickle.load(f)
        return data

    def _load_values(self):
        data = self._load_pickle()
        values = OrderedDict()
        for idx, key in enumerate(self.keys):
            values[key] = {
                "key": key,
                "value": data[key],
                "display_name": self.display_names[idx]
            }
        return values

    def flatten_values(self):
        values = None

        def _append(v, i):
            if len(i.shape) == 1:
                i = np.expand_dims(i, 0)
            if v is None:
                return i
            v = np.concatenate((v, i), axis=0)
            return v

        for key in self.values.keys():
            dict_value = self.values[key]["value"]
            if type(dict_value) == list:
                for item in dict_value:
                    values = _append(values, item)
            elif type(dict_value) == dict:
                for k in dict_value.keys():
                    values = _append(values, dict_value[k])
            else:
                values = _append(values, dict_value)
        return values


class Dim_Reduction:
    def __init__(self, pca_k, tsne_d, pca_q=25, pca_first=True):
        self.pca_k = pca_k
        self.tnse_d = tsne_d
        self.pca_q = pca_q
        self.pca_first = pca_first

    def pca(self, data):
        if not type(data) is Tensor:
            data = torch.from_numpy(data)
        (U, S, V) = torch.pca_lowrank(data, q=self.pca_q)
        reduced = torch.matmul(data, V[:, :self.pca_k])
        return reduced

    def tsne(self, data):
        embedded_features = TSNE(init='pca', random_state=42, n_components=self.tnse_d).fit_transform(data)
        return embedded_features

    def __call__(self, data):
        if self.pca_first:
            data = self.pca(data)
        data = self.tsne(data)
        return data


class Visualization:
    def __init__(self, output_directory=None, palette=None):
        self.output_directory = output_directory
        self.palette = palette

    def visualize2d(self, df: DataFrame, title=None, legend=True, label_axis=False, hue_key="call", x_key="x", y_key="y"):
        if self.palette is None:
            self.palette = sns.color_palette("Spectral")

        figure, ax = plt.subplots(dpi=600)

        scatter = sns.scatterplot(data=df, x=x_key, y=y_key, hue=hue_key, alpha=0.7, legend=legend)
        if title is not None:
            plt.title(title)
        if legend:
            figure.subplots_adjust(right=0.8)
            plt.legend(bbox_to_anchor=(1, 0.9))
        if not label_axis:
            scatter.set(xlabel=None)
            scatter.set(ylabel=None)
        if self.output_directory is not None:
            title = title if title is not None else "dim_reduction"
            plt.savefig(os.path.join(self.output_directory, f"{title}.png"), bbox_inches="tight")
            plt.close(figure)
        else:
            plt.show()


def load_cluster_label_info(label_file):
    if label_file is not None:
        cluster_label = dict()
        with open(label_file, "r") as l_f:
            for d_l in l_f:
                label_info = d_l.split("=")
                cluster_label[int(label_info[0].strip())] = label_info[1].strip()
        return cluster_label
    else:
        return None


def save_pickle(path, features, name=None):
    if not os.path.isdir(path):
        os.makedirs(path)

    if name is None:
        with open(path + "/features.p", "wb") as f:
            pickle.dump(features, f)
    else:
        with open(path + "/" + name + ".p", "wb") as f:
            pickle.dump(features, f)

def load_pickle(path):
    if path.endswith(".p"):
        with open(path, "rb") as f:
            features = pickle.load(f)
    else:
        with open(path + "/features.p", "rb") as f:
            features = pickle.load(f)

    return features

if __name__ == "__main__":
    debug = ARGS.debug
    log_dir = ARGS.log_dir
    tsne_comp = ARGS.tsne_comp
    output_dir = ARGS.output_dir
    label_info = ARGS.label_info
    cluster_data = ARGS.cluster_data
    deep_features = ARGS.deep_features
    principle_comp = ARGS.principle_comp

    log.info(f"Logging Level: {debug}")
    log.info(f"Directory to Store the Logging File: {log_dir}")
    log.info(f"Directory to Store the Final Output: {output_dir}")
    log.info(f"Deep Feature Input File (.p) Including Latent Embeddings and Other Information from previous clustering: {deep_features}")
    log.info(f"Clustering Input File (.p) Including Cluster Centers und Cluster Predictions with respect to the given deep features from the previous Clustering: {cluster_data}")
    log.info(f"Number of Principle Components: {principle_comp}")
    log.info(f"Number of t-SNE dimensions: {tsne_comp}")
    log.info(f"Label Input File: {label_info}")

    #Clustering Data
    clustering_data = load_pickle(cluster_data)
    cluster_predictions = clustering_data["cluster_pred"]
    cluster_centers = clustering_data["cluster_centers"]

    #Deep Feature Data
    deep_feature_data = load_pickle(deep_features)
    deep_feature_filenames = np.array(deep_feature_data["filenames"])
    deep_feature_vectors = np.array(deep_feature_data["features"])

    #Sorting
    sort_indices = np.argsort(list(cluster_predictions))
    cluster_predictions = cluster_predictions[sort_indices]
    deep_feature_filenames = deep_feature_filenames[sort_indices]
    deep_feature_vectors = deep_feature_vectors[sort_indices]

    #Labels
    labels = load_cluster_label_info(label_info)

    #Merge Centers and Feature Array
    merged_features = []
    for c_idx in range(cluster_centers.shape[0]):
        if labels is not None:
            lbl = labels.get(c_idx)
        else:
            lbl = c_idx
        merged_features.append(("Cluster-"+str(lbl)+"-Center", "NONE", cluster_centers[c_idx]))
        for p_idx in range(cluster_predictions.shape[0]):
            if c_idx == cluster_predictions[p_idx]:

                merged_features.append(("Cluster-"+str(lbl)+"-Element", deep_feature_filenames[p_idx], deep_feature_vectors[p_idx]))
            else:
                continue

    #Cluster and Deep Feature Information
    cluster_labels = [element[0] for element in merged_features]
    deep_feature_filenames = [element[1] for element in merged_features]
    deep_feature_vectors = np.array([element[2] for element in merged_features])

    #Dimensionality Reduction
    deep_features_dim_reduced = Dim_Reduction(pca_k=50, tsne_d=2, pca_first=True)(deep_feature_vectors)

    dim_reduced_result = {
        'cluster-labels': cluster_labels,
        'filenames': deep_feature_filenames,
        'deep_features': deep_feature_vectors,
        'deep_features_reduced': deep_features_dim_reduced
    }

    save_pickle(output_dir, features=dim_reduced_result, name="dim-reduction-data")

    #Code Example to load stored information from the dim-reduction-data pickle file ---> load_pickle(path=output_dir+"/dim-reduction-data.p")

    df = pd.DataFrame({
        "x": deep_features_dim_reduced[:, 0],
        "y": deep_features_dim_reduced[:, 1],
        "label": cluster_labels
    })

    Visualization(output_directory=output_dir).visualize2d(df, hue_key="label")
