"""
Module: knn_classifier.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import argparse
import numpy as np

from clustering.feature_extractor import Extractor
from sklearn.neighbors import KNeighborsClassifier

from utils.logging import Logger

parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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


parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

parser.add_argument(
    "--classified_deep_features",
    type=str,
    default=None,
    help="Pickle of the deep feature latent vectors which have been classified (any possible classification system/model) based on the previous clustering.",
)

parser.add_argument(
    "--cluster_data_classified_deep_features",
    type=str,
    default=None,
    help="Pickle of the clustering data (centers, predictions) from the clustering of the deep features which have already been classified.",
)

parser.add_argument(
    "--deep_features_to_classify",
    type=str,
    default=None,
    help="Pickle of the deep feature latent vectors which needed to be classified/assigned (uncategorized) based on the previous clustering.",
)

parser.add_argument(
    "--label_info",
    type=str,
    default=None,
    help="Mapping cluster numbers to real label names according to any previous applied (multi-class) classification system/model"
)

parser.add_argument(
    "--n_neighbors",
    type=int,
    default=5,
    help="The amount of neighbours used in KNN"
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
    help="The directory to store the logs."
)

ARGS = parser.parse_args()

log = Logger("KNN", ARGS.debug, ARGS.log_dir)

if __name__ == "__main__":
    debug = ARGS.debug
    log_dir = ARGS.log_dir
    output_dir = ARGS.output_dir
    classified_deep_features = ARGS.classified_deep_features
    n_neighbors = ARGS.n_neighbors
    label_info = ARGS.label_info
    deep_features_to_classify = ARGS.deep_features_to_classify
    cluster_data_classified_deep_features = ARGS.cluster_data_classified_deep_features

    log.info(f"Logging Level: {debug}")
    log.info(f"Directory to Store the Logging File: {log_dir}")
    log.info(f"Directory to Store the Final Output: {output_dir}")
    log.info(f"Deep Feature Input File (.p) Including Latent Embeddings which are already Classified: {classified_deep_features}")
    log.info(f"Clustering Input File (.p) Including Cluster Centers und Cluster Predictions from the already Classified Features: {cluster_data_classified_deep_features}")
    log.info(f"Deep Feature Input File (.p) Including Latent Embeddings which need to be Classified: {deep_features_to_classify}")
    log.info(f"The amount of neighbours used in KNN: {n_neighbors}")
    log.info(f"Label Input File: {label_info}")

    DeepFeatureExtractor = Extractor(cuda=False)

    # load required input data
    deep_features_classified = DeepFeatureExtractor.load_pickle(classified_deep_features)

    clustering_data_classified_deep_features = DeepFeatureExtractor.load_pickle(cluster_data_classified_deep_features)

    deep_features_to_be_classified = DeepFeatureExtractor.load_pickle(deep_features_to_classify)

    cluster_label_info = load_cluster_label_info(label_info)

    # extract required input data
    filenames = deep_features_classified["filenames"]
    features = deep_features_classified["features"]
    spectras = deep_features_classified["spectra_input"]
    clusters = clustering_data_classified_deep_features["cluster_pred"].tolist()

    # generate filename versus label structure
    zipper = zip(filenames, clusters)
    filename_cluster = dict(zipper)

    filename_label = dict()
    for fname in filename_cluster:
        cluster = filename_cluster.get(fname)
        label = cluster_label_info.get(cluster)
        filename_label[fname] = label

    # init KNN
    knn = KNeighborsClassifier(n_neighbors=ARGS.n_neighbors)

    labels = list(filename_label.values())

    knn.fit(np.array(features), labels)

    # using KNN to assign new unseen features (classification)
    knn_features_to_be_classified = []
    for filename, feature in zip(deep_features_to_be_classified["filenames"],
                                 deep_features_to_be_classified["features"]):
        feature = np.array(feature).reshape(1, -1).squeeze()
        knn_features_to_be_classified.append(np.array(feature))

    knn_prediction = knn.predict(np.array(knn_features_to_be_classified))

    # store finale KNN ouput within plain text file
    with open(output_dir + "/KNN.output", "w") as f:
        for cll, fname, feature in zip(knn_prediction, deep_features_to_be_classified["filenames"],
                                       deep_features_to_be_classified["features"]):
            log.info(fname + "=" + cll)
            f.write('{}={}\n'.format(fname, cll))

    log.close()
