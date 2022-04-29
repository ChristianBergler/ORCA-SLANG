"""
Module: cluster_algorithm.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import numpy as np

from sklearn.cluster import KMeans, SpectralClustering

import matplotlib.pyplot as plt

"""
Clustering class for all clustering and visualizations needs
"""
class cluster_algo:
    def __init__(self, features=None):
        self.features = features
        self.cluster_features = features

    """
    Set the vector features that are clustered
    """
    def set_features(self, features):
        self.features = features
        self.cluster_features = features

    """ 
    Kmeans clustering 
    """
    def kmeans(self, n_clusters, new_inits=100):
        cluster_algorithm = KMeans(n_clusters=n_clusters, n_init=new_inits)

        cluster_pred = cluster_algorithm.fit_predict(self.cluster_features)
        cluster_centers = cluster_algorithm.cluster_centers_

        return {
            'cluster_pred': cluster_pred,
            'cluster_centers': cluster_centers,
        }

    """ 
    Spectral clustering 
    """
    def spectral(self, n_clusters, assign_labels="kmeans", eigen_solver="arpack", affinity="nearest_neighbors",
                 n_neighbors=11, new_inits=100):
        cluster_algorithm = SpectralClustering(n_clusters,
                                               assign_labels=assign_labels,
                                               eigen_solver=eigen_solver,
                                               affinity=affinity,
                                               n_neighbors=n_neighbors,
                                               n_init=new_inits,
                                               )

        cluster_pred = cluster_algorithm.fit_predict(self.cluster_features)
        cluster_centers = self.compute_cluster_centers_mean(cluster_pred, self.cluster_features)

        return {
            'cluster_pred': cluster_pred,
            'cluster_centers': cluster_centers,
        }

    """
    For cluster algorithms that don't give cluster centers, compute them from the features
    """
    def compute_cluster_centers_mean(self, cluster_pred, features):
        features = np.array(features)

        W = {c: np.mean(features[cluster_pred == c, :], axis=0) for c in np.unique(cluster_pred)}

        return W

    """
    Save the clusters
    """
    def save_clusters(self, path, features, filenames, cluster_pred, cluster_centers):
        if not os.path.isdir(path):
            os.makedirs(path)

        distance_to_c_centers = []

        for cl, feature in zip(cluster_pred, features):
            distance_to_c_centers.append(np.linalg.norm(cluster_centers[cl] - np.array(feature)))

        cluster_texts = [[] for i in range(0, max(cluster_pred) + 1)]

        listing = sorted(zip(distance_to_c_centers, cluster_pred, filenames), key=lambda pair: pair[0])
        for distance, cl, file_name in listing:
            cluster_texts[int(cl)].append((cl, distance, file_name))

        with open(path + "/clusters.txt", "a") as f:
            for cluster in cluster_texts:
                if len(cluster) != 0:
                    for cl, distance, file_name in cluster:
                        f.write('{},{},{}\n'.format(cl, distance, file_name))

    """
    Visualize the clusterings in order of nearnes to the cluster centers
    """
    def vis_clusters(self, path, features, filenames, spectras, cluster_pred, cluster_centers):
        plt.rcParams.update({'figure.max_open_warning': 0})
        if not os.path.isdir(path):
            os.makedirs(path)

        distance_to_c_centers = []

        for cl, feature in zip(cluster_pred, features):
            distance_to_c_centers.append(np.linalg.norm(cluster_centers[cl] - np.array(feature)))

        cluster_pics = [0 for i in range(0, max(cluster_pred) + 1)]

        listing = sorted(zip(distance_to_c_centers, cluster_pred, filenames, spectras), key=lambda pair: pair[0])

        counter = -1
        for distance, cl, file_name, spec in listing:
            counter += 1
            cl = int(cl)

            cluster_pics[cl] += 1

            if cluster_pics[cl] <= 12:
                plt.figure(cl, figsize=(8, 8))
                plt.rcParams.update({'font.size': 7})
                plt.subplot(3, 4, cluster_pics[cl])
                plt.title(
                    '[' + str(cl) + "-" + str(round(distance, 3)) + "] " + self.get_reasonable_filename(file_name),
                    fontsize=7)
                plt.imshow(spec.squeeze().transpose(1, 0), origin='lower')

                if cluster_pics[cl] == 12:
                    plt.rcParams["axes.grid"] = False
                    plt.savefig(path + "/spec-" + str(cl).zfill(4) + "-" + str(counter).zfill(5) + ".png", dpi=200,
                                orientation="portrait", pad_inches=0.0, bbox_inches=0.0)
                    plt.close()

                    cluster_pics[cl] = 0

        for i in range(0, max(cluster_pred) + 1):
            if cluster_pics[i] != 0:
                plt.figure(i)

                plt.rcParams["axes.grid"] = False
                plt.savefig(path + "/spec-" + str(i).zfill(4) + "-" + str(counter).zfill(5) + ".png", dpi=200,
                            orientation="portrait", pad_inches=0.0, bbox_inches=0.0)
                plt.close()

                counter += 1

    """ 
    Get the filename without path and dummy label 
    """
    def get_reasonable_filename(self, file_name):
        fnSplit = file_name.split("/")[-1]
        fnSplit2 = fnSplit.split(".")[0]
        fnSplit2 = fnSplit2[0:10]
        fnSplit2 = fnSplit2.replace("orca_1_", "")

        return fnSplit2
