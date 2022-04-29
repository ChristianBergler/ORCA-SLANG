#!/usr/bin/env python3
"""
Module: cluster.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import argparse

import torch

from utils.logging import Logger

from clustering.feature_extractor import Extractor
from clustering.cluster_algorithm import cluster_algo

parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",

)

parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="The directory to store the logs."
)

parser.add_argument(
    "--audio_dir",
    type=str,
    default="",
    help="The path to the audiofile directory that should be clustered - could also have sub-directories, but has to contain only audio files.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    default="",
    help="The directory where the autoencoder model is stored.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The directory where everything will be saved",
)

parser.add_argument(
    "--denoiser_dir",
    type=str,
    default="",
    help="Path to the trained denoiser pickle",
)

parser.add_argument(
    "--denoise",
    type=str2bool,
    default=False,
    help="Denoise the inputs before the clustering.",
)

parser.add_argument(
    "--orca_detection",
    type=str2bool,
    default=False,
    help="Use the Orca detection to cut out",
)

parser.add_argument(
    "--perc_of_max_signal",
    type=float,
    default=1.0,
    help="The percentage of the maximum signal strength a signal needs to have to be cut out. For the orca detection.",
)

parser.add_argument(
    "--sequence_len",
    type=int,
    default=1280,
    help="Sequence length in ms.",
)

parser.add_argument(
    "--freq_compression",
    type=str,
    default="linear",
    help="Frequency compression to reduce GPU memory usage. Options: `'linear'` (default)",
)

parser.add_argument(
    "--n_freq_bins",
    type=int,
    default=256,
    help="Number of frequency bins after compression.",
)

parser.add_argument(
    "--n_fft",
    type=int,
    default=4096,
    help="FFT size.",
)

parser.add_argument(
    "--hop_length",
    type=int,
    default=441,
    help="FFT hop length.",
)

parser.add_argument(
    "--fmin",
    type=int,
    default=500,
    help="Minimum frequency",
)

parser.add_argument(
    "--fmax",
    type=int,
    default=10000,
    help="Maximum frequency",
)

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--visualize_files",
    type=str2bool,
    default=False,
    help="Save the incoming image and reconstruction",
)

parser.add_argument(
    "--visualize_clusters",
    type=str2bool,
    default=False,
    help="Save the spectrograms of the clusters",
)

parser.add_argument(
    "--save_spectra",
    type=str2bool,
    default=False,
    help="Save the spectra of the input in the pickle",
)

parser.add_argument(
    "--save_spectra_recon",
    type=str2bool,
    default=False,
    help="Save the reconstructed spectra of the input in the pickle",
)

parser.add_argument(
    "--latent_size",
    type=int,
    default=512,
    help="size of the latent/bottleneck layer.",
)

parser.add_argument(
    "--clusters",
    type=int,
    default=0,
    help="The amount of clusters used in the clustering",
)

parser.add_argument(
    "--cluster_algorithm",
    type=str,
    default="kmeans",
    help="The Cluster algorithm to be used, 'kmeans' or 'spectralclustering'.",
)

parser.add_argument(
    "--min_thres_detect",
    type=float,
    default=0.05,
    help="Minimum/Lower value (percentage) for the embedded peak finding algorithm to calculate the minimum frequency bin (n_fft/2+1 * min_thres_detect = min_freq_bin) in order to robustly calculate signal strength within a min_freq_bin and max_freq_bin range to extract a fixed temporal context from longer vocal events. For the orca detection.",
)

parser.add_argument(
    "--max_thres_detect",
    type=float,
    default=0.40,
    help="Maximum/Upper value (percentage) for the embedded peak finding algorithm to calculate the maximum frequency bin (n_fft/2+1 * max_thres_detect = max_freq_bin) in order to robustly calculate signal strength within a min_freq_bin and max_freq_bin range to extract a fixed temporal context from longer vocal events. For the orca detection.",
)

parser.add_argument(
    "--only_feature_extraction",
    action="store_true",
    help="Just computes feature extraction via ORCA-FEATURE without performing clustering.",
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

log = Logger("CLUSTER", ARGS.debug, ARGS.log_dir)

if __name__ == "__main__":

    cuda = ARGS.cuda
    debug = ARGS.debug
    log_dir = ARGS.log_dir
    audio_dir = ARGS.audio_dir
    model_dir = ARGS.model_dir
    output_dir = ARGS.output_dir
    denoiser_dir = ARGS.denoiser_dir
    denoise = ARGS.denoise
    orca_detection = ARGS.orca_detection
    perc_of_max_signal = ARGS.perc_of_max_signal
    sequence_len = ARGS.sequence_len
    freq_compression = ARGS.freq_compression
    n_freq_bins = ARGS.n_freq_bins
    n_fft = ARGS.n_fft
    hop_length = ARGS.hop_length
    fmin = ARGS.fmin
    fmax = ARGS.fmax
    min_max_norm = ARGS.min_max_norm
    visualize_files = ARGS.visualize_files
    visualize_clusters = ARGS.visualize_clusters
    save_spectra = ARGS.save_spectra
    save_spectra_recon = ARGS.save_spectra_recon
    latent_size = ARGS.latent_size
    clusters = ARGS.clusters
    cluster_algorithm = ARGS.cluster_algorithm
    min_thres_detect = ARGS.min_thres_detect
    max_thres_detect = ARGS.max_thres_detect
    only_feature_extraction = ARGS.only_feature_extraction

    log.info(f"GPU Support: {cuda}")
    log.info(f"Logging Level: {debug}")
    log.info(f"Directory to Store the Logging File: {log_dir}")
    log.info(f"Data Directory with the Input Audio: {audio_dir}")
    log.info(f"Directory where the Trained Autoencoder Model is Stored: {model_dir}")
    log.info(f"Directory to Store all the Clustering Output: {output_dir}")
    log.info(f"Directory where the Trained Denoising Model ist Stored: {denoiser_dir}")
    log.info(f"Activate Denoising: {denoise}")
    log.info(f"Activate Orca Detection: {orca_detection}")
    log.info(
        f"The Percentage Multiplied with the Maximum Signal Strength Resulting in the Target Height for the Maximum Intensity Peak Picking Orca Detection Algorithm: {perc_of_max_signal}")
    log.info(f"Temporal Audio Training Sequence Length: {sequence_len}")
    log.info(f"Type of Frequency Compression: {freq_compression}")
    log.info(f"Number of Frequency Bins After Compression: {n_freq_bins}")
    log.info(f"Size of FFT Window in Samples: {n_fft}")
    log.info(f"Size of Hop in Samples: {hop_length}")
    log.info(f"Minimum Frequency Considered for Freuqency Compression: {fmin}")
    log.info(f"Maximum Frequency Considered for Freuqency Compression: {fmax}")
    log.info(f"Min-Max Normalization: {min_max_norm}")
    log.info(f"Visualization of Audio Input Files: {visualize_files}")
    log.info(f"Visualization of Generated Clusters: {visualize_clusters}")
    log.info(f"Saving of Created Spectrograms in the Pickle File: {save_spectra}")
    log.info(f"Saving of Generated Reconstruction Spectrograms in the Pickle File: {save_spectra_recon}")
    log.info(f"Size of the Latent Layer: {latent_size}")
    log.info(f"Number of Clusters Used During Clustering: {clusters}")
    log.info(f"The Cluster Algorithm to be Used - kmeans or spectralclustering: {cluster_algorithm}")
    log.info(
        f"Minimum/Lower value (percentage) for the embedded peak finding algorithm to calculate intensity between min_freq_bin and max_freq_bin (n_fft/2+1 * min_thres_detect = min_freq_bin). For the Orca Detection: {min_thres_detect}")
    log.info(
        f"Maximum/Upper value (percentage) for the embedded peak finding algorithm to calculate intensity between min_freq_bin and max_freq_bin (n_fft/2+1 * max_thres_detect = max_freq_bin). For the Orca Detection: {max_thres_detect}")
    log.info(f"Only Feature Extraction without subsequent Clustering: {only_feature_extraction}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    DeepFeatureExtractor = Extractor(cuda=cuda, log=log)
    DeepFeatureExtractor.set_spec_options(
        n_fft=n_fft,
        hop_length=hop_length,
        n_freq_bins=n_freq_bins,
        sequence_len=sequence_len,
        fmin=fmin,
        fmax=fmax,
        min_thres_detect=min_thres_detect,
        max_thres_detect=max_thres_detect
    )

    DeepFeatureExtractor.load_ae_model(model_dir)

    if denoise:
        DeepFeatureExtractor.load_denoiser(denoiser_dir)

    pickle_name = "deep_features"

    if not os.path.isfile(output_dir + "/" + pickle_name + ".p"):
        log.info("starting feature extraction")
        features = DeepFeatureExtractor.start_latent_extraction(
            audio_dir,
            save_dir=output_dir,
            visualize_files=visualize_files,
            save_spectra=save_spectra,
            save_spectra_recon=save_spectra_recon,
            orca_detection=orca_detection,
            perc_of_max_signal=perc_of_max_signal,
            denoise=denoise,
            min_max_normalize=min_max_norm
        )
        log.info("Saving features!")
        DeepFeatureExtractor.save_pickle(output_dir, features, pickle_name)
    else:
        log.info("Deep Feature File already exist - Loading features!")
        features = DeepFeatureExtractor.load_pickle(output_dir + "/" + pickle_name + ".p")

    log.info("Feature Length: " + str(len(features["features"])))

    if not only_feature_extraction:

        if clusters == 0:
            clusters = int(len(features["features"]) / 100)
            log.info("using " + str(clusters) + " clusters")

        log.info("Starting clustering")

        cl = cluster_algo(features=features["features"])

        if cluster_algorithm == "kmeans":
            clustering = cl.kmeans(clusters)

        elif cluster_algorithm == "spectralclustering":
            clustering = cl.spectral(clusters, n_neighbors=11)

        log.info("Clustering finished")

        cl.save_clusters(
            output_dir + "/clustering_" + cluster_algorithm + "_" + str(clusters) + "/",
            features["features"],
            features["filenames"],
            clustering["cluster_pred"],
            clustering["cluster_centers"]
        )

        if visualize_clusters:
            cl.vis_clusters(
                output_dir + "/clustering_" + cluster_algorithm + "_" + str(clusters) + "/",
                features["features"],
                features["filenames"],
                features["spectra_input"],
                clustering["cluster_pred"],
                clustering["cluster_centers"]
            )

        DeepFeatureExtractor.save_pickle(output_dir, clustering, "clustering")
