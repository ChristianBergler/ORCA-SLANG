#!/usr/bin/env python3
"""
Module: main.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import json
import math
import pathlib
import argparse

from collections import OrderedDict

import torch.onnx
import torch.nn as nn
import torch.optim as optim

from models.loss import DeepFeatureLoss

from models.unet_model import UNet

from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder

from models.residual_decoder import DefaultDecoderOpts
from models.residual_decoder import ResidualDecoder as Decoder

from models.latent import Latent as Latent
from models.latent import DefaultLatentOpts as DefaultLatentOpts

from trainer import Trainer

from utils.logging import Logger

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    CsvSplit,
    DeepFeatureDataset,
    DefaultSpecDatasetOps,
)

parser = argparse.ArgumentParser()

"""
Convert string to boolean.
"""
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

parser.add_argument(
    "--data_dir",
    type=str,
    default="/tmp/data",
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="The path to the cache directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    default="/tmp/model",
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="/tmp/checkpoints",
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="The directory to store the logs.",
)

parser.add_argument(
    "--summary_dir",
    type=str,
    default="/tmp/summaries",
    help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
    "--noise_dir",
    type=str,
    default=None,
    help="Path to a directory with noise files used for data augmentation.",
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--augmentation",
    type=str2bool,
    default=True,
    help="Whether to augment the input data. Validation and test data will not be augmented.",
)

parser.add_argument(
    "--start_from_scratch",
    dest="start_from_scratch",
    action="store_true",
    help="Start training from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--pooling",
    type=str,
    default="avg",
    help="Pooling used after last residual layer, avg, max",
)

parser.add_argument(
    "--max_train_epochs",
    type=int,
    default=500,
    help="The number of epochs to train for the autoencoder.",
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="The number of images per batch.",
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of workers used in data-loading",
)

parser.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1",
    type=float,
    default=0.5,
    help="beta1 for the adam optimizer.",
)

parser.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate every N epochs.",
)

parser.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
    "--early_stopping_patience_epochs",
    type=int,
    default=20,
    help="Decay the learning rate every N epochs.",
)

parser.add_argument(
    "--output_activation",
    type=str,
    default="sigmoid",
    help="Final output activation function (decoder path, last residual layer) - Options: `'sigmoid (default)'`, '`relu`', '`tanh`', '`none`'",
)

parser.add_argument(
    "--dropout_prob_encoder",
    type=float,
    default=0.2,
    help="Decay factor to apply to the learning rate. In case it is 0 dropout functionality is deactivated.",
)

parser.add_argument(
    "--dropout_prob_decoder",
    type=float,
    default=0.2,
    help="Decay factor to apply to the learning rate. In case it is 0 dropout functionality is deactivated.",
)

parser.add_argument(
    "--denoiser_pickle",
    type=str,
    default=None,
    help="Path to the trained denoiser pickle",
)

parser.add_argument(
    "--denoise",
    type=str2bool,
    default=False,
    help="Denoise the inputs for the training using the denoiser.",
)

parser.add_argument(
    "--orca_detection",
    type=str2bool,
    default=False,
    help="Use the Orca detection to cut out",
)

parser.add_argument(
    "--filter_broken_audio",
    action="store_true",
    help="Filter by a minimum loudness.",
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
    help="Frequency compression to reduce GPU memory usage. Options: `'linear'` (default), '`mel`', `'mfcc'`",
)

parser.add_argument(
    "--train_mode",
    type=str,
    default="autoencoder",
    help="Training Modes using different architectural approaches. Options: `'autoencoder'` (network input = network output), '`autoencoder_denoised`' (denoised network input = denoised network output), `'denoiser'` (network input = denoised network output)",
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
    help=""
         "Minimal frequency.",
)

parser.add_argument(
    "--fmax",
    type=int,
    default=10000,
    help="Maximal frequency.",
)

parser.add_argument(
    "--sr",
    type=int,
    default=44100,
    help="Sample rate",
)

parser.add_argument(
    "--conv_kernel_size",
    nargs="*",
    type=int,
    default=7,
    help="Initial convolution kernel size.",
)

parser.add_argument(
    "--max_pool",
    type=int,
    default=2,
    help="Max pooling after the initial convolution layer. 0: No max pooling, stride 2, 1: Max pooling, 2: No Max pooling, stride 1",
)

parser.add_argument(
    "--latent_kernel_size",
    type=int,
    default=5,
    help="convolution kernel size in Latent layer.",
)

parser.add_argument(
    "--latent_kernel_stride",
    type=int,
    default=1,
    help="convolution kernel stride in Latent layer.",
)

parser.add_argument(
    "--latent_size",
    type=int,
    default=512,
    help="size of the latent/bottleneck layer.",
)

parser.add_argument(
    "--latent_channels",
    type=int,
    default=512,
    help="How many channels in the latent layer",
)

parser.add_argument(
    "--latent_conv_only",
    dest="latent_conv_only",
    action="store_true",
    help="Compression and decompression of the feature maps from the last residual layer via 1x1 conv/transposed convolutions only",
)

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--perc_of_max_signal",
    type=float,
    default=1.0,
    help="The percentage multiplied with the maximum signal strength resulting in the target height for the maximum intensity peak picking orca detection algorithm.",
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

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")
if ARGS.conv_kernel_size is not None and len(ARGS.conv_kernel_size):
    ARGS.conv_kernel_size = ARGS.conv_kernel_size[0]

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
"""
def get_audio_files():
    audio_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))
        if ARGS.filter_broken_audio:
            data_dir_ = pathlib.Path(ARGS.data_dir)
            audio_files = get_audio_files_from_dir(ARGS.data_dir)
            log.debug("Moving possibly broken audio files to .bkp:")
            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
            for f in broken_files:
                log.debug(f)
                bkp_dir = data_dir_.joinpath(f).parent.joinpath(".bkp")
                bkp_dir.mkdir(exist_ok=True)
                f = pathlib.Path(f)
                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.close()
            exit(1)
    return audio_files

"""
Save the autoencoder in a pickle
"""
def save_autoencoder(encoder, encoderOpts, latent, latentOpts, decoder, decoderOpts, dataOpts, path, train_mode, name="ORCA-FEATURE.pk"):
    encoder = encoder.cpu()
    latent = latent.cpu()
    decoder = decoder.cpu()
    encoder_state_dict = encoder.state_dict()
    latent_state_dict = latent.state_dict()
    decoder_state_dict = decoder.state_dict()

    save_dict = {
        "encoderOpts": encoderOpts,
        "latentOpts": latentOpts,
        "decoderOpts": decoderOpts,
        "dataOpts": dataOpts,
        "encoderState": encoder_state_dict,
        "latentState": latent_state_dict,
        "decoderState": decoder_state_dict,
        "train_mode": train_mode
    }
    if not os.path.isdir(path):
        os.makedirs(path)
    path = os.path.join(path, name)
    torch.save(save_dict, path)


if __name__ == "__main__":

    debug = ARGS.debug
    data_dir = ARGS.data_dir
    cache_dir = ARGS.cache_dir
    model_dir = ARGS.model_dir
    checkpoint_dir = ARGS.checkpoint_dir
    log_dir = ARGS.log_dir
    summary_dir = ARGS.summary_dir
    noise_dir = ARGS.noise_dir
    cuda = ARGS.cuda
    augmentation = ARGS.augmentation
    start_from_scratch = ARGS.start_from_scratch
    pooling = ARGS.pooling
    max_train_epochs = ARGS.max_train_epochs
    epochs_per_eval = ARGS.epochs_per_eval
    batch_size = ARGS.batch_size
    num_workers = ARGS.num_workers
    lr = ARGS.lr
    beta1 = ARGS.beta1
    lr_patience_epochs = ARGS.lr_patience_epochs
    lr_decay_factor = ARGS.lr_decay_factor
    early_stopping_patience_epochs = ARGS.early_stopping_patience_epochs
    dropout_prob_encoder = ARGS.dropout_prob_encoder
    dropout_prob_decoder = ARGS.dropout_prob_decoder
    denoiser_pickle = ARGS.denoiser_pickle
    denoise = ARGS.denoise
    orca_detection = ARGS.orca_detection
    filter_broken_audio = ARGS.filter_broken_audio
    sequence_len = ARGS.sequence_len
    freq_compression = ARGS.freq_compression
    n_freq_bins = ARGS.n_freq_bins
    n_fft = ARGS.n_fft
    hop_length = ARGS.hop_length
    fmin = ARGS.fmin
    fmax = ARGS.fmax
    sr = ARGS.sr
    conv_kernel_size = ARGS.conv_kernel_size
    max_pool = ARGS.max_pool
    latent_kernel_size = ARGS.latent_kernel_size
    latent_kernel_stride = ARGS.latent_kernel_stride
    latent_size = ARGS.latent_size
    latent_channels = ARGS.latent_channels
    min_max_norm = ARGS.min_max_norm
    train_mode = ARGS.train_mode
    perc_of_max_signal = ARGS.perc_of_max_signal
    min_thres_detect = ARGS.min_thres_detect
    max_thres_detect = ARGS.max_thres_detect
    latent_conv_only = ARGS.latent_conv_only
    output_activation = ARGS.output_activation

    log.info(f"Logging Level: {debug}")
    log.info(f"Data Directory with the Input Audio: {data_dir}")
    log.info(f"Cache Directory to Store Cached Files: {cache_dir}")
    log.info(f"Directory to Store the Final Model: {model_dir}")
    log.info(f"Directory to Store the Checkpoints: {checkpoint_dir}")
    log.info(f"Directory to Store the Logging File: {log_dir}")
    log.info(f"Directory to Store the Network Training Summaries: {summary_dir}")
    log.info(f"Directory of all Noise Files Required for Noise Augmentation: {noise_dir}")
    log.info(f"GPU Support: {cuda}")
    log.info(f"Data Augmentation: {augmentation}")
    log.info(f"Start Training from Scratch while Ignoring Previous Checkpoints: {start_from_scratch}")
    log.info(f"Type of Pooling After the Last Residual Layer: {pooling}")
    log.info(f"Number of Maximum Training Epochs: {max_train_epochs}")
    log.info(f"Number of Training Epochs After Evaluation: {epochs_per_eval}")
    log.info(f"Number of Files per Batch: {batch_size}")
    log.info(f"Number of Workers for Data Loading: {num_workers}")
    log.info(f"Network Initial Learning Rate: {lr}")
    log.info(f"Adam Optimizer Beta Value: {beta1}")
    log.info(f"Number of Epochs After Learning Rate Decay: {lr_patience_epochs}")
    log.info(f"Factor of Learning Rate Decay: {lr_decay_factor}")
    log.info(f"Number of Epochs After Training is Stopped in Case of no Improvements on Validation: {early_stopping_patience_epochs}")
    log.info(f"Probability of Dropout within Encoder: {dropout_prob_encoder}")
    log.info(f"Probability of Dropout within Decoder: {dropout_prob_decoder}")
    log.info(f"Model to Trained Denoiser: {denoiser_pickle}")
    log.info(f"Activate Denoising: {denoise}")
    log.info(f"Activate Orca Detection: {orca_detection}")
    log.info(f"Activate Filtering of Broken Audios (maximum amplitude < 10e-3): {filter_broken_audio}")
    log.info(f"Temporal Audio Training Sequence Length: {sequence_len}")
    log.info(f"Type of Frequency Compression: {freq_compression}")
    log.info(f"Number of Frequency Bins After Compression: {n_freq_bins}")
    log.info(f"Size of FFT Window in Samples: {n_fft}")
    log.info(f"Train Mode: {train_mode}")
    log.info(f"Size of Hop in Samples: {hop_length}")
    log.info(f"Minimum Frequency Considered for Freuqency Compression: {fmin}")
    log.info(f"Maximum Frequency Considered for Freuqency Compression: {fmax}")
    log.info(f"Target Sampling Rate: {sr}")
    log.info(f"Size of Convolutional Kernel from the Initial Convolution: {conv_kernel_size}")
    log.info(f"Type of Initial Max Pooling: {max_pool}")
    log.info(f"Size of Convolutional Kernel within Latent Layer: {latent_kernel_size}")
    log.info(f"Stride of Convolutional Kernel within Latent Layer: {latent_kernel_stride}")
    log.info(f"Size of Latent Feature Vector: {latent_size}")
    log.info(f"Channels of within the Latent Layer: {latent_channels}")
    log.info(f"Min-Max Normalization: {min_max_norm}")
    log.info(f"Latent layer only 1x1 conv/transposed conv: {latent_conv_only}")
    log.info(f"The Percentage Multiplied with the Maximum Signal Strength Resulting in the Target Height for the Maximum Intensity Peak Picking Orca Detection Algorithm: {perc_of_max_signal}")
    log.info(f"Minimum/Lower value (percentage) for the embedded peak finding algorithm to calculate intensity between min_freq_bin and max_freq_bin (n_fft/2+1 * min_thres_detect = min_freq_bin). For the Orca Detection: {min_thres_detect}")
    log.info(f"Maximum/Upper value (percentage) for the embedded peak finding algorithm to calculate intensity between min_freq_bin and max_freq_bin (n_fft/2+1 * max_thres_detect = max_freq_bin). For the Orca Detection: {max_thres_detect}")
    log.info(f"Final Output Activation Function: {output_activation}")

    train_modes = ["autoencoder", "autoencoder_denoised", "denoiser"]

    if train_mode in train_modes:
        log.info("Valid network training mode: " + train_mode)
    else:
        raise Exception("Invalid network training mode: " + str(train_mode))

    """
    If a denoiser model is available: load it
    """
    if denoise and train_mode != train_modes[0]:
        log.info("Denoising activated, Model: " + str(denoiser_pickle))
        denoiser_model_dict = torch.load(denoiser_pickle)
        denoiser = UNet(1, 1, bilinear=False)
        denoiser.load_state_dict(denoiser_model_dict["unetState"])
        denoiser = nn.Sequential(
            OrderedDict([("denoiser", denoiser)])
        )
        denoiser_dataOpts = denoiser_model_dict["dataOpts"]

        if torch.cuda.is_available() and cuda:
            denoiser = denoiser.cuda()
        denoiser.eval()

        log.info("Using denoiser for given input: " + denoiser_pickle)

    else:
        log.info("Denoising deactivated and/or not required due to chosen training mode!")

        if train_mode == train_modes[1] or train_mode == train_modes[2]:
            log.error("Training network mode=" + str(train_mode) + " requires an enabled denoising option plus a trained denoising model!")
            raise Exception("Training network mode=" + str(train_mode) + " requires an enabled denoising option plus a trained denoising model!")

        denoiser = None

    if orca_detection:
        log.info("Using orca_detection")

    dataOpts = DefaultSpecDatasetOps

    for arg, value in vars(ARGS).items():
        if arg in dataOpts and value is not None:
            dataOpts[arg] = value

    log.debug("Data Options: " + json.dumps(dataOpts, indent=4))

    encoderOpts = DefaultEncoderOpts
    latentOpts = DefaultLatentOpts
    decoderOpts = DefaultDecoderOpts

    for arg, value in vars(ARGS).items():
        if arg in encoderOpts and value is not None:
            encoderOpts[arg] = value

        if arg in decoderOpts and value is not None:
            decoderOpts[arg] = value

        if arg in latentOpts and value is not None:
            latentOpts[arg] = value

    log.debug("Encoder Options: " + json.dumps(encoderOpts, indent=4))
    log.debug("Decoder Options: " + json.dumps(decoderOpts, indent=4))
    log.debug("Latent Options: " + json.dumps(latentOpts, indent=4))

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    lr *= batch_size
    patience_lr = math.ceil(lr_patience_epochs / epochs_per_eval)
    patience_lr = int(max(1, patience_lr))

    sequence_len = int(float(sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"])

    log.info("Training with sequence length: {}".format(sequence_len))

    input_shape = (batch_size, 1, dataOpts["n_freq_bins"], sequence_len)

    if dropout_prob_encoder > 0 and dropout_prob_decoder > 0:
        log.info("Using Dropout!")
    elif dropout_prob_encoder <= 0 and dropout_prob_decoder <= 0:
        log.info("No Dropout is used!")
    else:
        log.error("Dropout settings are not consistent - check your command line parameters!")
        log.close()
        raise Exception("Dropout settings are not consistent - check your command line parameters!")

    log.info("Setting up model")

    # Setup encoder
    encoder = Encoder(encoderOpts)
    log.debug("Encoder: " + str(encoder))
    encoder_out_ch = 512 * encoder.block_type.expansion

    # Different initial max-pool settings
    if max_pool == 2:
        encoder_out_size = sequence_len // 16, dataOpts["n_freq_bins"] // 16
        decoderOpts["output_stride"] = 1  # Decoder must not use stride either
    else:
        encoder_out_size = sequence_len // 32, dataOpts["n_freq_bins"] // 32

    # Setup latent layer
    latentOpts["in_channels"] = encoder_out_ch
    latent = Latent(latentOpts)
    log.debug("latent model: " + str(latent))

    # Setup decoder
    decoderOpts["input_channels"] = encoder_out_ch
    decoder = Decoder(decoderOpts)
    log.debug("Decoder: " + str(decoder))

    # Setup Model
    model = nn.Sequential(
        OrderedDict(
            [("encoder", encoder), ("latent", latent), ("decoder", decoder)]
        )
    )

    split_fracs = {"train": .7, "val": .15, "test": .15}

    input_data = CsvSplit(split_fracs, working_dir=data_dir, split_per_dir=True)

    audio_files = get_audio_files()

    if noise_dir:
        noise_files = [str(p) for p in pathlib.Path(noise_dir).glob("*.wav")]
    else:
        noise_files = []

    datasets = {
        split: DeepFeatureDataset(
            file_names=input_data.load(split, audio_files),
            cuda=cuda,
            sr=dataOpts["sr"],
            dataset_name=split,
            cache_dir=cache_dir,
            working_dir=data_dir,
            seq_len=sequence_len,
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            n_fft=dataOpts["n_fft"],
            noise_files=noise_files,
            denoiser_model=denoiser,
            orca_detection=orca_detection,
            min_max_normalize=min_max_norm,
            min_thres_detect=min_thres_detect,
            max_thres_detect=max_thres_detect,
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            perc_of_max_signal=perc_of_max_signal if split == "train" else 1.0,
            freq_compression=dataOpts["freq_compression"],
            pure_feature_extraction=False,
            augmentation=augmentation if split == "train" else False

        )
        for split in split_fracs.keys()
    }

    for d in datasets.keys():
        log.debug("Number of files {}: {}".format(datasets[d].dataset_name, len(datasets[d].file_names)))

    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for split in split_fracs.keys()
    }

    trainer = Trainer(
        logger=log,
        model=model,
        n_summaries=4,
        mode=train_mode,
        prefix=train_mode,
        summary_dir=summary_dir,
        checkpoint_dir=checkpoint_dir,
        start_scratch=start_from_scratch,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

    metric_mode = "min"

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=1e-3,
        mode=metric_mode,
        patience=patience_lr,
        threshold_mode="abs",
        factor=lr_decay_factor,
    )

    deep_feature_loss = DeepFeatureLoss(reduction="sum")

    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        device=device,
        val_metric="loss",
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=deep_feature_loss,
        n_epochs=max_train_epochs,
        val_interval=epochs_per_eval,
        patience_early_stopping=early_stopping_patience_epochs,
    )

    if isinstance(model, nn.DataParallel):
        encoder = getattr(model.module, "encoder")
    else:
        encoder = getattr(model, "encoder")

    latent = model.latent

    decoder = model.decoder

    save_autoencoder(encoder, encoderOpts, latent, latentOpts, decoder, decoderOpts, dataOpts, model_dir, train_mode)

    log.close()
