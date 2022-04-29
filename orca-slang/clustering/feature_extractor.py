"""
Module: feature_extractor.py
Authors: Christian Bergler, Manuel Schmitt, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import pickle

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from collections import OrderedDict

from models.latent import Latent

from models.unet_model import UNet

from visualization.cm import viridis_cm
from visualization.utils import spec2img, flip

from data.audiodataset import DeepFeatureDataset

from models.residual_encoder import ResidualEncoder as Encoder
from models.residual_decoder import ResidualDecoder as Decoder

"""
Autoencoder class used for latent feature extraction which needs a trained autoencoder pickle
"""
class Extractor:
    def __init__(self, cuda=False, log=None):
        self.log = log
        self.cuda = cuda

        self.model = None
        self.denoiser = None
        self.train_mode = None

        self.train_modes = ["autoencoder", "autoencoder_denoised", "denoiser"]

    """
    Set the options for the spectrogram
    """

    def set_spec_options(
            self,
            sr=44100,
            preemphases=0.98,
            n_fft=4096,
            hop_length=441,  # 10 ms
            n_freq_bins=256,
            fmin=500,
            fmax=10000,
            freq_compression="linear",
            sequence_len=1280,
            min_thres_detect=0.05,
            max_thres_detect=0.40
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_freq_bins
        self.freq_compression = freq_compression
        self.fmin = fmin
        self.fmax = fmax
        self.sequence_len = sequence_len
        self.preemphases = preemphases
        self.min_thres_detect = min_thres_detect
        self.max_thres_detect = max_thres_detect

    """
    Go through a directory and get all the latent layer vectors, do visualizations of the incoming spectrograms if parameter is set
    """
    def start_latent_extraction(
            self,
            audio_dir,
            save_dir=None,
            visualize_files=False,
            save_spectra=False,
            save_spectra_recon=False,
            orca_detection=False,
            perc_of_max_signal=1.0,
            denoise=False,
            min_max_normalize=False
    ):
        audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir)])

        sequence_len = int(
            float(self.sequence_len) / 1000 * self.sr / self.hop_length
        )

        dataset = DeepFeatureDataset(
            file_names=audio_files,
            sr=self.sr,
            cuda=self.cuda,
            f_min=self.fmin,
            f_max=self.fmax,
            n_fft=self.n_fft,
            seq_len=sequence_len,
            hop_length=self.hop_length,
            n_freq_bins=self.n_freq_bins,
            denoiser_model=self.denoiser,
            orca_detection=orca_detection,
            min_thres_detect=self.min_thres_detect,
            max_thres_detect=self.max_thres_detect,
            min_max_normalize=min_max_normalize,
            perc_of_max_signal=perc_of_max_signal,
            freq_compression=self.freq_compression,
            pure_feature_extraction=True
        )

        if torch.cuda.is_available() and self.cuda:
            self.model = self.model.cuda()

        self.model.eval()

        file_names = []
        feature_array = []
        spectra_input = []
        spectra_output = []

        vis_dir = save_dir + "/visualizations"
        if not os.path.isdir(vis_dir):
            os.makedirs(vis_dir)

        count_seg_overall = 0
        file_count_overall = 0
        for features, features_denoised, label, times in dataset:

            file_count_overall += 1

            count_ex_seg = 0

            for i, _ in enumerate(features):
                count_seg_overall += 1

                count_ex_seg += 1

                self.log.info("Number of Files=" + str(file_count_overall) + ", Extracted Segment per File=" + str(count_ex_seg) + ", Total Extracted Segments=" + str(count_seg_overall) + ", Label=" + str(label))

                filename = label["file_name"].split("/")[-1]

                if self.train_mode == self.train_modes[0]:
                    network_input = features[i]
                elif self.train_mode == self.train_modes[1]:
                    network_input = features_denoised[i]
                elif self.train_mode == self.train_modes[2]:
                    network_input = features[i]

                network_input = network_input.unsqueeze(0)

                if torch.cuda.is_available() and self.cuda:
                    network_input = network_input.to("cuda")
                    self.model.to("cuda")
                    out = self.model(network_input)
                else:
                    out = self.model(network_input)

                feature_array.append(self.get_layer_output("latent", "code").cpu().detach().squeeze().numpy())

                from_t = 0
                to_t = 0

                if times is not None:
                    from_t, to_t = times[i]

                file_names.append(label["file_name"] + "_" + str(from_t) + "_" + str(to_t))

                if save_spectra:
                    spectra_input.append(self.get_layer_output("encoder", "input_layer").cpu().detach())

                if save_spectra_recon:
                    spectra_output.append(self.get_layer_output("decoder", "output_layer").cpu().detach())

                if visualize_files:
                    if denoise:
                        self.visualize_undenoised_enc_dec_calls(
                            vis_dir + "/" + filename + "_" + str(from_t) + "_" + str(to_t) + ".png",
                            features[i].unsqueeze(0))
                    else:
                        self.visualize_enc_dec_calls(
                            vis_dir + "/" + filename + "_" + str(from_t) + "_" + str(to_t) + ".png")

        return {
            'filenames': file_names,
            'features': feature_array,
            'spectra_input': spectra_input,
            'spectra_output': spectra_output,
        }

    """
    Get the output of one layer of the autoencoder
    """
    def get_layer_output(self, model_part, layer_name):
        if model_part == "encoder":
            out = self.model.encoder.get_layer_output().get(layer_name)
        elif model_part == "decoder":
            out = self.model.decoder.get_layer_output().get(layer_name)
        elif model_part == "latent":
            out = self.model.latent.get_layer_output().get(layer_name)

        return out

    """
    Helpers to do Visualizations
    """
    def spec_to_img(self, spec, channel_index=0, flip_image=True):
        img = spec2img(spec, cm=viridis_cm)
        if flip_image:
            img = flip(img, dim=-1)
        img = torch.transpose(img, 1, 3)
        img = img[channel_index]

        return img

    def visualize_enc_calls(self, filepath, channel_index=0):
        spec = self.get_layer_output("encoder", "input_layer")
        img_input = self.spec_to_img(spec.transpose(0, 1), channel_index)

        self.save_1_imgs(filepath, img_input)

    def visualize_enc_dec_calls(self, filepath, channel_index=0):
        spec = self.get_layer_output("encoder", "input_layer")
        img_input = self.spec_to_img(spec.transpose(0, 1), channel_index)
        spec = self.get_layer_output("decoder", "output_layer")
        img_output = self.spec_to_img(spec.transpose(0, 1), channel_index)

        self.save_2_imgs(filepath, img_input, img_output)

    def visualize_undenoised_enc_dec_calls(self, filepath, spec_undenoised, channel_index=0):
        spec = self.get_layer_output("encoder", "input_layer")
        img_input = self.spec_to_img(spec.transpose(0, 1), channel_index)

        spec = self.get_layer_output("decoder", "output_layer")
        img_output = self.spec_to_img(spec.transpose(0, 1), channel_index)

        self.save_3_imgs(filepath, self.spec_to_img(spec_undenoised), img_input, img_output)

    def save_1_imgs(self, filepath, img, cl=None):
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = 4, 7
        plt.grid(b=None)
        if cl is not None:
            plt.title('[' + str(cl) + "]", fontsize=10)
        plt.tight_layout()
        plt.imshow(img, origin='lower')
        plt.savefig(filepath, dpi=100, pad_inches=0.0, bbox_inches=0.0)
        plt.close()

    def save_2_imgs(self, filepath, img_1, img_2):
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = 8, 7
        plt.grid(b=None)
        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.imshow(img_1, origin='lower')
        plt.subplot(1, 2, 2)
        plt.imshow(img_2, origin='lower')
        plt.savefig(filepath, dpi=100, pad_inches=0.0, bbox_inches=0.0)
        plt.close()

    def save_3_imgs(self, filepath, img_1, img_2, img_3):
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = 12, 7
        plt.grid(b=None)
        plt.tight_layout()
        plt.subplot(1, 3, 1)
        plt.imshow(img_1, origin='lower')
        plt.subplot(1, 3, 2)
        plt.imshow(img_2, origin='lower')
        plt.subplot(1, 3, 3)
        plt.imshow(img_3, origin='lower')
        plt.savefig(filepath, dpi=100, pad_inches=0.0, bbox_inches=0.0)
        plt.close()

    def save_images(self, path, images):
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = 16, 14
        plt.grid(b=None)
        plt.tight_layout()

        for i, image in enumerate(images):
            plt.subplot(3, 4, (i % 12) + 1)
            plt.imshow(image, origin='lower')

            if i != 0 and i % 12 == 0:
                plt.savefig(path + "/kernel-" + str(i) + ".png", dpi=100, pad_inches=0.0, bbox_inches=0.0)
                plt.close()

                plt.rcParams['figure.figsize'] = 16, 14
                plt.grid(b=None)
                plt.tight_layout()

        plt.savefig(path + "/kernel-" + str(i) + ".png", dpi=100, pad_inches=0.0, bbox_inches=0.0)
        plt.close()

    def save_kernel_images(self, path, model_part):
        if not os.path.isdir(path):
            os.makedirs(path)

        weights = model_part.weight
        images = []
        for i in range(weights.size(0)):
            for j in range(weights.size(1)):
                w = weights[i, j].squeeze()
                img = spec2img(w, cm=viridis_cm)
                img = torch.transpose(img, 0, 2)
                images.append(img)

        self.save_images(path, images)

    """
    Load the Resnet Autoencoder
    """
    def load_ae_model(self, model_dir):
        model_dict_ae = torch.load(model_dir)

        encoder = Encoder(model_dict_ae["encoderOpts"])
        encoder.load_state_dict(model_dict_ae["encoderState"])

        decoder = Decoder(model_dict_ae["decoderOpts"])
        decoder.load_state_dict(model_dict_ae["decoderState"])

        latent = Latent(model_dict_ae["latentOpts"])
        latent.load_state_dict(model_dict_ae["latentState"])

        self.train_mode = model_dict_ae["train_mode"]

        self.model = nn.Sequential(
            OrderedDict(
                [("encoder", encoder), ("latent", latent), ("decoder", decoder)]
            )
        )

    """
    Load the denoiser
    """
    def load_denoiser(self, denoiser_dir):
        denoiser_model_dict = torch.load(denoiser_dir)

        denoiser_model = UNet(1, 1, bilinear=False)
        denoiser_model.load_state_dict(denoiser_model_dict["unet"])
        denoiser_model = nn.Sequential(
            OrderedDict([("unet", denoiser_model)])
        )

        if torch.cuda.is_available() and self.cuda:
            denoiser_model = denoiser_model.cuda()
        denoiser_model.eval()

        self.denoiser = denoiser_model

    """
    Save the extracted features 
    """
    def save_pickle(self, path, features, name=None):
        if not os.path.isdir(path):
            os.makedirs(path)

        if name is None:
            with open(path + "/features.p", "wb") as f:
                pickle.dump(features, f)
        else:
            with open(path + "/" + name + ".p", "wb") as f:
                pickle.dump(features, f)

    """
    Load the extracted features 
    """
    def load_pickle(self, path):
        if path.endswith(".p"):
            with open(path, "rb") as f:
                features = pickle.load(f)
        else:
            with open(path + "/features.p", "rb") as f:
                features = pickle.load(f)

        return features
