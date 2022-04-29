"""
Module: audiodataset.py
Authors: Christian Bergler, Manuel Schmitt
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import sys
import csv
import glob
import random
import pathlib
import numpy as np
import soundfile as sf
import data.transforms as T

import torch
import torch.utils.data
import torch.multiprocessing as mp

import data.signal as signal

from math import ceil
from types import GeneratorType
from utils.logging import Logger
from collections import defaultdict
from utils.FileIO import AsyncFileReader
from typing import Any, Dict, Iterable, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Data preprocessing default options
"""
DefaultSpecDatasetOps = {
    "sr": 44100,
    "preemphases": 0.98,
    "n_fft": 4096,
    "hop_length": 441,
    "n_freq_bins": 256,
    "fmin": 500,
    "fmax": 10000,
    "freq_compression": "linear",
    "min_level_db": -100,
    "ref_level_db": 20,
}

"""
Get audio files from directory
"""
def get_audio_files_from_dir(path: str):
    audio_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    audio_files = map(lambda p: pathlib.Path(p), audio_files)
    audio_files = filter(lambda p: not p.match("*.bkp/*"), audio_files)
    base = pathlib.Path(path)
    return map(lambda p: str(p.relative_to(base)), audio_files)

"""
Helper class in order to speed up filtering potential broken files
"""
class _FilterPickleHelper(object):
    def __init__(self, predicate, *pred_args):
        self.predicate = predicate
        self.args = pred_args

    def __call__(self, item):
        return self.predicate(item, *self.args)

"""
Parallel Filtering to analyze incoming data files
"""
class _ParallelFilter(object):
    def __init__(self, iteratable, n_threads=None, chunk_size=1):
        self.data = iteratable
        self.n_threads = n_threads
        self.chunk_size = chunk_size

    def __call__(self, func, *func_args):
        with mp.Pool(self.n_threads) as pool:
            func_pickle = _FilterPickleHelper(func, *func_args)
            for keep, c in pool.imap_unordered(func_pickle, self.data, self.chunk_size):
                if keep:
                    yield c

"""
Analyzing loudness criteria of each audio file by checking maximum amplitude (default: 1e-3)
"""
def _loudness_criteria(file_name: str, working_dir: str = None):
    if working_dir is not None:
        file_path = os.path.join(working_dir, file_name)
    else:
        file_path = file_name
    y, __ = sf.read(file_path, always_2d=True, dtype="float32")
    max_ampl = y.max()
    if max_ampl < 1e-3:
        return True, file_name
    else:
        return False, None

"""
Filtering all audio files in previous which do not fulfill the loudness criteria
"""
def get_broken_audio_files(files: Iterable[str], working_dir: str = None):
    f = _ParallelFilter(files, chunk_size=100)
    return f(_loudness_criteria, working_dir)


"""
Computes the CSV Split in order to prepare for randomly partitioning all data files into a training, validation, and test corpus
by dividing the data in such a way that audio files of a given tape are stored only in one of the three partitions.
The filenames per dataset will be stored in CSV files (train.csv, val.csv, test.csv). Each CSV File will be merged into
a train, val, and test file holding the information how a single partition is made up from single CSV files. These three
files reflects the training, validation, and test set.
"""
class CsvSplit(object):

    def __init__(
        self,
        split_fracs: Dict[str, float],
        working_dir: (str) = None,
        seed: (int) = None,
        split_per_dir=False,
    ):
        if not np.isclose(np.sum([p for _, p in split_fracs.items()]), 1.):
            raise ValueError("Split probabilities have to sum up to 1.")
        self.split_fracs = split_fracs
        self.working_dir = working_dir
        self.seed = seed
        self.split_per_dir = split_per_dir
        self.splits = defaultdict(list)
        self._logger = Logger("CSVSPLIT")

    """
    Return split for given partition. If there is already an existing CSV split return this split if it is valid or
    in case there exist not a split yet generate a new CSV split
    """
    def load(self, split: str, files: List[Any] = None):

        if split not in self.split_fracs:
            raise ValueError(
                "Provided split '{}' is not in `self.split_fracs`.".format(split)
            )

        if self.splits[split]:
            return self.splits[split]
        if self.working_dir is None:
            self.splits = self._split_with_seed(files)
            return self.splits[split]
        if self.can_load_from_csv():
            if not self.split_per_dir:
                csv_split_files = {
                    split_: (os.path.join(self.working_dir, split_ + ".csv"),)
                    for split_ in self.split_fracs.keys()
                }
            else:
                csv_split_files = {}
                for split_ in self.split_fracs.keys():
                    split_file = os.path.join(self.working_dir, split_)
                    csv_split_files[split_] = []
                    with open(split_file, "r") as f:
                        for line in f.readlines():
                            csv_split_files[split_].append(line.strip())

            for split_ in self.split_fracs.keys():
                for csv_file in csv_split_files[split_]:
                    if not csv_file or csv_file.startswith(r"#"):
                        continue
                    csv_file_path = os.path.join(self.working_dir, csv_file)
                    with open(csv_file_path, "r") as f:
                        reader = csv.reader(f)
                        for item in reader:
                            file_ = os.path.basename(item[0])
                            file_ = os.path.join(os.path.dirname(csv_file), file_)
                            self.splits[split_].append(file_)
            return self.splits[split]

        if not self.split_per_dir:
            working_dirs = (self.working_dir,)
        else:
            f_d_map = self._get_f_d_map(files)
            working_dirs = [os.path.join(self.working_dir, p) for p in f_d_map.keys()]
        for working_dir in working_dirs:
            splits = self._split_with_seed(
                files if not self.split_per_dir else f_d_map[working_dir]
            )
            for split_ in splits.keys():
                csv_file = os.path.join(working_dir, split_ + ".csv")
                self._logger.debug("Generating {}".format(csv_file))
                if self.split_per_dir:
                    with open(os.path.join(self.working_dir, split_), "a") as f:
                        p = pathlib.Path(csv_file).relative_to(self.working_dir)
                        f.write(str(p) + "\n")
                if len(splits[split_]) == 0:
                    raise ValueError(
                        "Error splitting dataset. Split '{}' has 0 entries".format(
                            split_
                        )
                    )
                with open(csv_file, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    for item in splits[split_]:
                        writer.writerow([item])
                self.splits[split_].extend(splits[split_])
        return self.splits[split]

    """
    Check whether it is possible to correctly load information from existing csv files
    """
    def can_load_from_csv(self):
        if not self.working_dir:
            return False
        if self.split_per_dir:
            for split in self.split_fracs.keys():
                split_file = os.path.join(self.working_dir, split)
                if not os.path.isfile(split_file):
                    return False
                self._logger.debug("Found dataset split file {}".format(split_file))
                with open(split_file, "r") as f:
                    for line in f.readlines():
                        csv_file = line.strip()
                        if not csv_file or csv_file.startswith(r"#"):
                            continue
                        if not os.path.isfile(os.path.join(self.working_dir, csv_file)):
                            self._logger.error("File not found: {}".format(csv_file))
                            raise ValueError(
                                "Split file found, but csv files are missing. "
                                "Aborting..."
                            )
        else:
            for split in self.split_fracs.keys():
                csv_file = os.path.join(self.working_dir, split + ".csv")
                if not os.path.isfile(csv_file):
                    return False
                self._logger.debug("Found csv file {}".format(csv_file))
        return True

    """
    Create a mapping from directory to containing files.
    """
    def _get_f_d_map(self, files: List[Any]):

        f_d_map = defaultdict(list)
        if self.working_dir is not None:
            for f in files:
                f_d_map[str(pathlib.Path(self.working_dir).joinpath(f).parent)].append(
                    f
                )
        else:
            for f in files:
                f_d_map[str(pathlib.Path(".").resolve().joinpath(f).parent)].append(f)
        return f_d_map

    """
    Randomly splits the dataset using given seed
    """
    def _split_with_seed(self, files: List[Any]):
        if not files:
            raise ValueError("Provided list `files` is `None`.")
        if self.seed:
            random.seed(self.seed)
        return self.split_fn(files)

    """
    A generator function that returns all values for the given `split`.
    """
    def split_fn(self, files: List[Any]):
        _splits = np.split(
            ary=random.sample(files, len(files)),
            indices_or_sections=[
                int(p * len(files)) for _, p in self.split_fracs.items()
            ],
        )
        splits = dict()
        for i, key in enumerate(self.splits.keys()):
            splits[key] = _splits[i]
        return splits


"""
Dataset for that returns just the provided file names.
"""
class FileNameDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        transform=None,
        logger_name="TRAIN",
        dataset_name=None,
    ):
        if isinstance(file_names, GeneratorType):
            self.file_names = list(file_names)
        else:
            self.file_names = file_names
        self.working_dir = working_dir
        self.transform = transform
        self._logger = Logger(logger_name)
        self.dataset_name = dataset_name

    def __len__(self):
        if not isinstance(self.file_names, list):
            self.file_names = list(self.file_names)
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.working_dir:
            return os.path.join(self.working_dir, self.file_names[idx])
        sample = self.file_names[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

"""
Dataset for loading audio data.
"""
class AudioDataset(FileNameDataset):

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        sr=44100,
        mono=True,
        *args,
        **kwargs
    ):
        super().__init__(file_names, working_dir, *args, **kwargs)

        self.sr = sr
        self.mono = mono

    def __getitem__(self, idx):
        file = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file)
        sample = T.load_audio_file(file, self.sr, self.mono)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

"""
Dataset for processing an audio tape via a sliding window approach using a given
sequence length and hop size.
"""
class StridedAudioDataset(FileNameDataset):
    def __init__(
        self,
        file_name,
        log,
        sequence_len: int,
        hop: int,
        sr: int = 44100,
        fft_size: int = 4096,
        fft_hop: int = 441,
        n_freq_bins: int = 256,
        freq_compression: str = "linear",
        f_min: int = 200,
        f_max: int = 18000,
        center=True,
        min_max_normalize=False
    ):

        self.sp = signal.signal_proc()

        self.sr = sr
        self.hop = hop
        self.center = center
        self.filename = file_name
        self.sequence_len = sequence_len
        self._logger = log
        self.audio = T.load_audio_file(file_name, sr=sr, mono=True)
        self.n_frames = self.audio.shape[1]

        spec_t = [
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(fft_size, fft_hop, center=self.center),
        ]

        self.spec_transforms = T.Compose(spec_t)

        if freq_compression == "linear":
            self.t_compr_f = (T.Interpolate(n_freq_bins, sr, f_min, f_max))
        elif freq_compression == "mel":
            self.t_compr_f = (T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max))
        else:
            raise Exception("Undefined frequency compression")

        self.t_compr_a = T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"])

        if min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
            self._logger.debug("Init min-max-normalization activated")
        else:
            self.t_norm = T.Normalize(
                min_level_db=DefaultSpecDatasetOps["min_level_db"],
                ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
            )
            self._logger.debug("Init 0/1-dB-normalization activated")

    def __len__(self):
        full_frames = max(int(ceil((self.n_frames + 1 - self.sequence_len) / self.hop)), 1)
        if (full_frames * self.sequence_len) < self.n_frames:
            full_frames += 1
        return full_frames

    """
    Extracts signal part according to the current and respective position of the given audio file.
    """
    def __getitem__(self, idx):
        start = idx * self.hop

        end = min(start + self.sequence_len, self.n_frames)

        y = self.audio[:, start:end]

        sample_spec, sample_spec_cmplx = self.spec_transforms(y)

        sample_spec_orig = self.t_compr_a(sample_spec)

        sample_spec = self.t_compr_f(sample_spec_orig)

        sample_spec = self.t_norm(sample_spec)

        return sample_spec_orig, sample_spec, sample_spec_cmplx, self.filename

    def __delete__(self):
        self.loader.join()

    def __exit__(self, *args):
        self.loader.join()

"""
Dataset for processing a folder of various audio files
"""
class SingleAudioFolder(AudioDataset):

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        cache_dir=None,
        sr=44100,
        n_fft=1024,
        hop_length=512,
        freq_compression="linear",
        n_freq_bins=256,
        f_min=None,
        f_max=18000,
        center=True,
        min_max_normalize=False,
        *args,
        **kwargs
    ):
        super().__init__(file_names, working_dir, sr, *args, **kwargs)
        if self.dataset_name is not None:
            self._logger.info("Init dataset {}...".format(self.dataset_name))

        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.center = center
        self.hop_length = hop_length
        self.freq_compression = freq_compression

        self.sp = signal.signal_proc()

        valid_freq_compressions = ["linear", "mel", "mfcc"]

        if self.freq_compression not in valid_freq_compressions:
            raise ValueError(
                "{} is not a valid freq_compression. Must be one of {}",
               format(self.freq_compression, valid_freq_compressions),
            )

        self._logger.debug(
            "Number of test files: {}".format(len(self.file_names))
        )

        spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=sr),
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(n_fft, hop_length, center=self.center)
        ]

        self.file_reader = AsyncFileReader()

        if cache_dir is None:
            self.t_spectrogram = T.Compose(spec_transforms)
        else:
            self.t_spectrogram = T.CachedSpectrogram(
                cache_dir=cache_dir,
                spec_transform=T.Compose(spec_transforms),
                n_fft=n_fft,
                hop_length=hop_length,
                file_reader=AsyncFileReader(),
            )

        if self.freq_compression == "linear":
            self.t_compr_f = T.Interpolate(n_freq_bins, sr, f_min, f_max)
        elif self.freq_compression == "mel":
            self.t_compr_f = T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
        else:
            raise Exception("Undefined frequency compression")

        self.t_compr_a = T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"])

        if min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
            self._logger.debug("Init min-max-normalization activated")
        else:
            self.t_norm = T.Normalize(
                min_level_db=DefaultSpecDatasetOps["min_level_db"],
                ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
            )
            self._logger.debug("Init 0/1-dB-normalization activated")

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file_name)
        else:
            file = file_name

        sample_spec, sample_spec_cmplx = self.t_spectrogram(file)

        sample_spec_orig = self.t_compr_a(sample_spec)

        sample_spec = self.t_compr_f(sample_spec_orig)

        sample_spec = self.t_norm(sample_spec)

        return sample_spec_orig, sample_spec, sample_spec_cmplx, file_name


"""
Dataset for processing and providing a given set of input data and returning a final spectrogram together with additional label information
"""
class DeepFeatureDataset(AudioDataset):

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        cuda=True,
        sr=44100,
        cache_dir=None,
        n_fft=4096,
        f_min=500,
        f_max=10000,
        seq_len=128,
        hop_length=441,
        noise_files=[],
        n_freq_bins=256,
        augmentation=False,
        denoiser_model=None,
        orca_detection=False,
        perc_of_max_signal=1.0,
        min_max_normalize=False,
        min_thres_detect=0.05,
        max_thres_detect=0.40,
        freq_compression="linear",
        pure_feature_extraction=False,
        min_level_db=DefaultSpecDatasetOps["min_level_db"],
        ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
        *args,
        **kwargs
    ):
        file_names = [x for x in file_names if x.endswith(".wav")]
        super().__init__(file_names, working_dir, sr, *args, **kwargs)

        self.cuda = cuda
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.seq_len = seq_len
        self.hop_length = hop_length
        self.augmentation = augmentation
        self.denoiser_model = denoiser_model
        self.orca_detection = orca_detection
        self.min_thres_detect = min_thres_detect
        self.max_thres_detect = max_thres_detect
        self.freq_compression = freq_compression
        self.min_max_normalize = min_max_normalize
        self.perc_of_max_signal = perc_of_max_signal
        self.pure_feature_extraction = pure_feature_extraction

        self.sp = signal.signal_proc()

        if self.dataset_name is not None:
            self._logger.info("Init dataset {} with {} files...".format(self.dataset_name, len(file_names)))

        self.filter_dataset()

        self.cache_dir = cache_dir
        self.file_cached = [False for _ in range(len(file_names))]

        spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=sr),
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(n_fft, hop_length, center=False),
        ]

        self.t_spectrogram = T.Compose(spec_transforms)

        if self.augmentation:
            self._logger.debug("Init augmentation transforms for intensity, time, and pitch shift")
            self.t_amplitude = T.RandomAmplitude(3, -6)
            self.t_timestretch = T.RandomTimeStretch()
            self.t_pitchshift = T.RandomPitchSift()
        else:
            self._logger.debug("Running without augmentation")

        if self.freq_compression == "linear":
            self.t_compr_f = T.Interpolate(n_freq_bins, sr, f_min, f_max)
        elif self.freq_compression == "mel":
            self.t_compr_f = T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
        else:
            raise Exception("Undefined frequency compression")

        if augmentation:
            if noise_files:
                self._logger.debug("Init augmentation transform for random noise addition")
                self.t_addnoise = T.RandomAddNoise(
                    noise_files,
                    self.t_spectrogram,
                    T.Compose(self.t_timestretch, self.t_pitchshift, self.t_compr_f),
                    min_length=self.seq_len,
                    return_original=True
                )
            else:
                self.t_addnoise = None
                self._logger.debug("Running without random noise augmentation")

        self.t_compr_a = T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"])

        if self.min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
            self._logger.debug("Init min-max-normalization activated")
        else:
            self.t_norm = T.Normalize(
                min_level_db=min_level_db,
                ref_level_db=ref_level_db,
            )
            self._logger.debug("Init 0/1-dB-normalization activated")

        if self.orca_detection:
            self.t_subseq = T.PaddedSubsequenceSampler(seq_len, dim=1, random=True, pad_only=True)
        else:
            self.t_subseq = T.PaddedSubsequenceSampler(seq_len, dim=1, random=True, pad_only=False)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file_name)
        else:
            file = file_name

        if self.file_cached[idx]:
            samples, samples_denoised, label, times = torch.load(os.path.join(self.cache_dir, file_name) + ".pt")
        else:
            sample, _ = self.t_spectrogram(file)

            # Data augmentation
            if self.augmentation:
                sample = self.t_amplitude(sample)
                sample = self.t_pitchshift(sample)
                sample = self.t_timestretch(sample)

            # Fixed temporal context
            times = []
            if self.orca_detection:
                # Only padding
                sample = self.t_subseq(sample)
                sample_orca_detect = sample.clone()
                sample_orca_detect = self.t_compr_a(sample_orca_detect)
                sample_orca_detect = self.t_norm(sample_orca_detect)
                # Subsampling
                samples, times = self.sp.detect_strong_spectral_region(spectrogram=sample_orca_detect, spectrogram_to_extract=sample, n_fft=self.n_fft, target_len=self.seq_len, perc_of_max_signal=self.perc_of_max_signal, min_bin_of_interest=int(self.min_thres_detect * sample_orca_detect.shape[-1]), max_bin_of_inerest=int(self.max_thres_detect * sample_orca_detect.shape[-1]))
            else:
                # Pad or subsample the input if necessary
                samples = self.t_subseq(sample)
                times.append((0, 128))

            samples_denoised = []

            for sample_idx in range(len(samples)):

                sample = samples[sample_idx]

                # Frequency compression
                sample = self.t_compr_f(sample.unsqueeze(dim=0))

                # Noise Augmentation
                if self.augmentation and self.t_addnoise is not None:
                    sample, _ = self.t_addnoise(sample)

                # Decibel spectrogram
                sample = self.t_compr_a(sample)

                # Normalization
                sample = self.t_norm(sample)

                samples[sample_idx] = sample

                #Denoising
                sample = sample.unsqueeze(dim=0)

                with torch.no_grad():
                    if self.denoiser_model is not None:
                        if torch.cuda.is_available() and self.cuda:
                            input = sample.cuda()
                            sample = self.denoiser_model(input).cpu()
                        else:
                            sample = self.denoiser_model(sample).cpu()

                sample = sample.squeeze(0)

                samples_denoised.append(sample)

            #Load Label
            label = self.load_label(file)

            #Cache Files
            if self.cache_dir is not None:
                torch.save((samples, samples_denoised, label, times), os.path.join(self.cache_dir, file_name) + ".pt")
                self.file_cached[idx] = True

        #Network Training (requires only a single spectral excerpt) VS. Feature Extraction (stores all spectral excerpts)
        if not self.pure_feature_extraction:
            random_idx = random.randint(0, len(samples)-1)
            return samples[random_idx], samples_denoised[random_idx], label, times[random_idx]
        else:
            return samples, samples_denoised, label, times

    def filter_dataset(self):
        if self.__len__() == 0:
            raise ValueError("No data files found")

    def load_label(self, file_name: str):
        label = dict()
        label["file_name"] = file_name

        return label
