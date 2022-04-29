# ORCA-SLANG
ORCA-SLANG: An Automatic Multi-Stage Semi-Supervised Deep Learning Framework for Large-Scale Killer Whale Call Type Identification

## General Description
ORCA-SLANG, is a machine-driven, multi-stage, semi-supervised, deep learning framework for killer whale (<em>Orcinus Orca</em>) call type identification,
designed for large-scale recognition of already known 
call types besides the detection of possible sub-call patterns
and/or unlabeled vocalization categories. ORCA-SLANG combines the following sequentially ordered components of deep learning-based algorithms:

1) Orca Sound Type VS. Noise Segmentation (see ORCA-SPOT - https://github.com/ChristianBergler/ORCA-SPOT)

2) Orca Signal Denoising/Enhancement (see ORCA-CLEAN - https://github.com/ChristianBergler/ORCA-CLEAN)

3) Deep Orca Sound Type Feature Learning (see ORCA-FEATURE - https://github.com/ChristianBergler/ORCA-SLANG/ORCA-FEATURE)

4) Hybrid Semi-Supervised Call Type Identification

    4.1) Unsupervised Call Type Clustering (k-means, spectral clustering) (see https://github.com/ChristianBergler/ORCA-SLANG/CLUSTERING)
    
    4.2) Supervised Call Type Classification (ORCA-TYPE - integrated within the ANIMAL-SPOT deep learning framework - https://github.com/ChristianBergler/ANIMAL-SPOT, or any other preferred classification algorithm/system) of all identified clusters (cluster-purity) to build a machine-driven annotated data repository for subsequent large-scall k-Nearest Neighbor (k-NN) classification (see https://github.com/ChristianBergler/ORCA-SLANG/k-NN)
     

## Reference
If ORCA-SLANG is used for your own research please cite the following publication: <em>ORCA-SLANG: An Automatic Multi-Stage Semi-Supervised Deep Learning
Framework for Large-Scale Killer Whale Call Type Identification</em>. In case just single components of ORCA-SLANG are used, please cite the publication listed within the corresponding GitHub repositories:

ORCA-SPOT - https://github.com/ChristianBergler/ORCA-SPOT

ORCA-CLEAN - https://github.com/ChristianBergler/ORCA-CLEAN

ORCA-TYPE/ANIMAL-SPOT - https://github.com/ChristianBergler/ANIMAL-SPOT

ORCA-FEATURE, CLUSTERING, k-NN - https://github.com/ChristianBergler/ORCA-SLANG/

```
@inproceedings{Bergler-OSL-2021,
author={Christian Bergler and Manuel Schmitt and Andreas Maier and Helena Symonds and Paul Spong and Steven R. Ness and George Tzanetakis and Elmar Nöth},
title={{ORCA-SLANG: An Automatic Multi-Stage Semi-Supervised Deep Learning Framework for Large-Scale Killer Whale Call Type Identification}},
year=2021,
booktitle={Proc. Interspeech 2021},
pages={2396--2400},
doi={10.21437/Interspeech.2021-616}
}
```
## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

## General Information
Manuscript Title: <em>ORCA-SLANG: An Automatic Multi-Stage Semi-Supervised Deep Learning Framework for Large-Scale Killer Whale Call Type Identification</em>. Within the <em>orca-slang</em> folder all deep learning sub-components are listed (either linked to other exsiting GitHub repositories, and/or stored within this repository) in order to set up the entire machine learning pipeline.

## Python, Python Libraries, and Version
ORCA-SLANG is a deep learning pipeline which was implemented in Python (Version=3.8) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.8.1, TorchVision=0.9.1, TorchAudio=0.8.1). Moreover it requires the following Python libraries: Pillow, MatplotLib, Librosa, TensorboardX, Soundfile, Scikit-image, Six, Resampy, Opencv-python, Pandas, Seaborn (recent versions). ORCA-SLANG is currently compatible with Python 3.8 and PyTorch (Version=1.11.0+cu113/cpu, TorchVision=0.12.0+cu113/cpu, TorchAudio=0.11.0+cu113/cpu).

## Required Filename Structure for Training
In order to properly load and preprocess your data to train the network you need to prepare the filenames of your audio data clips to fit the following template/format:

Filename Template: LABEL-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav

1st-Element: LABEL = a placeholder for any kind of string which describes the label of the respective sample, e.g. call-N9, orca, echolocation, etc.

2nd-Element: ID = unique ID (natural number) to identify the audio clip

3rd-Element: YEAR = year of the tape when it has been recorded

4th-Element: TAPENAME = name of the recorded tape (has to be unique in order to do a proper data split into train, devel, test set by putting one tape only in only one of the three sets

5th-Element: STARTTIME = start time of the audio clip in milliseconds with respect to the original recording (natural number)

6th-Element: ENDTIME = end time of the audio clip in milliseconds with respect to the original recording(natural number)

Due to the fact that the underscore (_) symbol was chosen as a delimiter between the single filename elements please do not use this symbol within your filename except for separation.

Examples of valid filenames:

call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919

Label Name=call-Orca-A12, ID=929, Year=2019, Tapename=Rec-031-2018-10-19-06-59-59-ASWMUX231648, Starttime in ms=2949326, Starttime in ms=2949919

orca-vocalization_2381_2010_101BC_149817_150055.wav

Label Name=orca-vocalization, ID=2381, Year=2010, Tapename=101BC, Starttime in ms=149817, Starttime in ms=150055

In case the original annotation time stamps, tape information, year information, or any filename-specific info is not available, also artificial/fake names/timestamps can be chosen. It only needs to be ensured that the filenames follow the given structure.

## ORCA-SLANG Framework Setup
A step-by-step introduction and description of the sequentially ordered sub-components from the entire ORCA-SLANG framework is documented within this section.

### ORCA-SPOT (Segmentation)
ORCA-SPOT is a deep learning based alogrithm which was initially designed for killer whale sound detection in noise heavy underwater recordings. ORCA-SPOT distinguishes between two types of sounds: killer whales and noise (binary classification problem). It is based on a convolutional neural network architecture which is capable to segment large bioacoustic archives. ORCA-SPOT includes a data preprocessing pipeline plus the network architecture itself for training the model. For a detailed description about the core concepts, network architecture, preprocessing and evaluation pipeline please see our corresponding publication https://www.nature.com/articles/s41598-019-47335-w.

Data preprocessing, network training, and evaluation concerning ORCA-SPOT, are illustrated in detail within the corresponding GitHub repository: https://github.com/ChristianBergler/ORCA-SPOT

###ORCA-CLEAN (Denoising)
ORCA-CLEAN, is a deep denoising network designed for denoising of killer whale (Orcinus Orca) underwater recordings, not requiring any clean ground-truth samples, in order to improve the interpretation and analysis of bioacoustic signals by biologists and various machine learning algorithms.
ORCA-CLEAN was trained exclusively on killer whale signals resulting in a significant signal enhancement. To show and prove the transferability, robustness and generalization of ORCA-CLEAN even more, a deep denoising was also conducted for bird sounds (Myiopsitta monachus) and human speech. For a detailed description about the core concepts, network architecture, preprocessing and evaluation pipeline please see our corresponding publication https://www.isca-speech.org/archive/interspeech_2020/bergler20_interspeech.html.

Data preprocessing, network training, and evaluation concerning ORCA-CLEAN, are illustrated in detail within the corresponding GitHub repository: https://github.com/ChristianBergler/ORCA-CLEAN

###ORCA-FEATURE (Deep Feature Learning & Semi-Supervised Call Type Identification)
ORCA-FEATURE is a deep feature learning network, based on a ResNet18-
based convolutional undercomplete autoencoder, initially introduced in https://www.isca-speech.org/archive/interspeech_2019/bergler19_interspeech.html, and trained on pre-segmented (ORCA-SPOT) noisy or denoised (ORCA-CLEAN) killer whale sound type spectral representations. ORCA-FEATURE enables the opportunity, to learn a compact spectral representation of a given bioacoustic spectral input sample (latent features/embeddings) in a fully unsupervised manner. Thus, a downstream categorization of known, but also unseen vocalization patterns can be accomplished, either via pure unsupervised clustering algorithms, or a hybrid combination between clustering and supervised trained classification systems.

Data preprocessing, network training, and evaluation concerning ORCA-FEATURE, next to deep latent spectral feature clustering and k-NN-based orca call type assignment (semi-supervised call type identification), is documented within this GitHub repository (see below): https://github.com/ChristianBergler/ORCA-SLANG

###ORCA-TYPE/ANIMAL-SPOT (Supervised Orca Call Type Classification)
ANIMAL-SPOT is an animal-independent deep learning software framework that addresses various bioacoustic signal identifcation scenarios, such as: (1) binary target/noise detection, (2) multi-class species identification, and (3) multi-class call type recognition. ORCA-TYPE is a ResNet18-based Convolutional Neural Network (CNN), integrated and embedded as part within the ANIMAL-SPOT framework, in order to perform multi-class classification (species and/or animal-specific call types). A more detailed and deeper instruction about ORCA-TYPE with corresponding references can be found here https://www.isca-speech.org/archive/interspeech_2021/bergler21_interspeech.html. 

Data preprocessing, network training, and evaluation concerning ORCA-TYPE, are illustrated in detail within the corresponding GitHub repository: https://github.com/ChristianBergler/ANIMAL-SPOT (publicly available soon)

## ORCA-FEATURE - Network Training and Evaluation
For a detailed description about each possible training option we refer to the usage/code in main.py (usage: <em>main.py -h</em>). This is just an example command in order to start network training:

```main.py --debug --latent_kernel_size 5 --latent_channels 512 --learning_rate 10e-4 --max_pool 2 --batch_size 8 --num_workers 4 --data_dir path_to_input_data_dir --conv_kernel_size 5 --model_dir path_to_model_dir --log_dir path_to_log_dir --checkpoint_dir path_to_checkpoint_dir --summary_dir path_to_summary_dir --augmentation 0 --orca_detection 1 --freq_compression linear --n_fft 4096 --hop_length 441 --cache_dir path_to_cache_dir --min_max_norm --denoise 1 --denoiser_pickle path_to_ORCA_CLEAN_pickle --latent_size 512 --sequence_len 1280 --early_stopping_patience_epochs 20 --perc_of_max_signal 0.7 --sr 44100 --train_mode autoencoder_denoised --max_train_epochs 200```

During training ORCA-FEATURE will be verified on an independent validation set. In addition ORCA-FEATURE will be automatically evaluated on the test set. As evaluation criteria the validation/reconstruction loss is utilized. All documented results/images and the entire training process could be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir /directory_to_model/summaries/```

## Deep Latent Feature Extraction and Clustering 
For a detailed description about each possible option we refer to the usage/code in cluster.py (usage: <em>cluster.py -h</em>). This is just an example command in order to start the clustering procedure:

```cluster.py --debug  --audio_dir path_to_audio_dir --model_dir path_to_trained_ORCA_FEATURE_pickle --output_dir path_to_output_folder --orca_detection 1 --sequence_len 1280 --perc_of_max_signal 0.7 --save_spectra 1  --save_spectra_recon 1 --min_max_norm --denoise 1 --visualize_files 1 --denoiser_dir path_to_ORCA_CLEAN_pickle --clusters 10 --visualize_clusters 1 --log_dir path_to_log_dir --cluster_algorithm kmeans```

The <em>cluster.py</em> algorithm will generate two pickle (.p) output files within the chosen output directory, named <em>deep_feature.p</em> and <em>clustering.p</em>. According to the chosen settings regarding visualization <em>cluster.py</em> will also create a folder named <em>visualizations</em> to store the corresponding spectral outputs.

The file <em>deep_feature.p</em> stores a dictionary containing the following format:

```python
{
   'filenames': [], 
   'features': [], 
   'spectra_input': [], 
   'spectra_output': [] 
}
```

The file <em>clustering.p</em> stores a dictionary including the following format:

```python
{
   'cluster_pred': [], 
   'cluster_centers': []
}
```

In addition <em>cluster.py</em> also generates a file named <em>clusters.txt</em> which stores the following content:

```
Cluster Indice, Distance to the Cluster Center, Filename 
```


All those information can be used together with any kind of data post-processing actions, e.g. visualization of the embeddings, classification, etc.!

## Deep Latent Feature Dimensionality Redudction and Visualization
Usually the embedded feature vectors (latent features) are still of higher dimensions and therefor hard to interpret. In order to provide a mechanism to visualize and analyze/interpret those latent feature embeddings a algorithm for dimensionality reduction and visualization is helpful. 

```dim_reduct.py --debug --cluster_data path_to_clustering_pickle_file --deep_features path_to_deep_features_pickle_file --principle_comp 50 --tsne_comp 2 --log_dir path_to_log_dir --output_dir path_to_output_dir --label_info path_to_label_info_file```

For a detailed description about each possible training option we refer to the usage/code in <em>dim_reduct.py</em> (usage: <em>dim_reduct.py -h</em>). This is just an example command. The above mentioned option <em>--label_info</em> provides the human-/machine-annotated labels of the identified clusters. This file is not mandatory. In case it is not provided the cluster indices will be used as labels. Otherwise each identified cluster will be renamed according the provided class names. These labels can be either derived by manual auditive and/or visual inspection of the cluster-specific content, or via any machine-learning classification system. Each cluster has to be assigned to a respective category, e.g. when the cluster purity is > 70% in favor of a specific class, all elements will be assigned to the corresponding category. A cluster which can not be assigned to any class due to impurity, etc. can be mapped to a <em>GARBAGE/UNKNOWN</em> category. An example of a valid <em>label_info</em> file using for example 10 clusters is given here (<em>CLUSTER_INDICE=LABEL</em>):

```
0=Orca
1=Bird
2=Garbage
3=Dolphin
4=Seal
5=Chimpanzee
6=Orca
7=Garbage
8=Garbage
9=Seal
```

The algorithm <em>dim_reduct.py</em> computes a Principle Component Analysis (PCA), in combination with a t-distributed stochastic neighbor embedding (t-SNE), to reduce feature dimensionality to any preferred size. The entire output will be saved to a pickle (.p) file, named <em>dim-reduction-data.p</em>, including the following information:

```python
{
   'cluster-labels': [],
   'filenames': [],
   'deep_features': [],
   'deep_features_reduced': []
}
```

In case of a final 2D-feature size (see <em>reduced feature vector</em>) an additional 2D-visualization, named <em>dim_reduction.png</em>, is generated for additonal visual inspection!

## Semi-Supervised Classification

The last step, which involves semi-supervised k-NN-based classification, is only possible if there exist a portion of deep features which have been clustered and afterwards assigned to real-world categories, either via human annotation and/or machine-driven classification (any classification system possible). In case there exist a set of already classified deep features (each cluster and all corresponding elements are assigned to a single category - explanation see above), the corresponding deep features (<em>deep_feature.p</em>), clustering information file (<em>clustering.p</em>), as well as the associated label info file (<em>--label_info</em>, file structure - see above) have to be provided (for the classified samples only), together with any arbitrary number of additional non-classified deep features, generated via ORCA-FEATURE and the <em>cluster.py</em> algorithm (using the option <em>--only_feature_extraction</em>). The already classified features will be used as baseline data for the k-NN assignments. All non-classified features will be assigned to the existing class repertoire, while considering the chosen number of nearest neighbors.
 
 
For a detailed description about each possible option we refer to the usage/code in <em>knn_classifier.py</em> (usage: <em>knn_classifier.py -h</em>). This is just an example command:


```knn_classifier.py --debug --classified_deep_features path_to_deep_feature_pickle_file_of_classified_features --cluster_data_classified_deep_features path_to_clustering_pickle_file_of_classified_deep_features --label_info path_to_label_info_file  --deep_features_to_classify path_to_deep_feature_pickle_file_needs_to_be_classified --log_dir path_to_log_dir --output_dir path_to_output_dir --n_neighbors 5```

As a final result the <em>knn_classifier.py</em> returns a <em>KNN.output</em> file, which stores all the information about each unseen filename and the corresponding class assignment (based on k-NN):

```
FILENAME-A=Orca
FILENAME-B=Orca
FILENAME-C=Garbage
FILENAME-D=Garbage
FILENAME-E=Chimpanzee
FILENAME-F=Garbage
FILENAME-G=Garbage
FILENAME-H=Seal
FILENAME-I=Dolphin
FILENAME-J=Bird
```