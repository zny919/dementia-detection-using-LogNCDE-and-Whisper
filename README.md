# **Speech-Based Dementia Detection Using Whisper and Log Neural Controlled Differential Equations**
This repository contains the implementation of a speech-based dementia detection model combining Whisper and LogNCDEs.

**Anonymous Repository Declaration:** > To strictly comply with double-blind review policies, this repository and its associated GitHub account have been completely anonymized. No author identities, affiliations, or any other identifying information will be disclosed until the manuscript is officially accepted.

## **Dateset**
Our experiments were conducted using four publicly available datasets: ADReSS, ADReSSo, DementiaBank, and NCMMSC, each containing speech samples from healthy controls (HC) and Alzheimer's dementia (AD) patients.In the DementiaBank dataset specifically, some participants exhibited changes in diagnostic categories during follow-up (mainly MCI converting to AD), so all MCI cases were excluded from this study to maintain a clean binary classification.For detailed participant selection refer to DB_list. The table below shows the speech sample counts from each dataset:
| Dataset       | HC Samples | AD Samples | Total |
|:--------------|-----------:|:----------:|------:|
| ADReSS        |   78       |    78      | 156   |
| ADReSSo       |   122      |    115     | 237   |
| DementiaBank  |   222      |    255     | 477   |
| NCMMSC        |   153      |    246     | 399   |

Since Whisper requires 30-second audio inputs, we segmented all audio files into 30-second segments (see data/audio_cut for details). Additionally, as Whisper's output contains up to 768 channels, we employed an autoencoder to reduce the dimensionality to 32(NCMMSC is 64) for improved model efficiency ((see data/autoencoder for details).
## **Models**
The models folder includes implementations of both LogNCDEs and baseline models, with complete hyperparameter configurations documented in models/hyperparameter. Since the audio is first split into short segments for prediction and then the segment-level results are aggregated back to the original recording, our model produces outputs under two aggregation strategies. The first takes the mean of the segment-level prediction scores and assigns label 1 if this mean exceeds 0.5. The second counts how many segments have prediction scores greater than 0.5 and assigns label 1 if this count exceeds half of the total number of segments for that recording. In our paper, we report results based on the latter strategy, which is recorded in the code using the fields “acc_vote” and “F1_vote”.

## Operations Guide

This project consists of three sequential scripts that must be executed in order. First, raw audio data is ingested and sliced to extract features using the Whisper model. Next, these extracted features are passed through an Autoencoder to reduce their dimensionality. Finally, the reduced features are fed into the Log-NCDEs model for training and evaluation.

While the core workflow remains identical across all datasets, please note that the **NCMMSC2021** dataset requires the `whisper-medium` model, whereas the others use `whisper-small`. When configuring the models, you can use the `single` parameter to extract features from one specific Whisper layer, or the `last` parameter to fuse the final few layers. For the standard feature extraction process in the first script, simply set `single = 12`.

During data processing, the training and testing sets are initially divided into healthy control (`cc`) and clinical dementia (`cd`) categories. After feature extraction, the data is stacked sequentially into unified `train` and `test` tensors: all `cc` samples are placed at the front, followed entirely by `cd` samples. Consequently, the label arrays are constructed as a continuous block of 0s followed by a block of 1s. The DataLoader handles automatic shuffling during training, ensuring this initial ordering does not negatively impact model performance. 

Audio slices follow this identical 0-then-1 stacking format. The model first generates predictions for individual slices, which are then mapped and aggregated back to their original parent audio file using an `indices` array. This mapping format is strictly `[original_audio_index, slice_index_within_audio]`.

To execute the model, input your specific random seeds at the bottom of the third script within the control interface. Upon completion, the results for each seed are automatically exported to the designated save directory. Each run generates two files: a metrics file containing aggregate statistics (Accuracy, Precision, Recall, F1-score) and a detailed predictions log for each individual audio segment.

## Environment Specifications

### Hardware Configuration
* **CPU:** Intel Xeon Gold 6326 (32-core @ 2.90GHz)
* **GPU:** NVIDIA A100 80GB PCIe

### Software Environment

| Component | Version |
| :--- | :--- |
| Python | 3.9.25 |
| JAX | 0.4.30 |
| jaxlib | 0.4.30 |
| PyTorch (torch) | 2.4.1+cu121 |
| torchaudio | 2.4.1+cu121 |
| CUDA Toolkit | 12.1 |
| cuDNN | 9.1.0 |
| NVIDIA Driver | 535.216.01 |
| diffrax | 0.5.1 |
| equinox | 0.11.4 |
| optax | 0.2.2 |
| signax | 0.1.1 |
| librosa | 0.10.1 |
| transformers | 4.44.2 |
| soundfile | 0.12.1 |
| numpy | 1.26.3 |
| scikit-learn | 1.5.2 |
| matplotlib | 3.8.4 |

## **Result** ##
The "result" folder contains the reported results from the paper as well as the detailed results for each seed used in the experiments.




