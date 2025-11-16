# **Speech-Based Dementia Detection Using Whisper and Log Neural Controlled Differential Equations**
This repository contains the implementation of a speech-based dementia detection model combining Whisper and LogNCDEs.
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


## **Environment Specifications** ##
### **Hardware Configuration** ###
**CPU**:Intel Xeon Gold 6326 (32-core @ 2.90GHz)

**GPU**:NVIDIA A100 80GB PCIe
### **Software Environment** ###
| Component       | Version       |
|-----------------|---------------
| Python          | 3.10.15       | 
| JAX             | 0.4.34        | 
| PyTorch         | 2.5.0         | 
| CUDA Toolkit    | 12.1          |
| cuDNN           | 9.1.0         |
| NVIDIA Driver   | 535.216.01    |
| diffrax         | 0.6.0         |
| equinox         | 0.11.8        |

## **Result** ##
The "result" folder contains the reported results from the paper as well as the detailed results for each seed used in the experiments.




