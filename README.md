# **Speech-Based Dementia Detection Using Whisper and Log Neural Controlled Differential Equations**
This repository contains the implementation of a speech-based dementia detection model combining Whisper and LogNCDEs.
## **Dateset**
Our experiments were conducted using three publicly available datasets: ADReSS, ADReSSo, and DementiaBank, each containing speech samples from healthy controls (HC) and Alzheimer's dementia (AD) patients.In the DementiaBank dataset specifically, some participants exhibited changes in diagnostic categories during follow-up (mainly MCI converting to AD), so all MCI cases were excluded from this study to maintain a clean binary classification.For detailed participant selection refer to DB_list. The table below shows the speech sample counts from each dataset:
| Dataset       | HC Samples | AD Samples | Total |
|:--------------|-----------:|:----------:|------:|
| ADReSS        | 78        | 78          | 156   |
| ADReSSo       | 122       | 115         | 237   |
| DementiaBank  | 222        | 255        | 477   |

Since Whisper requires 30-second audio inputs, we segmented all audio files into 30-second segments (see data/audio_cut for details). Additionally, as Whisper's output contains up to 768 channels, we employed an autoencoder to reduce the dimensionality to 32 for improved model efficiency ((see data/autoencoder for details).
## **Models**
The models folder includes implementations of both LogNCDEs and baseline models, with complete hyperparameter configurations documented in models/param.
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




