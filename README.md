# **Speech-Based Dementia Detection Using Whisper and Log Neural Controlled Differential Equations**
This repository contains the implementation of a speech-based dementia detection model combining Whisper and LogNCDEs.
## **Dateset**
Our experiments were conducted using three publicly available datasets: ADReSS, ADReSSo, and DementiaBank, each containing speech samples from healthy controls (HC) and Alzheimer's dementia (AD) patients.In the DementiaBank dataset specifically, some participants exhibited changes in diagnostic categories during follow-up (mainly MCI converting to AD), so all MCI cases were excluded from this study to maintain a clean binary classification.For detailed participant selection refer to DB_list. The table below shows the speech sample counts from each dataset:
| Dataset       | HC Samples | AD Samples | Total |
|:--------------|-----------:|:----------:|------:|
| ADReSS        | 78        | 78          | 156   |
| ADReSSo       | 122       | 115         | 237   |
| DementiaBank  | 222        | 255        | 477   |


*Note: HC = Healthy Controls, AD = Alzheimer's Dementia*
