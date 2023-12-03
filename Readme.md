# FakeBuster: A DeepFakes Detection Tool for Video Conferencing Scenarios

This repository contains the source code for the paper- [FakeBuster: A DeepFakes Detection Tool for Video Conferencing Scenarios](https://dl.acm.org/doi/10.1145/3397482.3450726)

This paper proposes FakeBuster, a novel DeepFake detector for (a) detecting impostors during video conferencing, and (b) manipulated faces on social media. FakeBuster is a standalone deep learning- based solution, which enables a user to detect if another person’s video is manipulated or spoofed during a video conference-based meeting. This tool is independent of video conferencing solutions and has been tested with Zoom and Skype applications. It employs a 3D convolutional neural network for predicting video fakeness. The network is trained on a combination of datasets such as Deeperforensics, DFDC, VoxCeleb, and deepfake videos created using locally captured images (specific to video conferencing scenarios). Diversity in the training data makes FakeBuster robust to multiple environments and facial manipulations, thereby making it generalizable and ecologically valid.

# Installation
```
conda create -n fakeBuster python=3.7
conda activate fakeBuster
conda install -c anaconda pyqt joblib pyqtgraph matplotlib
conda install -c conda-forge opencv python-mss
pip install --no-cache-dir torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
# Run
This tool requires model weights for face detector and deepfake detector which can be downloaded from the [drive link](https://drive.google.com/drive/folders/1Vej_l-g6wvwaaAcigxuf4l8-IHxNWk75?usp=sharing)  \
Configure parameters in `config.py`
1. Update the path parameters `checkpointpath` and `weights_path`.
2. Configure model device placement- `fake_det_device` and `fake_det_device`

Run command:-
```
python main.py
```


# Additional notes

## Citation

```
@inproceedings{10.1145/3397482.3450726,
author = {Mehta, Vineet and Gupta, Parul and Subramanian, Ramanathan and Dhall, Abhinav},
title = {FakeBuster: A DeepFakes Detection Tool for Video Conferencing Scenarios},
year = {2021},
isbn = {9781450380188},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3397482.3450726},
doi = {10.1145/3397482.3450726},
abstract = {This paper proposes FakeBuster, a novel DeepFake detector for (a) detecting impostors during video conferencing, and (b) manipulated faces on social media. FakeBuster is a standalone deep learning- based solution, which enables a user to detect if another person’s video is manipulated or spoofed during a video conference-based meeting. This tool is independent of video conferencing solutions and has been tested with Zoom and Skype applications. It employs a 3D convolutional neural network for predicting video fakeness. The network is trained on a combination of datasets such as Deeperforensics, DFDC, VoxCeleb, and deepfake videos created using locally captured images (specific to video conferencing scenarios). Diversity in the training data makes FakeBuster robust to multiple environments and facial manipulations, thereby making it generalizable and ecologically valid.},
booktitle = {26th International Conference on Intelligent User Interfaces - Companion},
pages = {61–63},
numpages = {3},
keywords = {Deepfakes detection, spoofing, neural networks},
location = {College Station, TX, USA},
series = {IUI '21 Companion}
}
```

ACM Ref.
```
Vineet Mehta, Parul Gupta, Ramanathan Subramanian, and Abhinav Dhall. 2021. FakeBuster: A DeepFakes Detection Tool for Video Conferencing Scenarios. In 26th International Conference on Intelligent User Interfaces - Companion (IUI '21 Companion). Association for Computing Machinery, New York, NY, USA, 61–63. https://doi.org/10.1145/3397482.3450726
```