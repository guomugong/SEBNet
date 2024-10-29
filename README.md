# SEBNet
Efficient Multi-scale Learning via Scale Embedding for Diabetic Retinopathy Multi-lesion Segmentation
Please read our [paper](https://doi.org/10.1016/j.bspc.2024.107078) for more details!

## Introduction:
Automated segmentation of diabetic retinopathy (DR) lesions in fundus images is significant in computer aided
diagnosis. While numerous studies have tackled lesion segmentation, most have relied on traditional backbone networks for direct multi-scale learning, primarily focusing on the effective integration of multi-scale features. However, few studies have delved into efficient multi-scale learning specifically tailored for DR lesion segmentation. In previous research, parallel convolutional operations with varying kernel sizes or residual connections have been commonly employed for multi-scale learning. Nevertheless, the spatial resolution of features remains unchanged throughout the whole process. This limitation hinders multi-resolution interaction within these blocks, preventing them from fully embedding multi-resolution information. To this end, we present a Multi-step Scale Embedding Block (MSEB), comprising two branches that operate on different spatial resolutions to enhance multi-resolution interaction. Furthermore, to facilitate improved multi-scale learning, we have constructed a VGG-like encoder network based on MSEB. This encoder is complemented by a lightweight decoder designed to seamlessly integrate multi-scale information and restore lesion details. By combining the strengths of both the encoder and decoder networks, we have developed SEBNet for DR multi-lesion segmentation. Extensive experiments are conducted on the IDRiD, DDR, and FGADR datasets, and experimental results demonstrate that SEBNet achieves competitive performance compared to state-of-the-art methods.

## Usage
1) Training SEBNet on the IDRiD dataset
```
Preparing the dataset
Modify train.py
Running script train.py
```
2) Testing
```
python3 predict.py
```

3) Evaluation
```
cd evaluation
run pr_idrid.m
```

## Usage
Pretrained checkpoints on the IDRiD and DDR datasets are available at the snapshot folder.
## License
This code can be utilized for academic research.
