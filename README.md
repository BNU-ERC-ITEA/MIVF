# MIVF
Codes for MIVF. (Network Implementation of Video Flare Removal). [arXiv]([[2512.11327\] Physics-Informed Video Flare Synthesis and Removal Leveraging Motion Independence between Flare and Scene](https://arxiv.org/abs/2512.11327)).

**Major Update**



**To do**

1. ~~Data synthesis and generation code.~~
2. ~~Image training code.~~
3. ~~Weights.~~

## Preparing

### 1. Dataset

[Baidu Netdisk](https://pan.baidu.com/s/1lxfngy7yb5AdqtH1JxbC3Q?pwd=3ga9)

[Google Drive](https://drive.google.com/drive/folders/1CAwT3D_pFdC84mwxXsHqSBU7FYTcUvQb?usp=sharing)

Trainsets has been processed by `create_lmdb_for_vlare.py`.

The code of synthesis pipeline and data generation will update in the future.

### 2. Environment

This repository was run with the following environment configurations.

- Cuda                        11.8
- torch                        2.0.1+cu118
- torchaudio              2.0.2+cu118
- torchvision             0.15.2+cu118

To use the selective scan with efficient hard-ware design, the `mamba_ssm` library is needed to install with the following command.

```shell
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

Other necessary python libraries with this `requirement.txt`, run the following command:

```
pip install -r requirement.txt
```

If you want to Evaluate other works(e.g. BasicVSR++), please proceed with the environment installation according to their installation guides.

### 3. Weights

Coming soon.

## Training

You should modify the json file from options first, for example, setting "gpu_ids": [0,1,2,3\] if 4 GPUs are used. 

You can run the `main_train_multiFrame.py` by using:

```shell
python main_train_multiFrame.py --opt options/MIVF/MIVF_v1.json
```

Distributed Training setting details, please refer to  [KAIR]([cszn/KAIR: Image Restoration Toolbox (PyTorch). Training and testing codes for DPIR, USRNet, DnCNN, FFDNet, SRMD, DPSR, BSRGAN, SwinIR](https://github.com/cszn/KAIR).

## Testing

### 1. Evaluation

You can run the `main_test_multiFrame.py` by using:

```shell
python main_infer_multiFrame.py --opt options/MIVF/MIVF_v1.json --folder_lq testsets/vflare_240p/lq --folder_gt testsets/vflare_240p/hq --checkpoints experiments/MIVF/models/MIVF.pth --save_result
```

### 2. Inference

You can run the `main_test_multiFrame.py` by using:

```shell
python main_test_multiFrame.py --opt options/MIVF/MIVF_v1.json --folder_lq testsets/vflare_240p/lq --checkpoints experiments/MIVF/models/MIVF.pth --save_result
```

## References

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{wang2025videoflare,
      title={Physics-Informed Video Flare Synthesis and Removal Leveraging Motion Independence between Flare and Scene}, 
      author={Junqiao Wang and Yuanfei Huang and Hua Huang},
      year={2025},
      eprint={2512.11327},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.11327}, 
}
```

## Acknowledgement

This repository is built based on [KAIR]([cszn/KAIR: Image Restoration Toolbox (PyTorch). Training and testing codes for DPIR, USRNet, DnCNN, FFDNet, SRMD, DPSR, BSRGAN, SwinIR](https://github.com/cszn/KAIR) repository. Thanks for its awesome work.
