# Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring

[Huicong Zhang](https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&gmla=AETOMgEHtB1sAOmB8EMhprsRACCsD_wLbTGpnaBrkyshm-oVsQtYAjL8q9BRZI6gOiD6nQZSg_urpfJV1FgXa1iGGU6rPo0&user=ASaPjIgAAAAJ),  [Haozhe Xie](https://haozhexie.com), [Hongxun Yao](https://scholar.google.com/citations?user=aOMFNFsAAAAJ)

Harbin Institute of Technology, S-Lab, Nanyang Technological University

![Overview](https://vilab.hit.edu.cn/projects/bsstnet/images/BSSTNet-Teaser.png)



## Update
- [2024/07/04] The training and testing code are released.  
- [2024/02/29] The repo is created. 

## Datasets

We use the [GoPro](https://github.com/SeungjunNah/DeepDeblur_release) and [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/) datasets in our experiments, which are available below:

- [GoPro](https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8?usp=sharing)
- [DVD](https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8?usp=sharing)

You could download the zip file and then extract it to the [datasets](datasets) folder. 

## Pretrained Models

You could download the pretrained model from [here](https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8?usp=sharing) and put the weights in [model_zoos](model_zoos). 

## Prerequisites
#### Clone the Code Repository

```
git clone https://github.com/huicongzhang/BSSTNet.git
```
### Install Denpendencies

```
conda create -n BSSTNet python=3.8
conda activate BSSTNet
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop
```

## Test
To train BSSTNet, you can simply use the following commands:

GoPro dataset
```
scripts/dist_test.sh 2 options/test/BSST/gopro_BSST.yml
```

DVD dataset
```
scripts/dist_test.sh 2 options/test/BSST/dvd_BSST.yml
```

## Train
To train BSSTNet, you can simply use the following commands:

GoPro dataset
```
scripts/dist_train.sh 2 options/test/BSST/gopro_BSST.yml
```

DVD dataset
```
scripts/dist_train.sh 2 options/test/BSST/dvd_BSST.yml
```


## Cite this work

```
@inproceedings{zhang2024bsstnet,
  title     = {Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring},
  author    = {Zhang, Huicong and 
               Xie, Haozhe and 
               Yao, Hongxun},
  booktitle = {CVPR},
  year      = {2024}
}
```

## License

This project is open sourced under MIT license. 

## Acknowledgement
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [ProPainter](https://github.com/sczhou/ProPainter) and [Shift-Net](https://github.com/dasongli1/Shift-Net). 

