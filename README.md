# RefinerViT
This repo is the official implementation of ["Refiner: Refining Self-attention for Vision Transformers"](https://arxiv.org/abs/2106.03714). The repo is build on top of [timm](https://github.com/rwightman/pytorch-image-models) and include the relabbeling trick included in [TokenLabelling](https://arxiv.org/abs/2104.10858). 


## Introduction

**Refined Vision Transformer** is initially described in [arxiv](https://arxiv.org/abs/2106.03714), which observes vision transformers require much more datafor model pre-training. Most of recent works thus are dedicated to designing morecomplex architectures or training methods to address the data-efficiency issue ofViTs. However, few of them explore improving the self-attention mechanism, akey factor distinguishing ViTs from CNNs.  Different from existing works, weintroduce a conceptually simple scheme, calledrefiner, to directly refine the self-attention maps of ViTs.  Specifically, refiner exploresattention expansionthatprojects the multi-head attention maps to a higher-dimensional space to promotetheir diversity.  Further, refiner applies convolutions to augment local patternsof the attention maps, which we show is equivalent to adistributed local atten-tion—features are aggregated locally with learnable kernels and then globallyaggregated with self-attention.  Extensive experiments demonstrate that refinerworks surprisingly well. Significantly, it enables ViTs to achieve 86% top-1 classifi-cation accuracy on ImageNet with only 81M parameters.

<p align="center">
<img src="https://github.com/zhoudaquan/Refiner_ViT/blob/master/figures/overall_flow.png" | width=500>
</p>

Please run git clone with --recursive to clone timm as submodule and install it with ` cd pytorch-image-models && pip install -e ./`


#### Requirements

torch>=1.4.0
torchvision>=0.5.0
pyyaml
numpy
timm==0.4.5

A summary of the results are shown below for quick reference. Details can be found in the paper.

| Model                          | head  | layer|   dim |Image resolution| Param | Top 1 |
| :--------------------------    | :-----|:---  |:------|:--------------:| -----:| -----:|
| Refiner-ViT-S                  |  12   | 16   | 384   |    224         | 25M   | 83.6  |
| Refiner-ViT-S                  |  12   | 16   | 384   |    384         | 25M   | 84.6  |
| Refiner-ViT-M                  |  12   | 32   | 420   |    224         | 55M   | 84.6  |
| Refiner-ViT-M                  |  12   | 32   | 420   |    384         | 55M   | 85.6  |
| Refiner-ViT-L                  |  16   | 32   | 512   |    224         | 81M   | 84.9  |
| Refiner-ViT-L                  |  16   | 32   | 512   |    384         | 81M   | 85.8  |
| Refiner-ViT-L                  |  16   | 32   | 512   |    448         | 81M   | 86.0  |

#### Training

Train the Refiner-ViT-S from scratch: 

```
bash run.sh scripts/refiner_s.yaml 
```

To use the re-labbeling tricks for improving the accuracy, download the [relabel_data](https://drive.google.com/file/d/1Cat8HQPSRVJFPnBLlfzVE0Exe65a_4zh/view) based on NFNet. This is provided in TokenLabelling repo. Then, copy the relabbeling data to the data folder.

