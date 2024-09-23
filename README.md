# TSC-UAP

# Texture Re-Scalable Universal Adversarial Perturbation
This is the official repository of [Texture Re-Scalable Universal Adversarial Perturbation](https://arxiv.org/pdf/2406.06089.pdf).
The paper is accepted by TIFS 2024.

[![arXiv](https://img.shields.io/badge/arXiv-2406.06089-b31b1b.svg)]([https://arxiv.org/abs/2406.06089](https://arxiv.org/pdf/2406.06089.pdf))

> **Texture Re-Scalable Universal Adversarial Perturbation**<br>
> Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Xiaojun Jia, Xiaochun Cao, Geguang Pu, Yang Liu <br>

>**Abstract**: <br>
> Universal adversarial perturbation (UAP), also known as image-agnostic perturbation, is a fixed perturbation map that can fool the classifier with high probabilities on arbitrary images, making it more practical for attacking deep models in the real world. Previous UAP methods generate a scale-fixed and texture-fixed perturbation map for all images, which ignores the multi-scale objects in images and usually results in a low fooling ratio. Since the widely used convolution neural networks tend to classify objects according to semantic information stored in local textures, it seems a reasonable and intuitive way to improve the UAP from the perspective of utilizing local contents effectively. In this work, we find that the fooling ratios significantly increase when we add a constraint to encourage a small-scale UAP map and repeat it vertically and horizontally to fill the whole image domain. To this end, we propose texture scale-constrained UAP (TSC-UAP), a simple yet effective UAP enhancement method that automatically generates UAPs with category-specific local textures that can fool deep models more easily. Through a low-cost operation that restricts the texture scale, TSC-UAP achieves a considerable improvement in the fooling ratio and attack transferability for both data-dependent and data-free UAP methods. Experiments conducted on two state-of-the-art UAP methods, eight popular CNN models and four classical datasets show the remarkable performance of TSC-UAP.


# Requirements

```
python3
torch == 1.10.0
torchvision == 0.11.2
```

# References
```
@article{10.1109/TIFS.2024.3416030,
author = {Huang, Yihao and Guo, Qing and Juefei-Xu, Felix and Hu, Ming and Jia, Xiaojun and Cao, Xiaochun and Pu, Geguang and Liu, Yang},
title = {Texture Re-Scalable Universal Adversarial Perturbation},
year = {2024},
issue_date = {2024},
publisher = {IEEE Press},
volume = {19},
issn = {1556-6013},
url = {https://doi.org/10.1109/TIFS.2024.3416030},
doi = {10.1109/TIFS.2024.3416030},
month = jun,
pages = {8291–8305},
numpages = {15}
}
```
