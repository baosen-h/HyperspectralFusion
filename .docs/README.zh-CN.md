# HyperspectralImageFusion

<p align="center">
  <a href="README.md">English</a> · Chinese
</p>

HyperspectralImageFusion 是一个用于传统高光谱图像融合与 HSI-MSI 融合方法复现的 Python 工具。

本项目面向实验和方法对比。项目中的方法依据对应论文和公开算法描述进行复现。

```text
put input .mat files in DATA/INPUT
run hyperspectral_image_fusion.py
get fused .mat results in DATA/OUTPUT
```

## 方法

- `sfim`: smoothing filter-based intensity modulation
- `mapsmm`: MAP estimation with stochastic mixing model
- `gsa`: Gram-Schmidt adaptive fusion
- `cnmf`: coupled nonnegative matrix factorization
- `glp`: MTF-generalized Laplacian pyramid hypersharpening
- `fuse`: fast fusion of multi-band images
- `hysure`: convex subspace-based fusion

默认参数用于稳定的 HSI/MSI 融合实验。也可以使用 `--mode` 修改 SFIM/GLP 共用的模式参数。

## 对比结果

![Hyperspectral fusion comparison](../IMG/comparison.png)

下表是 Pavia University dataset example 上的当前 Python 结果。

| Method | SAM↓ | PSNR↑ | ERGAS↓ | CC↑ | RMSE↓ | SSIM↑ |
|---|---:|---:|---:|---:|---:|---:|
| SFIM | 2.9939 | 37.76 | 3.4266 | 0.9728 | 0.0261 | 0.9257 |
| MAPSMM | 3.4522 | 38.43 | 3.9386 | 0.9628 | 0.0313 | 0.9260 |
| GSA | 3.1182 | 38.56 | 3.1894 | 0.9769 | 0.0241 | 0.9294 |
| CNMF | 3.8827 | 37.03 | 4.1661 | 0.9596 | 0.0328 | 0.9195 |
| GLP | 3.1788 | 36.29 | 3.3004 | 0.9798 | 0.0226 | 0.9385 |
| FUSE | 3.4954 | 37.56 | 3.8107 | 0.9675 | 0.0291 | 0.9280 |
| HySure | 2.8935 | 38.31 | 3.2831 | 0.9757 | 0.0253 | 0.9454 |

## 参考论文

- `sfim`: J. G. Liu, "Smoothing Filter-based Intensity Modulation: A Spectral
  Preserve Image Fusion Technique for Improving Spatial Details," International
  Journal of Remote Sensing, 2000.
  [[论文](https://doi.org/10.1080/014311600750037499)]
- `mapsmm`: M. T. Eismann and R. C. Hardie, "Application of the Stochastic
  Mixing Model to Hyperspectral Resolution Enhancement," IEEE TGRS, 2004.
  [[论文](https://doi.org/10.1109/TGRS.2004.830644)]
- `gsa`: B. Aiazzi, S. Baronti, and M. Selva, "Improving Component
  Substitution Pansharpening Through Multivariate Regression of MS + Pan Data,"
  IEEE TGRS, 2007.
  [[论文](https://doi.org/10.1109/TGRS.2007.901007)]
- `cnmf`: N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled Nonnegative Matrix
  Factorization Unmixing for Hyperspectral and Multispectral Data Fusion,"
  IEEE TGRS, 2012.
  [[论文](https://doi.org/10.1109/TGRS.2011.2161320)]
- `glp`: M. Selva, B. Aiazzi, F. Butera, L. Chiarantini, and S. Baronti,
  "Hypersharpening: A First Approach on SIM-GA Data," IEEE JSTARS, 2015.
  [[论文](https://doi.org/10.1109/JSTARS.2015.2440092)]
- `fuse`: Q. Wei, N. Dobigeon, and J.-Y. Tourneret, "Fast Fusion of Multi-Band
  Images Based on Solving a Sylvester Equation," IEEE TIP, 2015.
  [[论文](https://doi.org/10.1109/TIP.2015.2458572)]
- `hysure`: M. Simoes, J. Bioucas-Dias, L. B. Almeida, and J. Chanussot,
  "A Convex Formulation for Hyperspectral Image Superresolution via
  Subspace-Based Regularization," IEEE TGRS, 2015.
  [[论文](https://doi.org/10.1109/TGRS.2014.2375320)]

如果将本项目用于研究，请引用对应的原始论文。

## 项目结构

```text
HyperspectralImageFusion
├─ DATA
│  ├─ INPUT
│  │  ├─ HR_MSI.mat
│  │  └─ LR_HSI.mat
│  ├─ REF.mat
│  └─ OUTPUT
├─ hyperspectral_image_fusion.py
├─ FUNCTION
│  ├─ common.py
│  ├─ sfim.py
│  ├─ mapsmm.py
│  ├─ gsa.py
│  ├─ cnmf.py
│  ├─ glp.py
│  ├─ fuse.py
│  └─ hysure.py
├─ IMG
│  └─ comparison.png
├─ .docs
│  ├─ README.md
│  └─ README.zh-CN.md
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 输入数据

使用 `.mat` 文件：

```text
DATA/INPUT/LR_HSI.mat
DATA/INPUT/HR_MSI.mat
```

仓库中已包含这些示例 HSI/MSI 输入文件，安装依赖后可以直接运行下面的命令。

`DATA/REF.mat` 仅用于在生成融合结果后进行可选的视觉对比或指标对比，不是运行融合命令的必要输入。

期望的数据维度为：

```text
rows x cols x bands
```

如果数组保存为：

```text
bands x rows x cols
```

当第一维是最小维度时，脚本会把 band 维移动到最后。

普通 `.mat` 文件和 v7.3 `.mat` 文件都可以读取。如果 `.mat` 文件中包含多个变量，请传入 `--hsi-key` 和 `--msi-key`。

## 使用方法

运行一个方法：

```bash
python hyperspectral_image_fusion.py --hsi DATA/INPUT/LR_HSI.mat --hsi-key data --msi DATA/INPUT/HR_MSI.mat --msi-key data --method cnmf
```

运行全部方法：

```bash
python hyperspectral_image_fusion.py --hsi DATA/INPUT/LR_HSI.mat --hsi-key data --msi DATA/INPUT/HR_MSI.mat --msi-key data --method all
```

可用方法名称：

```text
sfim, mapsmm, gsa, cnmf, glp, fuse, hysure
```

结果保存为：

```text
DATA/OUTPUT/<input_name>_<method>.mat
```

输出变量名为：

```text
data
```

## Python 接口

```python
from hyperspectral_image_fusion import run_method
from FUNCTION.cnmf import cnmf
from FUNCTION.sfim import sfim

fused_cnmf = cnmf(lr_hsi, hr_msi)
fused_sfim = sfim(lr_hsi, hr_msi)
fused = run_method("mapsmm", lr_hsi, hr_msi)
```

## 许可证

本项目使用 MIT License。详见 [LICENSE](../LICENSE)。
