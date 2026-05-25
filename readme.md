# Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising [Paper](https://arxiv.org/abs/2404.07846)


## 项目扩展（Confocal 显微图像去噪）

在原 TBSN/AP-BSN 基础上增加了 confocal 显微图像的自监督训练与单图推理支持，主要新增和修改如下：

### 新增文件

| 文件 | 说明 |
|---|---|
| `dataset/confocal.py` | Confocal 显微图像数据集，支持 512×512 / 1024×1024 灰度图（png/jpg/tif 等），训练时随机裁剪到 `patch_size` |
| `validate/inference.py` | **单图去噪推理脚本**。支持两种模式：`--tile 0` 全图推理（图尺寸需匹配训练 patch_size）、`--tile N` 分块推理（无重叠拼接）。内置 R3 (Random Replacement Refinement) 后处理，从 config JSON 读取 `R3`/`R3_T`/`R3_p` 参数，默认跟随 config 配置 |
| `denoise.sh` | 推理脚本的命令行包装器，用法：`sh denoise.sh <checkpoint> <input> <output> [config] [tile]` |
| `option/tbsn_confocal.json` | Confocal 训练配置（patch_size=64, pd_a=2, pd_b=2, R3 开启） |
| `option/tbsn_confocal_512.json` | Confocal 训练配置（patch_size=512, pd_a=2, pd_b=2, R3 开启） |

### 修改文件

| 文件 | 变更 |
|---|---|
| `dataset/__init__.py` | 添加 `ConfocalTrainDataset` 导入 |
| `dataset/base_function.py` | 更新 `dataset_path` |
| `model/apbsn.py` | 添加 TensorBoard 日志记录（Loss/LR），添加学习率调度注释 |
| `model/base.py` | 添加 `SummaryWriter` 支持 TensorBoard 日志 |
| `train.sh` | 默认 config 改为 `tbsn_confocal.json` |

### 推理用法

```bash
# 全图推理（图尺寸 = 训练 patch_size，如 512×512）
python validate/inference.py \
    --config option/tbsn_confocal_512.json \
    --checkpoint path/to/checkpoint.pth \
    --input noisy.tif \
    --output denoised.tif
    --tile 0

# 分块推理（图尺寸 > 训练 patch_size，如 1024×1024 图用 64×64 模型）
python validate/inference.py \
    --config option/tbsn_confocal.json \
    --checkpoint path/to/checkpoint.pth \
    --input noisy.tif \
    --output denoised.tif \
    --tile 64

# 或使用 shell 包装器
sh denoise.sh checkpoint.pth noisy.tif denoised.tif tbsn_confocal.json 64
```

### R3 后处理

R3 (Random Replacement Refinement) 在初始去噪后，随机将部分像素替换回原始噪声值并重新通过网络，重复 T 次后取平均。由 config JSON 中的三个参数控制：

- `R3`: 是否启用（`true`/`false`）
- `R3_T`: 迭代次数（默认 8）
- `R3_p`: 每次替换像素的比例（默认 0.16，即 16%）

当前所有 confocal config 默认开启 R3。

### 注意事项

- TBSN 网络内部有 `PatchUnshuffle/PatchShuffle`（固定 2x 降/升采样），要求输入尺寸能被 4 整除。`--tile` 参数也需要能被 4 整除（如 64、128）。
- 分块推理时 R3 在每块上独立执行。由于网络 attention 的感受野限制，极小的 tile 可能在边界产生伪影，建议 tile 尺寸与训练 patch_size 一致。


## Usage
### Datasets
Download [SIDD](https://abdokamel.github.io/sidd/) and [DND](https://noise.visinf.tu-darmstadt.de/) datasets, and modify `dataset_path` in `dataset/base_function.py` accordingly.
```
|- dataset_path
  |- SIDD
    |- SIDD_Medium_Srgb
      |- Data
        |- 0001_001_S6_00100_00060_3200_L
        |- 0002_001_S6_00100_00020_3200_N
        |- ...
    |- SIDD_Validation
      |- ValidationNoisyBlocksSrgb.mat
      |- ValidationGtBlocksSrgb.mat
    |- SIDD_Benchmark
      |- BenchmarkNoisyBlocksSrgb.mat
  |- DND
    |- info.mat
    |- images_srgb
```

### Validation
Validate on SIDD Validation dataset,
```
cd validate
python base.py --config_file "../option/tbsn_sidd.json"
```

### Training
Training on SIDD Medium dataset,
```
sh train.sh
```

## Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{li2025rethinking,
  title={Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising},
  author={Li, Junyi and Zhang, Zhilu and Zuo, Wangmeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
