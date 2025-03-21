# Gear Fault Classification using Time Series Models

本项目为本人在北京交通大学机械电子工程专业的本科毕业设计，旨在利用时间序列深度学习 SOTA 模型对弧齿锥齿轮进行多维度分类检测。项目通过对采集的多维时间序列信号进行建模，探索深度学习在机械故障诊断中的应用，实现齿轮故障的自动识别与分类。实验中选取了 **TimesNet**、**PatchMixer** 和 **PatchTST** 三种先进模型，进行性能对比与评估，验证其在工业场景下的实用性和优越性。😊😊😊

---

## 模型简介

- **TimesNet**  
  TimesNet 是一种基于时序块的全局建模方法，能够高效捕捉时间序列中的多尺度特征，在多个时间序列基准任务中均达到 SOTA 性能。  
  📄 论文链接：[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186)

- **PatchMixer**  
  PatchMixer 利用 Patch 分块与特征混合策略，在时间序列分类任务中展现出优越的性能，尤其适合多变量时间序列的高效建模。  
  📄 论文链接：[PatchMixer: A Patch-Mixing Architecture for Time-Series Classification](https://arxiv.org/abs/2305.01740)

- **PatchTST**  
  PatchTST 是一种基于 Patch 的时间序列 Transformer 模型，能够有效提升时间序列预测和分类任务的泛化能力。  
  📄 论文链接：[PatchTST: Contextualizing Time-Series Data with Patch Attention](https://arxiv.org/abs/2211.14730)

---

## 环境配置

### 1. Conda 环境搭建

建议使用 Conda 创建虚拟环境：

```bash
conda create -n gear-classification python=3.9 -y
conda activate gear-classification
```

### 2. 安装依赖库

项目所需的 Python 库可通过以下命令安装：

```bash
pip install -r requirements.txt
```

---

## 文件结构说明

```bash
├── TimesNet/                       # TimesNet 模型代码
│   └── model.py                    # TimesNet 主模型
│   └── layers/                     # TimesNet 模型内部各个模块
│
├── logs_patchmixer/                # PatchMixer 训练日志
├── logs_patchtst/                  # PatchTST 训练日志
│
├── plot/                           # Loss, Acc曲线
│
├── results_20nm(3class)/          # 3分类实验结果（PatchTST的一个小demo~）
├── results_patchmixer/            # PatchMixer 结果
├── results_patchtst/              # PatchTST 结果
├── results_timesnet/              # TimesNet 实验结果
│
├── .gitignore                      # Git 忽略配置
├── LICENSE                         # 开源协议
├── README.md                       # 项目说明文档
│
├── data_processer.py               # 数据预处理脚本
├── plot_results.py                 # 结果绘图脚本
│
├── patchmixer_9class.ipynb         # PatchMixer 9分类实验 notebook
├── patchtst_3class.ipynb           # PatchTST 3分类实验 notebook
├── patchtst_9class.ipynb           # PatchTST 9分类实验 notebook
├── timesnet_9class.ipynb           # TimesNet 9分类实验 notebook
│
└── requirements.txt                # 依赖库列表
```

---

## 使用说明

1. **数据准备**  
   项目根目录下创建 `./data/` 文件夹并将 `.csv` 文件形式的数据集存放其中，供模型训练与测试使用。

   如需获取完整数据集或示例数据文件，请联系项目作者：21222039@bjtu.edu.cn

2. **训练模型**  
   通过 Jupyter Notebook 运行以下文件中的任意一个进行模型训练和评估：

   - `patchmixer_9class.ipynb`
   - `patchtst_3class.ipynb`
   - `patchtst_9class.ipynb`
   - `timesnet_9class.ipynb`

3. **查看结果**  
   - 实验结果保存在 `results_*/` 文件夹中：
     - [📄 PatchMixer 结果展示](./plot/patchmixer_result.pdf)
     - [📄 PatchTST 结果展示](./plot/patchtst_result.pdf)
     - [📄 TimesNet 结果展示](./plot/timesnet_result.pdf)

---
