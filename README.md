# 实验一：新闻邮件文本分类

仓库地址：https://github.com/ouyangfengan/ecnu-contemporary-AI-01

使用 scikit-learn 比较逻辑回归（LR）和多层感知机（MLP），完成十分类文本预测。仓库包含两份训练脚本及其实际运行结果；课程原始数据和实验报告不在仓库中。

## 仓库内容

```text
README.md
scripts/
  draft_main_LR.py
  draft_main_MLP.py
results/
  metrics/                 # 调参记录、LR 损失及 MLP 逐轮/逐批次损失 CSV
  plots/                   # 两个模型的原始损失图
  predictions/             # LR 与 MLP 的无标签测试集预测
```

## 运行准备

需要 Python 及 `numpy`、`pandas`、`matplotlib`、`scikit-learn`。本实验使用 Python 3.13 和 scikit-learn 1.9.1。在本机已有 `project1` Conda 环境时可先运行 `conda activate project1`；其他环境可安装依赖：

```powershell
python -m pip install numpy pandas matplotlib "scikit-learn==1.9.1"
```

从课程渠道获取数据，在仓库根目录下放置以下文件（不要改名或调整行顺序）：

```text
data/
  train_data.csv             # 列：text,target
  test_data_unlabeled.csv    # 列：text
```

脚本不会创建数据或输出目录。首次运行前，在仓库根目录的 PowerShell 中执行（已有目录不会被清空）：

```powershell
New-Item -ItemType Directory -Force -Path data, results/metrics, results/plots, results/predictions | Out-Null
```

然后将两份数据放入 `data/`，运行任一脚本：

```powershell
python scripts/draft_main_LR.py
python scripts/draft_main_MLP.py
```

## 实验与输出

两个脚本均先按类别分层划分训练集和验证集（8:2，随机种子 42），清理邮件头并保留标题与正文；TF-IDF 词表仅在训练集上拟合，验证集和无标签测试集只做转换，最多使用 5,000 个特征。以验证集 Macro F1 选参：LR 比较 `C=0.1、1、10、100`；MLP 先用单隐藏层 `(100,)` 比较四个 `alpha` 值，再固定选出的 `alpha` 比较 `(50,)`、`(100,)`、`(200,)`、`(100,50)`。

运行后，`results/metrics/` 保存调参记录和损失数据，`results/plots/` 保存损失图，`results/predictions/` 保存两个模型各自的无标签预测（单列、无表头，保持测试集行顺序）。本次实验依据验证集 Macro F1 选用 MLP 的 `alpha=0.001`、隐藏层 `(100,)`（Macro F1 约 0.9202）；课程提交使用 `results/predictions/mlp_predictions.csv`，另存为 `prediction.csv`。脚本复用训练子集上拟合的模型，没有合并训练集与验证集重新训练。

MLP 使用默认小批次大小 `auto`（本实验每批最多 200 条）。`mlp_best_batch_loss.csv` 记录每次更新前的原始批次损失，`mlp_best_loss_curve.csv` 保留模型记录的逐轮平均损失；图中同时展示完整批次轨迹、末 3 轮放大图和参数对比。每轮末批次仅有 94 条，损失含按批大小计算的 L2 项，因此局部尖点不能直接解释为模型不稳定。批次记录依赖 scikit-learn 1.9.1 的内部 `_backprop` 接口，升级版本时需重新确认兼容性。LR 图上的点来自不同迭代预算的独立训练，不是单次训练的逐轮轨迹；图像均不做内部平滑。
