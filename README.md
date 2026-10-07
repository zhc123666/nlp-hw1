# 作业一 文本分类

实现三种词袋表示、三组 100 维词向量表示和 BERT 微调，全部使用同一份 NYT 训练、验证、测试划分。实验报告为同目录的 `实验报告.pdf`。

## 文件

| 路径 | 用途 |
| --- | --- |
| `experiment.py` | 数据校验、固定划分、7 组训练、评价、模型保存 |
| `download_resources.py` | 下载官方 GloVe 100d 和固定版本的 BERT 权重 |
| `build_report.py` | 从实际预测重新校验指标并生成中文 PDF |
| `predict.py` | 加载模型，对新的文本分类 |
| `verify_outputs.py` | 重新加载 7 个保存模型，核对全部测试预测 |
| `make_submission.py` | 打包报告、代码、划分和评价记录 |
| `run_all.ps1` | Windows 一键运行 |
| `requirements.txt` | 依赖版本 |
| `tests/test_experiment.py` | 数据隔离、表示定义、指标与模型保存检查 |
| `HW-1/nyt.csv` | 课程 NYT 数据，必须有 `text,label` 两列 |
| `HW-1/ag.csv` | 课程 AG News 数据，必须有 `text` 列 |
| `outputs/split.json` | 固定划分、类别统计、数据与划分 SHA-256 |
| `outputs/split_assignments.csv` | 每条原始记录所属集合，row_id 从 0 开始 |
| `outputs/results.csv` | 7 种方法的验证与测试指标 |
| `outputs/metrics/` | 各方法配置、分类报告、混淆矩阵和逐条预测 |
| `outputs/bert_history.json` | BERT 三轮训练记录 |
| `outputs/duplicate_sensitivity.json` | 排除与训练集重复文本后的评价 |
| `outputs/verification.json` | 7 个保存模型的预测一致性检查结果 |
| `outputs/models/` | 训练后的模型与词表，本机保留，不上传普通 GitHub |

## 环境安装

建议 Python 3.12。Windows PowerShell 中进入本目录，创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# 有 NVIDIA GPU 时先安装与驱动兼容的 PyTorch CUDA 版本。
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU 也可以运行，但 BERT 微调耗时明显增加。CPU 安装可将 PyTorch 的 index-url 改为 `https://download.pytorch.org/whl/cpu`。
本机已有 CUDA PyTorch，因此本次使用隔离虚拟环境继承现有科学计算依赖，只在虚拟环境内安装了指定 Transformers、tokenizers 和 huggingface-hub 版本。各组结果 JSON 均记录实际环境。

PDF 生成在 Windows 自动使用微软雅黑并嵌入字体；其他系统使用 ReportLab 自带中文 CID 字体。若跨平台显示字体不同，可在 `build_report.py` 配置本机的中文 TrueType 字体。

## 运行

确认完整数据位于 `HW-1/nyt.csv` 与 `HW-1/ag.csv`。程序拒绝 `.wxdownload` 等未完成下载，也不会悄悄丢弃异常行。

一键运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

也可以分步骤执行：

```powershell
$env:PYTHONHASHSEED = '42'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$env:TOKENIZERS_PARALLELISM = 'false'
.\.venv\Scripts\python.exe download_resources.py all
.\.venv\Scripts\python.exe experiment.py --task prepare
.\.venv\Scripts\python.exe experiment.py --task bow
.\.venv\Scripts\python.exe experiment.py --task embeddings
.\.venv\Scripts\python.exe experiment.py --task bert
.\.venv\Scripts\python.exe build_report.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

指定其他数据位置：

```powershell
.\.venv\Scripts\python.exe experiment.py --nyt 'D:\data\nyt.csv' --ag 'D:\data\ag.csv' --task all
.\.venv\Scripts\python.exe build_report.py --nyt 'D:\data\nyt.csv'
```

资源已下载时自动复用。BERT 默认读取 `models/bert-base-uncased`，也可通过 `--bert-model google-bert/bert-base-uncased` 让 Hugging Face 下载。GloVe 默认位于 `data/glove.6B.100d.txt`，可用 `--glove` 指定。
下载器从 Stanford 官方 ZIP 按范围获取所需的 100d 成员，校验原始长度与 CRC；BERT 下载固定版本 `86b5e0934494bd15c9632b12f734a8a67f723594` 的 safetensors 权重。

单独跑某组词向量：

```powershell
.\.venv\Scripts\python.exe experiment.py --task embeddings --embedding word2vec_nyt
```

若更改数据或随机种子，必须指定新的输出目录，例如 `--output outputs_seed43 --seed 43`，避免混用不同划分的结果。报告对应指定的 `--output`。降低显存占用可使用 `--batch-size 8 --gradient-accumulation 4`，有效 batch 仍为 32。

## 方法和选模规则

- 随机种子 42，随机打乱一次；训练 9,215 条、验证 1,151 条、测试 1,153 条。比例按 80%、10%、10% 取整，所有模型共用。
- Binary BoW、Word Frequency、TF-IDF 仅在训练集构建词表和 IDF，不移除停用词。作业总述要求三种词袋方法但细则仅列出两种，故补充 TF-IDF。
- GloVe、AG News Word2Vec、NYT Word2Vec 均使用 100 维词向量，对有效词出现逐次平均。OOV 不计入分母；全 OOV 文档为零向量。
- 两组 Word2Vec 都用 Skip-gram、window=5、min_count=2、negative=5、epochs=10、workers=1；NYT Word2Vec 仅用训练集。
- 全部传统分类器为 Logistic Regression，在验证集上从 C={0.1,1,10} 选择 Macro-F1 最优者；相同时比较 Accuracy。词向量先按训练集统计量标准化，词袋保留稀疏格式。
- BERT 为 `google-bert/bert-base-uncased`，max_length=64，训练恰好 3 轮，AdamW 学习率 2e-5，FP16，batch=16，梯度累积 2。每轮评价验证集，恢复验证集最优检查点后评价一次测试集。
- Accuracy 和 Macro-F1 使用完整类别列表计算，不使用 weighted-F1 替代 Macro-F1。指标文件保留小数原值，PDF 中换算为百分数。
- 测试结果只用于最终比较，没有用它选择 C、训练轮数或检查点。

## 使用保存的模型

```powershell
.\.venv\Scripts\python.exe predict.py --method tfidf --text 'The football team won the championship.'
.\.venv\Scripts\python.exe predict.py --method bert --text 'The central bank announced an interest rate decision.'
```

`glove_subset.joblib` 仅保存本次 NYT 中出现过的 GloVe 词，适合复现实验；对全新领域文本，若需要完整预训练词覆盖，可重新加载完整的 GloVe 100d 文本。不要加载来源不可信的 joblib 或 gensim 模型文件。

## 结果与数据限制

本次实际运行的 NYT 测试集结果如下，BERT 已完成 3 个 epoch；验证集最优检查点来自第 2 轮。

| 方法 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Binary BoW | 98.96% | 97.48% |
| Word Frequency | 98.87% | 97.29% |
| TF-IDF | **99.05%** | **97.54%** |
| GloVe 100d | 98.70% | 96.91% |
| Word2Vec AG News | 98.35% | 95.89% |
| Word2Vec NYT | 98.87% | 96.99% |
| BERT | 98.18% | 96.07% |

6 项代码检查全部通过；7 个保存模型重新加载后，各自的 1,153 条测试预测全部与实验记录一致。可运行以下命令复核模型保存结果，再重新生成提交包：

```powershell
.\.venv\Scripts\python.exe verify_outputs.py
.\.venv\Scripts\python.exe make_submission.py
```

以 `outputs/results.csv` 和 `实验报告.pdf` 为准。原始数据有类别不平衡及少量重复文本。报告给出精确重复文本计数，并复用相同测试预测，排除与训练集重复的测试文本后再次评价；没有在检查中重新训练或选模。
仅运行一个随机种子，BERT 输入限长与传统方法全文输入不同，不能将小分差解释为对所有文本任务的稳定结论。

## GitHub 提交

应提交代码、运行说明、PDF 报告、`outputs` 中的划分与评价记录。`.gitignore` 排除了虚拟环境、预训练权重、训练后大模型、下载缓存和课程原始数据；无需将超过 GitHub 单文件限制的 BERT 权重提交到普通仓库。
原始 CSV 由课程提供，复现时按上述目录放置；如果教师要求仓库也附带课程数据且允许再分发，可用 `git add -f HW-1/nyt.csv HW-1/ag.csv` 单独添加。

在 GitHub 建立用于本次作业的仓库，然后执行：

```powershell
git init
git add .
git commit -m 'Complete NLP homework 1 text classification experiments'
git branch -M main
git remote add origin <替换为自己的仓库地址>
git push -u origin main
```

提交仓库页面链接，并确认教师能访问。`作业一_提交包.zip` 包含报告、代码、运行说明及评价记录，可解压后按需上传。数据与大模型留在本机原目录。
