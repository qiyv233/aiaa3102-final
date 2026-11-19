# aiaa3102 final

AIAA 3102 期末项目。我们用同一个分类 Agent 处理表格、文本和图像任务，并按题目给定的指标自动选择模型。

- [课程项目要求原文](课程项目要求.md)包含任务、评分比例、预算、提交材料和时间安排。
- [推荐方案与验证简报（Word）](AIAA3102推荐方案与验证简报.docx)总结了现有结果、研究方法和三个可并行开展的方向。
- [完整组员分享包](https://github.com/qiyv233/aiaa3102-final/releases/tag/v1.0.0)包含已训练模型、离线权重和原始归档。下载 Release 中的 `aiaa3102-final-full.zip`，解压后进入“组员分享”目录，即可按下文重放模型或从原始数据复现全部实验。

仓库直接提供代码、整理后的示例、实验日志和预测。下文的首次表格训练可在克隆后直接运行。涉及历史模型、图像权重或原始归档的命令请在完整分享包中运行。文件校验清单对应完整包内的文件。

先读同目录的《AIAA3102推荐方案与验证简报.docx》。本包是组会与开发材料。42 次本轮验证都有预测、逐候选日志和最终模型；另外保留了上一版 18 次对照的评分依据。三条研究方向和验收方式见简报第 3 页，成员可以自由认领。

## 第一次运行

在本目录打开 PowerShell，使用 64 位 Python 3.11。下面的命令均使用相对路径。首次安装依赖需要网络，随后的表格和文本实验可以离线运行。

```powershell
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r Agent/requirements.txt
& $py Agent/agent.py run --task 验证/bank/task-113.json --pool reviewed --strategy rule --seconds 180 --out runs/bank01
& $py Agent/agent.py predict --model runs/bank01/model.joblib --input 验证/bank/test.csv --output runs/bank01/replay.csv
Get-FileHash runs/bank01/predictions.csv,runs/bank01/replay.csv
```

两个哈希应相同。重新训练时换一个新的输出目录。文本可换为 `验证/movies/task-113.json`，图像可换为 `验证/cifar/task-113.json`。已有结果在对应任务的 `113-reviewed-rule` 中。运行 `Agent/start.py table` 也可以练习读入、训练与模型重放。使用推荐策略时显式传 `--pool reviewed --strategy rule`，三次候选尝试为默认上限。

## 图像环境

本轮使用 torch 2.5.1+cu121、torchvision 0.20.1+cu121。需要重现图像迁移路线时，按设备选择兼容构建。下面为 CPU 环境示例。已安装合适版本的同学只需设置权重路径。CUDA 与 CPU 的运行时间和浮点结果可能不同。

```powershell
& $py -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
$env:AIAA_RESNET18_WEIGHTS = (Resolve-Path weights/resnet18-f37072fd.pth).Path
& $py Agent/agent.py run --task 验证/cifar/task-113.json --pool reviewed --strategy rule --max-trials 3 --seconds 180 --out runs/image01
```

权重用于重新训练。已保存的迁移模型包含冻结编码器，重放需要 torch 和 torchvision。没有图像深度学习依赖时，推荐策略可以试轻量梯度特征。[PyTorch 历史安装指令](https://pytorch.org/get-started/previous-versions/) · [torchvision 权重与预处理](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## 怎样检查证据

| 文件或目录 | 内容 |
| --- | --- |
| 验证/bank、youtube、cifar | 三种既有策略共 27 次。彩色图像另有 3 次迁移专项。每题还有 2 次种子 113 配对复核。summary.json 和 confirmation.json 保存分数、策略、赢家与用时。 |
| 验证/wdbc、movies | 方法冻结后加入的两份整任务，各比较三种策略。independent.json 保存所有结果。 |
| 各次运行目录 | model.joblib、predictions.csv、run.jsonl 与 experiments.jsonl。种子 113 的 12 次验证另存了新进程产生的 replay.csv。 |
| protocol.txt、reviewed-protocol.txt、independent-protocol.txt | 按开发时间先后固定的三份协议。前两步使用过的外层数据属于开发材料。 |
| 历史依据 | 18 次旧对照的预测、重放、模型、留出答案和汇总。audit.json 是独立复算与重叠检查结果。 |
| 历史脚本回归 | 另有 12 次跨来源评论压力测试重跑。用同一正常入口验证旧脚本，并复核约 5∶1 人为训练失衡下的加权收益。 |
| 原始数据、weights、Agent/examples | 新任务的原始归档、预训练权重和以前的独立小示例。训练清单均未指向 evaluation_labels.csv，该文件由外部评分脚本读取。 |
| 文件校验.json | 每项交付文件的 SHA-256，可核对副本是否完整。 |

以下命令会生成新的运行目录。CPU 上整套对照可能比开发机慢。比较时遵循原协议，同题各策略使用同一训练上限、划分和指标。

```powershell
& $py rescore.py 历史依据 验证 audit-check.json
& $py new_tasks.py bank 原始数据 runs/rebuild
& $py new_tasks.py youtube 原始数据 runs/rebuild
& $py new_tasks.py cifar 原始数据 runs/rebuild
& $py check_recommendation.py runs/rebuild bank youtube cifar
& $py independent.py wdbc 原始数据 runs/sealed
& $py independent.py movies 原始数据 runs/sealed
& $py Agent/reviews_study.py Agent/data/sentiment-uci.zip runs/review-stress
```

## 正式运行和可选规划器

支持 accuracy、macro_f1、log_loss 和 roc_auc。先核对课程任务公布的指标、划分、正类及提交模板。演示的 180 秒是本轮训练对照预算，导出预测也占墙钟时间。日志中的验证分数用于选模，外层分数只由评分器计算。运行中基线也未完成时，没有检查点可用。

使用 DeepSeek 前，由组员确认模型快照、课程计价口径及实际平台。按 `Agent/planner.example.json` 的字段准备本机配置，环境变量保存密钥。相同评价题跨进程使用同一 budget_task_id 和持久账本 AIAA_BUDGET_DB。由真实调用记录核对输入和输出 token 及每题 20 元上限。当前代码使用配置的一对费率，若课程要求分时计价，须先记录并实现相应的时间口径。包内本轮 42 次运行均离线，LLM 调用与费用均为零。

## 数据及代码来源

- 纸币与视频评论按 UCI 发布数据制备；视频文本做了 Unicode 规范化去重。出处与作者见数据卡。[Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) · [YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection)
- CIFAR-10 由 Alex Krizhevsky、Vinod Nair、Geoffrey Hinton 创建。使用 Python 归档，按官方训练与测试分开抽取四类共 1,200 张，原始归档 MD5 为 c58f30108f718f92721af3b95e74349a。[数据主页](https://www.cs.toronto.edu/~kriz/cifar.html)
- WDBC 来自 Wisconsin Diagnostic Breast Cancer 数据，经 scikit-learn 加载。此处只评估公开分类基准。[scikit-learn 数据说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- Cornell polarity v2.0 由 Bo Pang 与 Lillian Lee 整理，经 NLTK 分发，保留正负标签，以原始 cv 编号划分。NLTK 数据目录标注 CC BY 4.0。[作者主页](https://www.cs.cornell.edu/people/pabo/movie-review-data/) · [NLTK 数据目录](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml)
- 候选模型使用 scikit-learn、PyTorch 与 torchvision，表格类别特征扩展可选 CatBoost 1.2.8。六篇方法论文及其如何影响设计见简报第 4 页。[scikit-learn](https://github.com/scikit-learn/scikit-learn) · [torchvision](https://github.com/pytorch/vision) · [CatBoost](https://github.com/catboost/catboost)
