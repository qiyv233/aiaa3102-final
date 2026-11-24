# SuperAgent 整合与验证

本版从 Harry Wang 的 `superagent-paired-text` 提交 `6565203` 接续开发。两道已发布课程题共用一个 Agent。训练阶段读取题目说明和训练材料，程序按训练数据确定候选与参数，锁定后才打开课程验证集。本次四题均使用本地规划器，付费调用为零。

## 两个版本如何取舍

| 比较对象 | 在相同数据上的结果 | 整合决定 |
| --- | --- | --- |
| 标题纯 TF‑IDF 与原混合模型 | 对方在课程验证集答对 650/940，我们答对 644/940。换成训练集内相同的四折分组，混合模型答对 4,136/6,225，纯 TF‑IDF 答对 4,066/6,225。 | 收入纯 TF‑IDF 作为可选模型，本次锁定混合模型。 |
| 优惠券树模型与原分组响应模型 | 对方答对 1,512/2,030，我们答对 1,558/2,030。我们的方法也在训练确认集比自己的基线多答对 15 条。 | 默认使用原分组响应模型，保留可重放的树模型基线。 |
| 压缩包、数据格式、预测接口 | 对方支持直接从课程 ZIP 读 CSV 和成对的 NPY；我们已有自动读题、分组选模、锁定模型和预算记录。 | 共用数据读取与预测入口。既有模型仍可重放，新模型可以从同一 ZIP 独立导出预测。 |
| 验证与提交 | 对方的课程验证成绩来自参与挑选模型的验证集；我们的选模流程还保留一份训练确认集。 | 课程主流程分别执行训练与验证，记录实际读取的文件。最终评分路径只在合成数据上演练。 |

标题的两种方法确实在不同样本上有效。训练四折中，换用纯 TF‑IDF 会纠正 326 条，也会新增 396 条错误。另留的 2,076 条训练确认样本上，纯 TF‑IDF 多答对 4 条（1,400 对 1,396），低于预先设定的 0.5 个百分点替换门槛。划分时，将共享任一标准化标题的样本归在同一组，四折及确认集的组间重叠均为零。

这次把“下一步做哪个实验”与具体的训练错误关联。对准确率任务，程序在训练反馈折分别统计相似与不同措辞（文本对），或含稀有类别与常见类别（分类表格）的答对数。两组准确率的 95% Wilson 区间分离且对应候选可用时，优先试针对较弱一组的表示；证据不足时沿用原候选顺序。区间只是安排试验的启发式规则。晋级仍由同样的四折和训练确认门槛决定。

标题基线在相似措辞组答对 74/141，在其余组答对 965/1,416。旧版先试纯 TF-IDF，本版先试着重词项差异的 `contrast_rank`。它在训练反馈折答对 1,027/1,557，基线答对 1,039/1,557，因此继续比较其他候选。日志保留各次试验的依据、实际效果和入选决定。

## 最终版本实测

| 任务 | 全量训练行数 | 锁定方法 | 验证答对 / 总数 | 准确率 |
| --- | ---: | --- | ---: | ---: |
| 标题点击率二选一 | 8,301 | 配对混合模型 `pair_hybrid` | 644 / 940 | 68.51% |
| 车载优惠券接受预测 | 8,117 | 分组响应模型 `profile_response_pool` | 1,558 / 2,030 | 76.75% |
| HWU64 意图分类（公开迁移） | 8,954 | 词与字符特征 `union` | 969 / 1,076 | 90.06% |
| Banking77 意图分类（公开迁移） | 8,622 | `union` 与文本近邻 | 1,398 / 1,540 | 90.78% |

四份验证预测文件与整合前保存的对应结果逐字节相同，包含各类概率。课程两题另从锁定模型重新导出，得到相同文件。课程验证集先前已用于项目比较，不能据此估计隐藏测试成绩。两项公开实验仅用于检验同一入口迁移到其他分类题的表现。

另用 scikit-learn 自带的十类手写数字验证图像路径。分层随机划分（种子 20261124）得到 1,437 张训练图片与 360 张验证图片。相同图片采用目录和 ZIP 各自训练，均答对 358/360（99.44%）。此实验专门核对不同输入封装、分组和重放的等价性；该准确率只描述这次手写数字划分。

## 组员怎样复核

1. 按 [README 的课程步骤](README.md#整合版与课程任务)安装依赖并设置 `$py` 和 `$archive`。在新目录分别运行两题的 `train` 与 `validate`。每题输出模型、训练分组、候选日志、最终决定与验证预测。课程归档由组员自行提供。
2. 运行下列命令检查读题、旧模型兼容、锁定模型导出，以及标题模型在相同样本上的差异。标题比较使用上一步生成的模型目录和训练材料。

~~~powershell
& $py verify_intake.py --course $archive --out runs/intake-checks.json
& $py checks/verify_integration.py
& $py Agent/verify_framework.py
& $py checks/compare_titles.py --run runs/final/headline --data $archive --out runs/title-comparison.json
~~~

3. 复核公开任务时，将指定修订的 README 和 `data/train-00000-of-00001.parquet` 保存在 `inputs/<任务>/` 下，保留文件层级。将官方评分脚本保存为 `benchmark/evaluate.py`，供读题程序识别指标。验证 Parquet 单独保存到 `public-validation/<任务>.parquet`。脚本会核对验证文件和模型的 SHA‑256。[HWU64 数据](https://huggingface.co/datasets/FastFit/hwu_64/tree/9fb5bff1e37c4c5a6d46c1ff6f286b6d3f362222) · [Banking77 数据](https://huggingface.co/datasets/FastFit/banking_77/tree/2797084dd7eb150c200cdf538ed2afcf7255b13c) · [DialoGLUE 评分脚本](https://github.com/alexa/dialoglue/blob/9074e4f27c973c087e9b814ce286219c6643380b/evaluate.py)

~~~powershell
foreach ($task in 'hwu','banking_77') {
    & $py Agent/agent.py develop --data "inputs/$task" --out "runs/public/$task" --provider local --strengthen --max-actions 3 --seconds 1200 --train-only
    if ($LASTEXITCODE -ne 0) { throw "训练中止 $task" }
    & $py checks/score_public_text.py $task --run "runs/public/$task" --data "inputs/$task" --validation "public-validation/$task.parquet"
    if ($LASTEXITCODE -ne 0) { throw "验证中止 $task" }
}
~~~

[四题汇总、模型与预测校验值、课程文件读取记录](evidence/integration.json)

后续对标题相近措辞的两个模型做了独立训练对照。两者均未达到替换门槛，交付模型继续使用原混合模型。[逐折结果与复核命令](studies/title_similarity/README.md)
