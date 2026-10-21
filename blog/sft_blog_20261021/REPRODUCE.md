# 运行与复现说明

## 1. 阅读归档无需安装深度学习环境

Markdown、JSON 和 CSV 可以直接在本地查看。结果校验脚本仅依赖 Python 标准库。

## 2. 源码快照与历史实验的区别

`code/learning/` 是 2026-10-21 整理时从当前工作区复制的文件，包含用户当时的未提交代码。没有把 Git 提交版本当作这批文件的精确历史版本。实验配置与数值来自各次运行的配置、日志和 W&B 最终摘要。

`run_five_sizes.sh` 和 `learning/__init__.py` 是整理时新增的辅助文件。五组脚本保持主要历史参数，默认 W&B offline，替换了原机绝对路径和账户设置。未实际重新执行 GPU 训练，因此不声称能在任意机器上开箱即用，也不保证重跑得到完全相同数字。

## 3. 训练环境

原运行使用 Python 3.11、NVIDIA A100 80GB，训练与 vLLM 评估在同一张可见 GPU 上。主要包版本见 `code/environment_versions.txt`。`pyproject.original.toml` 是原仓库依赖声明的参考副本，其中包含其他作业任务依赖；不是本素材包的可安装项目配置。

训练涉及 PyTorch、Transformers、vLLM、flash-attn、W&B、NumPy、tqdm，以及评分器依赖 sympy、math-verify、latex2sympy2_extended、pylatexenc。CUDA、PyTorch、vLLM 和 flash-attn 需要相互兼容。源代码使用 vLLM 内部接口，不宜直接换最新版后假定完全兼容。

## 4. 准备数据与模型

本包未附训练数据、测试数据或模型权重。复现实验应使用原始 `Qwen2.5-Math-1.5B` 本地模型目录，包含权重、配置和 tokenizer 文件。

训练 JSONL 每行至少需要：

```json
{"prompt": "模型接收的完整提示", "response": "作为监督目标的回答"}
```

可选字段 `is_correct` 只在开启 `--filter_correct` 时参与筛选。训练程序不会再给训练 `prompt` 套模板，因此准备数据时应保证提示格式正确。

评估 JSONL 每行至少需要：

```json
{"question": "题目", "answer": "推导文本 #### 最终答案"}
```

程序读取最后一个 `####` 后的内容作为标准答案；若无分隔符，则使用整个 answer。评估提示由 `code/cs336_alignment/prompts/r1_zero.prompt` 生成。

原数据文件的行数、大小和 SHA-256 见 `results/data_manifest.json`。重新生成的训练推理文本即使条数相同，也不等于同一训练集。

## 5. 启动五组实验

在已准备好的 GPU 环境中运行：

```bash
cd code
MODEL_PATH=/path/to/Qwen2.5-Math-1.5B \
TRAIN_DATA_PATH=/path/to/train_sft_reason_gsm8k_raw.jsonl \
VAL_DATA_PATH=/path/to/test.jsonl \
PYTHON_BIN=/path/to/python \
bash run_five_sizes.sh
```

脚本默认只暴露 GPU 0，依次跑 128、256、512、1024、7473 五组。可以用 `OUTPUT_ROOT` 改输出目录；已存在同名组目录时会停止，避免覆盖。它默认保存 W&B 离线记录；如需在线同步，请自行配置账户并设置 `WANDB_MODE=online`。

这是复现历史设置的启动方式，因此仍使用测试集每 8 步评估一次。开展新实验时，应把 `VAL_DATA_PATH` 指向独立验证集，并额外实现最佳 checkpoint 保存，最终再测测试集。

## 6. 结果精度和统计口径

- `final_metrics.json` 和汇总最终指标取自 W&B summary，保留其原始数值精度。
- `accuracy_log_4dp` 是终端日志四位小数的准确率；`avg_length_chars_log_1dp` 是日志一位小数的平均字符长度。
- 每组固定 1319 题、奖励为二值，题数间隔约为 0.000758，大于日志四舍五入误差。因此可唯一重建正确题数：`round(accuracy_log_4dp * 1319)`。`accuracy_reconstructed` 为该题数除以 1319；并已和最终完整精度结果交叉校验。
- 峰值按已保存的 33 个评估点计算。`best_steps` 保留并列最高点，样例目录仅取首次最高点，且去除与初始/最终步骤的重复。
- 摘录日志保留全部评估行；机器元数据、账号元数据及训练进度条没有复制进去。来源文件的哈希保存在 provenance 中。
- 五组只有单次运行，不能提供跨种子误差条；所保存的前 100 题样例不足以做全测试集配对显著性检验。

## 7. 代码来源与署名

SFT 实现为用户工作区代码快照。`drgrpo_grader.py` 文件头说明了上游来源，并保留 Garena Online Private Limited 的 Apache-2.0 版权许可声明。发布时应保留这些声明，不把依赖代码描述成个人原创。
