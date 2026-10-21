# SFT 学习与五组实验归档

这是可整体下载、离线阅读的博客素材包，整理日期为 2026-10-21。

**先看 [BLOG.md](BLOG.md)**：包含 SFT 代码原理、全部指标定义、五组结果、逐组分析、实验限制和改进方向。文章不依赖在线图片或 W&B 登录。

## 文件布局

```text
BLOG.md                       博客正文，可复制到 Markdown 编辑器
README.md                     本说明
REPRODUCE.md                  环境、数据、运行说明及复现边界
code/
  learning/                   当前 SFT 源码快照及评分依赖
  cs336_alignment/prompts/    原始评估提示模板
  run_five_sizes.sh           新增的五组启动脚本
  environment_versions.txt   7473 组记录的关键包版本
  pyproject.original.toml     原仓库依赖声明参考
results/
  summary.csv / summary.json 五组最终指标与峰值
  eval_history_all.csv        五组共 165 个评估点
  runs/size_*/                各组配置、最终指标、曲线及日志摘录
  samples/                   初始、首次最高点、最终的前 100 题样例
  provenance.json            结果来源及原始日志 SHA-256
  data_manifest.json         原始数据文件大小、条数和 SHA-256
  source_manifest.json       代码快照及其来源、SHA-256
  verification.txt           本包实际执行的校验结果
  checksums.sha256            打包前文件校验和
 tools/verify_results.py      无需 GPU 的结果校验程序
```

## 上传博客

1. 将 `BLOG.md` 粘贴到博客 Markdown 编辑器。所有核心结论和表格都在文章内。
2. 如果要保留正文中的代码与 CSV 链接，上传相应相对目录作为附件，或把链接替换成博客/代码仓库中的实际地址。仅粘贴正文不会自动上传附件。
3. 源码快照和代表性样例可作为 ZIP 附件提供下载。CSV 使用 UTF-8 BOM，便于本地表格软件读取。
4. 文章是依据真实实验产物整理的草稿；发布时可按自己的口吻调整，并保留证据边界。评分依赖的上游版权声明已保留。

## 包含与不包含的内容

- 包含 `learning` 中与 SFT 直接相关的三份源文件，以及运行所需的 `drgrpo_grader.py`；没有混入 Day 1—7 的其他学习练习。
- 原始 Python 文件和四组 Shell 脚本按字节复制，未改变训练逻辑；原脚本内有原机器目录和 W&B 账户配置，跨机器运行请使用新增启动脚本。
- 保存了所有评估点的 accuracy 和平均字符长度、全部最终标量指标，以及指定阶段的生成样例。
- **没有导出完整逐步训练 loss/entropy 曲线。** 日志中的最终值不能替代全程曲线；如需这些曲线，应另从 W&B 历史导出。
- 日志文件是准确率行的摘录，不是原始全量终端输出；没有打包 W&B 登录记录、机器信息或缓存。
- 未包含模型权重和训练/测试数据。阅读文章与验证结果不需要这些文件；重新训练需要另外准备。
- 原始样例表每次只记录前 100 题，而准确率使用全部 1319 题；两者分母不同。

## 快速验证

```bash
python tools/verify_results.py
```

该命令只用 Python 标准库检查已有记录，不训练模型，不访问网络。
