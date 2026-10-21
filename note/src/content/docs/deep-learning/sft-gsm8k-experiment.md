---
title: SFT 理论与实现要点
description: 从条件语言模型、response mask 到 LoRA 和训练评估的监督微调理论与关键代码
---

这篇笔记专门整理监督微调（Supervised Fine-Tuning，SFT）的理论和实现方法。博客文章已经记录了 GSM8K 数据规模实验的设置、曲线、结果和复盘；这里不重复实验表格，重点放在可以迁移到其他任务的概念、公式和代码。

完整实验案例见：[从 128 条到 7473 条：一次 SFT 实现与五组 GSM8K 实验复盘](https://link-blog.vercel.app/posts/sft-gsm8k-data-scale-review-20261021/)。

## 一、SFT 在训练流程中的位置

语言模型训练通常可以分成几类目标：

| 阶段 | 训练数据 | 学到的东西 |
| --- | --- | --- |
| 预训练 | 大规模无标注文本 | 语言规律、知识和通用表示 |
| SFT | prompt/response 或多轮对话示范 | 按示范完成任务、遵循格式和指令 |
| 偏好优化 | chosen/rejected 回答对或奖励 | 哪种回答更符合偏好 |
| 可验证强化学习 | 可执行验证器或过程奖励 | 在尝试中寻找更可靠的解法 |

SFT 更接近“行为克隆”：给定输入和示范回答，训练模型提高示范回答的条件概率。它通常是后续 DPO、RLHF 或 Reasoning RL 的起点，但 SFT 本身不会直接优化“人更喜欢哪个回答”的奖励函数。

## 二、SFT 的概率目标与 teacher forcing

给定输入序列 $x$ 和回答序列 $y=(y_1,\ldots,y_T)$，因果语言模型把回答概率分解为：

$$
p_\theta(y\mid x)
=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t})
$$

训练时最常见的目标是最大化示范回答的对数似然，或等价地最小化 token 级负对数似然（NLL）：

$$
\mathcal{L}_{\mathrm{SFT}}
=-\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{<t})
$$

这里的 `y_<t` 使用真实示范 token，而不是模型自己在前一步生成的 token，这种训练方式叫 **teacher forcing**。它带来两个重要性质：

- 训练信号稳定，因为每个位置都能直接对照正确 token；
- 训练和自由生成存在差异，模型在生成时可能进入训练中没有见过的错误前缀，这就是 exposure bias 的来源之一。

SFT 的训练 loss 低，表示模型更愿意复现示范回答；它不自动保证答案事实正确、推理过程可靠或对未见任务泛化良好。

## 三、从对话样本到训练张量

### 3.1 数据契约和 chat template

最简单的数据格式是：

```json
{
  "prompt": "解释梯度下降的基本思想。",
  "response": "梯度下降通过沿损失函数的负梯度方向更新参数……"
}
```

多轮对话通常使用 `messages`：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个严谨的助理。"},
    {"role": "user", "content": "什么是梯度下降？"},
    {"role": "assistant", "content": "梯度下降是一种优化方法……"}
  ]
}
```

不同模型的角色标记和特殊 token 不同，应该优先使用 tokenizer 自带的 chat template，而不是手写 `<|user|>` 或 `<|assistant|>`：

```python
def render_sft_example(messages, tokenizer):
    # 只把用户上下文放进 prompt，保留生成 assistant 的起始标记
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    response = messages[-1]["content"]
    if tokenizer.eos_token:
        response += tokenizer.eos_token
    return prompt, response
```

`EOS` 很重要：它告诉模型回答何时结束。截断时如果把回答末尾的 `EOS` 一起截掉，生成阶段可能不容易停止；如果把下一个样本的内容拼进来而没有边界，模型也会学到错误的转移关系。

### 3.2 causal shift 和 response mask

假设拼接后的 token 是 `[P1, P2, R1, R2]`，因果语言模型的输入和标签需要错开一个位置：

```text
完整序列：      P1  P2  R1  R2
模型输入：      P1  P2  R1
目标 labels：   P2  R1  R2
回答 mask：      0   1   1
```

`R1` 由 prompt 的最后一个 token 预测，`R2` 由 prompt 和 `R1` 共同预测。mask 必须和 `labels` 一样在 shift 后对齐；如果仍使用未 shift 的 mask，会把错误的位置计入 loss。

下面是一个不依赖具体模型的最小整理函数：

```python
import torch


def pack_prompt_response(prompt, response, tokenizer):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    ids = prompt_ids + response_ids
    # 0：prompt，不监督；1：response，参与监督
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    return {
        "ids": ids,
        "response_mask": response_mask,
    }


def shift_for_causal_lm(ids, response_mask, attention_mask):
    ids = torch.as_tensor(ids)
    response_mask = torch.as_tensor(response_mask, dtype=torch.bool)
    attention_mask = torch.as_tensor(attention_mask, dtype=torch.bool)

    return {
        "input_ids": ids[:-1],
        "labels": ids[1:],
        "response_mask": response_mask[1:],
        # label 对应的位置如果是 padding，也必须排除
        "label_valid": attention_mask[1:],
    }
```

在 batch 中通常先右侧 padding，再分别构造 `attention_mask`、`response_mask`，最后同时取 `[:, :-1]` 和 `[:, 1:]`。`attention_mask` 负责告诉 Transformer 哪些输入位置存在；`response_mask` 负责告诉损失函数哪些目标位置属于回答。这两个 mask 的职责不同，不能混为一谈。

## 四、log probability、NLL 与三种归一化

模型输出的 logits 形状通常是 `[B, L, V]`：batch、序列长度和词表大小。只取目标 token 的 log probability：

```python
import torch.nn.functional as F


def token_nll(logits, labels):
    # 在 float 中计算归一化，避免低精度下 logsumexp 数值不稳定
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return -target_log_probs  # [B, L]
```

拿到每个 token 的 NLL 后，mask 掉 prompt 和 padding。常见的归一化方式有三种：

| 方式 | 公式 | 特点 |
| --- | --- | --- |
| 样本求和后平均 | `mean_i(sum_t nll_it)` | 长回答贡献更大 |
| 每样本 token 平均 | `mean_i(sum_t nll_it / T_i)` | 每个样本等权 |
| 全 batch token 平均 | `sum_i,t nll_it / sum_i,t 1` | 每个有效 token 等权 |

当前素材中的实现属于第一种：每条回答先求和，再按样本数平均。它适合明确把完整回答看作一个训练样本，但不能把日志 loss 直接当作 token 平均交叉熵，也不能跨项目比较后直接取指数当作 perplexity。

一个同时展示两种口径的实现如下：

```python
def masked_sft_loss(logits, labels, response_mask, label_valid=None):
    nll = token_nll(logits, labels)
    mask = response_mask.bool()
    if label_valid is not None:
        mask = mask & label_valid.bool()

    if not mask.any(dim=1).all():
        raise ValueError("每个样本都必须至少有一个有效 response token")

    masked_nll = nll.masked_fill(~mask, 0.0)
    tokens_per_sample = mask.sum(dim=1)

    # 口径 A：回答 token 求和，再按样本平均
    sample_sum_loss = masked_nll.sum(dim=1).mean()

    # 口径 B：整个 batch 按有效 token 平均
    token_mean_loss = masked_nll.sum() / mask.sum().clamp_min(1)

    # 口径 C：每个样本先做 token 平均，再对样本平均
    per_sample_mean = masked_nll.sum(dim=1) / tokens_per_sample
    sample_mean_loss = per_sample_mean.mean()
    return sample_sum_loss, token_mean_loss, sample_mean_loss
```

如果采用 Hugging Face 的 `CrossEntropyLoss(ignore_index=-100)`，可以把 prompt 和 padding 对应的 label 设为 `-100`。但 `-100` 不能直接传给 `gather` 作为词表索引；使用手写 `gather` 时应保留合法的 label，再单独用 mask 排除无监督位置。

## 五、数据质量、截断与 packing

SFT 的上限通常由示范数据决定，而不只是由模型大小决定。整理数据时需要检查：

- **重复和近重复**：完全重复会放大某些回答的梯度；近重复会让训练集看起来很大，实际覆盖仍然狭窄。
- **错误示范**：SFT 会认真模仿错误答案、错误格式和不安全行为，除非在数据层过滤或提供更好的监督。
- **冲突示范**：同一个问题出现互相矛盾的回答时，模型学到的是混合分布，训练 loss 可能难以下降。
- **模板偏差**：固定的开头、结尾和格式可能被模型记住，却没有学会任务本身。
- **数据泄漏**：训练样本与验证/测试样本的题面、答案或推理过程重合，会让生成指标虚高。
- **长度和截断**：截断要记录发生比例，并确认没有系统性丢掉问题末尾、回答末尾或 EOS。

多个短样本可以 packing 到同一条序列以提高 GPU 利用率，但必须提供边界 mask，避免一个样本的回答预测到下一个样本的 token。最简单、最不容易出错的实现是先使用 padding，确认 mask 和评估正确后再做 packing 优化。

## 六、让 SFT 稳定收敛

### 6.1 有效 batch 与梯度累积

显存放不下目标 batch 时，可以把一个更新拆成多个 micro-batch。核心是每次反向传播前除以累积步数：

```python
for update in range(num_updates):
    optimizer.zero_grad(set_to_none=True)

    for _ in range(accum_steps):
        batch = next(train_iter)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss, _, _ = masked_sft_loss(
            outputs.logits,
            batch["labels"],
            batch["response_mask"],
            batch.get("label_valid"),
        )
        (loss / accum_steps).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

有效 batch 大致等于 `micro_batch_size × accum_steps`，但变长序列时显存占用和有效 token 数仍会变化。`batch_size` 这个参数名不会自动改变累积逻辑；应在代码里显式计算并断言实际的累积步数。

### 6.2 优化器、学习率与数值精度

- AdamW 的 weight decay 应与学习率、可训练参数范围一起设定；不要把 LayerNorm 和 bias 无条件地当作普通权重衰减。
- warmup 可以避免刚开始训练时更新过猛，后续再使用 cosine 或线性衰减。
- 梯度裁剪能限制少数异常 batch 造成的巨大更新，但不能代替清洗数据。
- bf16 通常比 fp16 有更大的指数范围；低精度下计算 log-softmax、loss 汇总和梯度范数时要留意溢出与精度。
- gradient checkpointing 用计算换显存，适合长序列训练，但会增加训练时间。
- 固定 Python、NumPy、PyTorch 和数据采样器的随机种子，并保存配置与抽样索引，才能复现训练曲线。

### 6.3 过拟合和早停

SFT 过拟合不一定首先表现为训练 loss 爆炸，常见信号是训练 loss 继续下降而独立验证集的生成质量下降。更可靠的流程是：用验证集选择最佳 checkpoint，再只对测试集报告最终结果；不要用测试集曲线反复决定训练步数。

## 七、全参数微调、LoRA 与 QLoRA

全参数微调更新基座模型的全部权重，表达能力强，但显存、优化器状态和 checkpoint 都很大。参数高效微调（PEFT）冻结基座，只训练少量附加参数。

LoRA 将权重更新约束为低秩矩阵：

$$
W' = W + \Delta W,
\qquad
\Delta W = \frac{\alpha}{r}BA
$$

其中 $W$ 冻结，$A$ 和 $B$ 是可训练矩阵，$r$ 是 rank，$\alpha$ 控制缩放。rank 越高，适配能力和参数量通常越大；rank 太低可能无法表达任务所需的变化。

使用 PEFT 的最小配置示例：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

`target_modules` 必须按具体模型的层名称检查，不能盲目照抄。训练完成后通常只保存 adapter；部署时可以加载基座和 adapter，或在确认兼容后 merge。QLoRA 进一步把基座以 4-bit 等低精度加载，再训练 LoRA 参数；量化降低显存，不等于把 LoRA 更新也量化成不可训练的权重。

## 八、怎样评估 SFT

训练 NLL 是 teacher-forcing 条件下的指标，生成质量还需要独立评估：

1. **验证集 NLL**：适合观察拟合和选择 checkpoint，但不等于任务准确率。
2. **生成式指标**：固定 tokenizer、chat template、停止条件、`max_new_tokens` 和采样参数后比较输出。
3. **任务指标**：数学题可以 exact match 或 verifier score，代码题可以运行测试，结构化任务可以检查 schema。
4. **错误分解**：把格式错误、答案错误、截断和空输出分开统计，避免把所有失败归为同一种问题。
5. **数据隔离**：验证集和测试集不能参与训练调参，也要检查近重复和 benchmark contamination。

生成评估的骨架可以写成：

```python
@torch.no_grad()
def generate_and_score(model, tokenizer, prompts, score_fn):
    model.eval()
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return [score_fn(text) for text in texts]
```

如果训练使用 PyTorch、生成使用另一个推理引擎，还要确认评估前已经同步最新权重，并清理推理引擎缓存。否则指标可能对应旧 checkpoint。

## 九、常见实现陷阱

- `input_ids`、`labels` 和 `response_mask` 没有同步 shift，导致监督位置错一位。
- prompt token 或 padding token 被意外计入 loss，模型学会复述提示或 padding。
- 把 `-100` 直接传给 `gather`，产生非法词表索引；`ignore_index` 和显式 mask 是两条不同路径。
- 空回答或全部被截断时，mask 的有效 token 数为 0，归一化会产生 NaN。
- 把字符长度当成 token 长度，或把生成上限误认为训练截断长度。
- 梯度累积时忘记除以 `accum_steps`，或每个 micro-batch 使用不同的 token 归一化导致样本权重改变。
- 训练时忘记 `model.train()`，评估时忘记 `model.eval()` 或 `torch.no_grad()`。
- 只保存最终权重，没有保存最佳验证 checkpoint、tokenizer、配置和随机种子。
- LoRA 的 target module 名称与模型结构不匹配，实际没有训练到预期层。
- 把格式正确当成答案正确，把训练 loss 下降当成泛化能力提升。

排查时可以按这个顺序做一个极小实验：先用一条样本过拟合，确认生成完全正确；再检查 mask 可视化；然后验证 padding、截断和 EOS；最后才扩大 batch、引入 packing 或 PEFT。

## 十、一个可复用的检查清单

- 数据是否明确区分 prompt、response、角色和 EOS？
- chat template 是否与推理时保持一致？
- `labels` 是否右移，response mask 是否同步右移？
- padding、截断和空回答是否被排除？
- loss 的分母是样本数、每样本 token 数，还是全 batch token 数？
- 梯度累积、梯度裁剪、学习率调度和混合精度是否显式实现？
- 是否有独立验证集、最佳 checkpoint 和固定随机种子？
- 生成评估是否固定解码条件，并区分格式、答案和截断错误？
- 使用 LoRA 时是否确认 target modules 和可训练参数数量？

实验结果、数据规模比较和具体曲线继续放在[博客文章](https://link-blog.vercel.app/posts/sft-gsm8k-data-scale-review-20261021/)中；这篇笔记只保留能够迁移到其他 SFT 任务的理论与实现方法。
