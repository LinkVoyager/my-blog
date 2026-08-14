---
title: RoPE 旋转位置编码详解
description: RotaryPositionalEmbedding 模块 __init__ 部分的逐行讲解与手算全过程示例
---

本文档整理了 `RotaryPositionalEmbedding` 模块 `__init__` 部分的逐行讲解，以及一个用具体小数字手算一遍全过程的例子，方便对照理解。

## 目标代码

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, context_length: int, device=None):
        """
        初始化RoPE模块
        theta:基准频率，通常为10000
        d_k: 每个Head的维数，必须为偶数
        context_length: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k

        #1.计算频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers) #形状：(d_k / 2)

        #2. 创建位置序列[0,1,...., context_length - 1]
        t = torch.arange(context_length, device=device).float() # 形状: (context_length,)

        #3.计算所有位置的所有角度(外积)
        # freqs_matrix 形状: (context_length, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)
```

---

## 第一部分：逐段拆解

### 函数签名和参数

```python
def __init__(self, theta: float, d_k: int, context_length: int, device=None):
```

- **`theta`**：旋转角的"基准频率"（base），公式里控制不同维度旋转快慢的那个常数。
  ⚠️ 文档字符串写的"通常为1000"应该是笔误——RoPE 论文和几乎所有主流实现（LLaMA、GPT-NeoX 等）里的标准值都是 **10000**，不是 1000。这个值影响的是低频维度能表达多长的位置范围，取错量级会明显影响模型对长距离位置的建模能力，建议检查一下这里是不是手误。
- **`d_k`**：单个注意力头（head）的维度，必须是偶数——因为 RoPE 是把这 $d_k$ 维两两分组做二维旋转，奇数没法两两配对。
- **`context_length`**：要支持的最大序列长度，决定了预计算多少个位置的 cos/sin。
- **`device`**：让张量直接建在目标设备（CPU/GPU）上，避免后面再 `.to(device)` 搬一次。

### 第 1 步：计算频率 ω

```python
powers = torch.arange(0, d_k, 2, device=device).float() / d_k
freqs = 1.0 / (theta ** powers)
```

`torch.arange(0, d_k, 2)` 生成 `[0, 2, 4, ..., d_k-2]`，一共 $d_k/2$ 个数——对应公式里的下标 $2i$（$i=0,\dots,d_k/2-1$）。除以 `d_k` 后就是指数 $\frac{2i}{d_k}$，所以：

$$
\omega_i = \theta^{-2i/d_k} = \frac{1}{\theta^{2i/d_k}}
$$

`freqs` 的形状是 `(d_k/2,)`。

**为什么只算 $d_k/2$ 个而不是 $d_k$ 个**：因为旋转是把向量两两分组做二维旋转的（第 0、1 维一组，第 2、3 维一组……），每一组共用一个角频率 $\omega_i$，所以频率的个数正好是维度数的一半。$i$ 越大（越靠后的维度），$\omega_i$ 越小（因为分母指数变大），也就是转得越慢——这跟原始 Transformer 的 sin/cos 位置编码"低维高频、高维低频"的设计思路是一脉相承的。

### 第 2 步：位置序列

```python
t = torch.arange(context_length, device=device).float()
```

就是 $[0, 1, 2, \dots, \text{context\_length}-1]$，对应每个可能的绝对位置 $m$。形状 `(context_length,)`。

### 第 3 步：外积——算出所有位置的所有角度

```python
freqs_matrix = torch.outer(t, freqs)
```

`torch.outer(a, b)` 是外积，结果形状 `(len(a), len(b))`，第 `(m, i)` 个元素是 `a[m] * b[i]`。这里就是：

$$
\text{freqs\_matrix}[m, i] = m \times \omega_i = m\theta_i
$$

也就是"第 $m$ 个位置、第 $i$ 组维度"应该旋转的角度。形状 `(context_length, d_k/2)`——每一行对应一个位置，每一列对应一组维度的角频率。这一步本质上是用一次矩阵运算，把 `context_length × d_k/2` 种"位置-频率"组合的角度全部一次性算出来，避免写循环。

### 第 4 步：预计算 cos/sin 并注册为 buffer

```python
self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)
```

- 先把角度矩阵整体取 `cos()`、`sin()`，得到两个形状同为 `(context_length, d_k/2)` 的表，之后 `forward()` 里对 $q$、$k$ 做旋转时直接按位置索引查表用，不用每次前向传播都重新算三角函数，是常见的"预计算+缓存"优化。
- 用 `register_buffer` 而不是普通属性（`self.cos_cached = ...`）的好处：buffer 会跟着模型一起被 `.to(device)`、`.cuda()`、`.half()` 这类操作自动搬运/转换 dtype，也会出现在 `state_dict()` 能感知的位置——本质上告诉 PyTorch"这是模型的一部分状态，但不是需要梯度更新的可学习参数"（RoPE 的 cos/sin 是固定值，不参与训练）。
- **`persistent=False`**：让这两个 buffer **不**被保存进 `state_dict()`（也就是存 checkpoint 的时候不会把这张表也存进去）。这样做合理，因为 `cos_cached`/`sin_cached` 完全是由 `theta`、`d_k`、`context_length` 这几个超参数决定的，加载模型时用同样的超参数重新 `__init__` 一次就能精确复现，没必要占存储空间存下来。

### 小结

这个 `__init__` 做的事情，其实就是把"对每个位置 $m$、每组维度 $i$，需要旋转 $m\theta_i$ 度"这件事，提前算好、存成两张查表用的 `(context_length, d_k/2)` 大小的 cos/sin 表；真正对 $q$、$k$ 做旋转的逻辑在 `forward()` 里，按传入的位置索引从这两张表里取出对应的 cos/sin，再对向量做"两两一组"的旋转变换。

---

## 第二部分：具体数值例子

用一组具体的小数字，把上面这段代码从头到尾"手算"一遍：`d_k = 4`（方便手算），`theta = 10000`，`context_length = 4`（序列最多 4 个 token）。

### 第 0 步：为什么要两两分组

$d_k=4$ 维的向量，比如 $(x_0, x_1, x_2, x_3)$，RoPE 会把它拆成 **2 组**，每组 2 个数：

- 第 0 组：$(x_0, x_1)$
- 第 1 组：$(x_2, x_3)$

每一组被当成平面上的一个点 $(x, y)$，旋转的时候整组一起转。$d_k=4$ 能拆成 2 组，所以后面所有跟"频率""角度"有关的东西，个数都是 $d_k/2 = 2$ 个，而不是 4 个——这是看懂后面代码形状的关键。

### 第 1 步：算频率

`torch.arange(0, 4, 2)` = `[0, 2]`（从 0 开始，每次跳 2，跳到 4 之前停）。这两个数正好对应"第 0 组""第 1 组"。

除以 `d_k=4`：

```
powers = [0/4, 2/4] = [0.0, 0.5]
```

代入 `freqs = 1 / (10000 ** powers)`：

```
freqs[0] = 1 / 10000^0.0 = 1 / 1     = 1.0
freqs[1] = 1 / 10000^0.5 = 1 / 100   = 0.01
```

所以 `freqs = [1.0, 0.01]`。这两个数字的含义是：**第 0 组维度转得快（每走一个位置转 1.0 弧度），第 1 组维度转得慢（每走一个位置只转 0.01 弧度）**——"低维转得快、高维转得慢"的具体体现。

### 第 2 步：位置序列

序列长度是 4，所以 `t = [0, 1, 2, 3]`，分别代表第 0、1、2、3 个 token 的位置。

### 第 3 步：外积——算出"每个位置、每一组"该转多少角度

外积就是把 `t` 里每个数，分别乘 `freqs` 里每个数，摆成一张表（4 行 2 列）：

| 位置 m | 第0组角度 = m×1.0 | 第1组角度 = m×0.01 |
|---|---|---|
| m=0 | 0.0 | 0.0 |
| m=1 | 1.0 | 0.01 |
| m=2 | 2.0 | 0.02 |
| m=3 | 3.0 | 0.03 |

可以看到：**位置越靠后（m 越大），角度越大**；同一个位置里，第 0 组（转得快）积累的角度明显比第 1 组（转得慢）大很多。这张表里第 `(m, i)` 个数，就是"第 $m$ 个 token、第 $i$ 组维度，应该旋转多少弧度"，即 $m\theta_i$。

### 第 4 步：把角度转成 cos / sin，存起来

拿上面那张"角度表"，每个数分别取 `cos` 和 `sin`：

**`cos_cached`**（对上面每个角度取余弦）：

| 位置 m | 第0组 | 第1组 |
|---|---|---|
| 0 | 1.0000 | 1.0000 |
| 1 | 0.5403 | 1.0000 |
| 2 | -0.4161 | 0.9998 |
| 3 | -0.9900 | 0.9996 |

**`sin_cached`**（同样的角度取正弦）：

| 位置 m | 第0组 | 第1组 |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.8415 | 0.0100 |
| 2 | 0.9093 | 0.0200 |
| 3 | 0.1411 | 0.0300 |

**为什么要提前算好存起来**：旋转矩阵长这样：

$$
R(\theta) = \begin{pmatrix}\cos & -\sin\\ \sin & \cos\end{pmatrix}
$$

要对某个位置的向量做旋转，本质上就是要用到这个位置对应的 `cos` 和 `sin` 这两个数字。与其等到真正跑 `forward()` 的时候，每次都现算 `cos(m*freq)`、`sin(m*freq)`（三角函数计算比较慢），不如在模型初始化的时候，把 0 到 `context_length-1` 每一个可能出现的位置、每一组维度，需要用到的 cos、sin 值，一次性全部算好存成两张表。之后不管处理哪一句话，`forward()` 里只要按这句话里每个 token 的位置去查表取数就行，不用重复计算——这是纯粹的"空间换时间"优化。

`register_buffer` 就是 PyTorch 里"把一个张量登记成模型的一部分、但不是需要训练更新的参数"的写法（因为这两张表是根据 `theta`、`d_k`、`context_length` 算出来的固定数字，不需要梯度下降去学）；`persistent=False` 是说"存模型的时候不用把这两张表也存进去"，因为下次加载模型、用同样的 `theta`/`d_k`/`context_length` 重新跑一遍这段 `__init__`，就能一模一样地重新生成出来，没必要占地方存。

---

## 一句话总结

提前把"第几个位置 × 第几组维度 → 应该转多少度、对应的 cos/sin 是多少"这张表整个算出来存好，之后 `forward()` 里对 `q`、`k` 做旋转的时候，直接按 token 的位置去查这张表拿 cos/sin 用，不用现算。
