# 大模型微调参数学习笔记

> 面向 LLaMA-Factory / Hugging Face / LoRA / GaLore 等常见训练场景。  
> 目标：用通俗语言、统一例子和公式，把常见微调参数讲清楚。

## 目录

1. RoPE 与长上下文扩展
2. 训练加速方式
3. 梯度裁剪与随机种子
4. 截断长度、Batch Size 与梯度累积
5. 学习率调节器与 Warmup
6. 其他常用训练参数
7. 部分参数微调 Freeze
8. LoRA
9. RLHF
10. 多模态参数
11. GaLore
12. GaLore + Adam：3×3 完整例子

---

# 1. RoPE 与长上下文扩展

## 1.1 RoPE 的频率公式

RoPE 中每一组旋转频率：

$$
\omega_i=\frac{1}{\theta^{2i/d}}=\theta^{-2i/d}
$$

其中：

- $\theta$：RoPE base，常见如 `10000`
- $d$：参与 RoPE 的维度
- $i$：第几组二维旋转
- $\omega_i$：该组旋转频率

位置 $p$ 的旋转角：

$$
\phi_i(p)=p\omega_i
$$

### 统一例子

假设：

- 原生上下文：`8000`
- 目标上下文：`32000`
- 扩展倍数：

$$
factor=\frac{32000}{8000}=4
$$

再设：

$$
\theta=10000,\qquad d=8
$$

则四组频率：

$$
\boxed{\omega=[1,\ 0.1,\ 0.01,\ 0.001]}
$$

可以粗略理解为：

- 高频：更擅长区分相邻 token
- 低频：更适合描述较远位置

## 1.2 none

不修改 RoPE：

$$
\omega_i'=\omega_i
$$

特点：

- 短距离能力完全保持
- 超出原生长度时容易超出训练分布

## 1.3 linear

统一缩放：

$$
\boxed{\omega_i'=\frac{\omega_i}{factor}}
$$

`8000 -> 32000` 时 `factor=4`：

$$
[1,0.1,0.01,0.001]
\rightarrow
[0.25,0.025,0.0025,0.00025]
$$

也可理解为：

$$
p'=p/4
$$

因此：

$$
32000\rightarrow8000
$$

优点：简单、能把长位置压回熟悉范围。  
缺点：连近距离也一起压缩。

## 1.4 Dynamic NTK

Dynamic 不直接做 $\omega/4$，而是修改 $\theta$：

$$
\theta'
=
\theta
\left[
f\frac{L}{L_0}-(f-1)
\right]^{d/(d-2)}
$$

统一例子：

$$
f=4,\quad L=32000,\quad L_0=8000,\quad d=8
$$

得到：

$$
\theta'
=
10000
[4\times4-3]^{8/6}
=
10000\times13^{8/6}
$$

约：

$$
\boxed{\theta'\approx305674}
$$

再重新计算各维频率。

特点：

- 高频变化少
- 中频变化更多
- 低频变化更大

## 1.5 YaRN

可粗略理解为：

$$
\omega'
=
\alpha\omega
+
(1-\alpha)\frac{\omega}{factor}
$$

其中：

- 高频：$\alpha\approx1$
- 中频：$0<\alpha<1$
- 低频：$\alpha\approx0$

核心思想：

> 高频尽量保持，中频平滑过渡，低频更多采用扩展后的频率。

## 1.6 Llama3 RoPE

先根据频率计算波长：

$$
\lambda=\frac{2\pi}{\omega}
$$

然后分段：

- 短波长 / 高频：基本不改
- 中间区域：平滑过渡
- 长波长 / 低频：按 factor 缩放

## 1.7 怎么比较

不能看“数字越大越好”。

真正要同时考虑：

1. **近距离能力**：例如 token 1000 和 1001
2. **远距离能力**：例如 token 1000 和 31000

| 方法 | 近距离 | 长距离 |
|---|---|---|
| none | 保持最好 | 容易超纲 |
| linear | 也被压缩 | 压得直接 |
| dynamic | 高频少改 | 低频强调整 |
| YaRN | 高频保护好 | 低频扩展 |
| Llama3 | 高频保护好 | 按波长分段扩展 |

---

# 2. 训练加速方式

这些不是 RoPE，而是为了让训练/推理更快、更省显存。

## auto

让框架自己选择可用实现。适合先跑通环境。

## FlashAttention2

仍计算：

$$
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt d}\right)V
$$

但通过分块、减少显存读写等方式优化 GPU 执行。

特点：

> 序列越长，通常收益越明显。

## Unsloth

偏训练流程级优化，常涉及：

- Attention
- LoRA
- RoPE
- RMSNorm
- 反向传播
- 显存管理

常用于 LoRA / QLoRA。

## Liger Kernel

为 Transformer 多个算子提供高性能 GPU kernel，例如：

- RMSNorm
- RoPE
- SwiGLU
- CrossEntropy
- fused kernel

| 方式 | 主要用途 |
|---|---|
| auto | 自动选择 |
| flashattn2 | 重点优化 Attention |
| unsloth | 整体优化 LoRA/QLoRA |
| liger_kernel | 优化多个 Transformer GPU 算子 |

---

# 3. 梯度裁剪与随机种子

## 3.1 梯度裁剪

例如：

```text
max_grad_norm = 1.0
```

假设：

$$
g=[3,4]
$$

梯度范数：

$$
\|g\|=\sqrt{3^2+4^2}=5
$$

阈值是 1，则缩放比例：

$$
\frac{1}{5}=0.2
$$

所以：

$$
[3,4]\rightarrow[0.6,0.8]
$$

通用公式：

$$
\boxed{
g'
=
g\times
\min\left(1,\frac{max\_norm}{\|g\|}\right)
}
$$

重点：

> 保留梯度方向，只限制整体长度。

作用：

- 防止梯度爆炸
- 降低训练发散风险
- 降低 NaN 风险

## 3.2 随机种子

例如：

```text
seed = 42
```

影响：

- 数据 shuffle
- dropout
- 参数初始化
- 随机采样

作用：

> 尽量让实验可复现。

`42` 本身并不会比别的数字训练得更好。

---

# 4. 截断长度、Batch Size 与梯度累积

假设：

```text
cutoff_len = 2048
batch_size = 4
gradient_accumulation_steps = 8
```

## 截断长度

`cutoff_len=2048`：

> 单条训练样本最多允许 2048 token。

注意：梯度累积不会把最大上下文长度从 2048 变长。

## Batch Size

`batch_size=4`：

> 每张 GPU 一次 forward/backward 同时处理 4 条样本。

## 梯度累积

`gradient_accumulation_steps=8`：

> 连续做 8 次 forward/backward，再执行一次 `optimizer.step()`。

单 GPU：

$$
\boxed{4\times8=32}
$$

双 GPU：

$$
\boxed{4\times8\times2=64}
$$

公式：

$$
\boxed{
effective\_batch
=
per\_device\_batch
\times
accumulation
\times
GPU\_count
}
$$

过程：

```text
第1次：4条样本 → 累积梯度
第2次：4条样本 → 累积梯度
...
第8次：4条样本 → 累积完成
                 ↓
            optimizer.step()
```

---

# 5. 学习率调节器与 Warmup

Scheduler 决定：

> 训练过程中学习率怎么变化。

## linear

Warmup 后直线下降。

## cosine

常见形式：

$$
LR(t)
=
\frac{LR_{max}}{2}
\left(1+\cos(\pi t)\right)
$$

特点：

- 下降较平滑
- 大模型训练常见

## cosine_with_restarts

余弦下降后重新拉高，再进行下一轮下降。

## polynomial

可理解为：

$$
LR=LR_{max}(1-t)^p
$$

## constant

学习率全程不变。

## constant_with_warmup

先 warmup，再保持恒定。

## inverse_sqrt

大致：

$$
LR\propto\frac{1}{\sqrt{step}}
$$

## reduce_lr_on_plateau

验证指标长期不改善时再降低学习率。

## cosine_with_min_lr

余弦下降，但设置最低学习率。

## warmup_stable_decay

三段式：

1. Warmup
2. Stable
3. Decay

## Warmup

例如：

```text
learning_rate = 2e-5
warmup_steps = 100
```

前 100 步大致：

```text
step 0   -> lr ≈ 0
step 25  -> lr ≈ 5e-6
step 50  -> lr ≈ 1e-5
step 75  -> lr ≈ 1.5e-5
step 100 -> lr = 2e-5
```

作用：

> 训练刚开始时不要直接使用完整学习率，先小步起步，减少初始阶段不稳定。

也可以：

```text
warmup_ratio = 0.1
```

例如总训练 1000 step：

$$
1000\times0.1=100
$$

即前 100 步 warmup。

---

# 6. 其他常用训练参数

## logging_steps

`logging_steps=5`：每 5 个 global step 输出一次日志。

## save_steps

`save_steps=100`：每 100 step 保存一次 checkpoint。

## NEFTune

在 embedding 上加入小噪声，作为正则化方式。

`neftune_noise_alpha=0` 表示关闭。

## packing

把多个短样本打包进一个较长训练序列，减少 padding 浪费。

## neat packing

多个样本虽然打包到一起，但尽量避免不同样本之间互相 Attention。

## train_on_prompt

决定 prompt 部分是否也计算 loss。

普通 SFT 常见做法：

> 只训练 assistant 输出。

## mask_history

多轮对话时，只训练最后一轮回答，历史作为上下文但不计算历史轮 loss。

## resize_vocab

新增 tokenizer token 时，可能需要调整 embedding / lm_head 大小。

## 外部记录面板

| 工具 | 用途 |
|---|---|
| none | 不启用外部面板 |
| tensorboard | 本地看训练曲线 |
| wandb | 云端实验对比 |
| mlflow | 偏工程化实验/模型管理 |
| neptune | 实验跟踪 |
| trackio | 日志/可视化 |
| all | 全部上报 |

建议重点看：

- train_loss
- eval_loss
- learning_rate
- grad_norm

---

# 7. 部分参数微调 Freeze

假设模型有 32 层。

## 可训练层数

```text
2
```

通常表示训练最后 2 层：

```text
Layer 0 ~ 29  冻结
Layer 30      训练
Layer 31      训练
```

如果是：

```text
-2
```

则表示最前面的 2 层。

## 可训练模块

`all` 的意思不是整个模型都训练，而是：

> 在选中的可训练层内部，所有模块都参与训练。

如果：

```text
q_proj,v_proj
```

则只训练这些模块。

## 额外模块

例如：

```text
lm_head
```

表示除了选中的 Transformer 层，再额外训练 `lm_head`。

## Freeze 与 LoRA

Freeze：

$$
W\rightarrow W'
$$

直接更新原模型的一部分参数。

LoRA：

$$
W'=W+\Delta W
$$

原始 $W$ 通常冻结，只训练低秩参数。

---

# 8. LoRA

LoRA：

$$
\Delta W=BA
$$

最终：

$$
W'=W+sBA
$$

其中通常：

$$
s=\frac{\alpha}{r}
$$

## rank

例如：

```text
r = 8
```

若：

$$
W\in\mathbb R^{4096\times4096}
$$

则：

$$
A\in\mathbb R^{8\times4096}
$$

$$
B\in\mathbb R^{4096\times8}
$$

参数量：

$$
8\times4096+4096\times8=65536
$$

而原矩阵：

$$
4096^2=16777216
$$

## alpha

例如：

```text
r = 8
alpha = 16
```

则：

$$
\frac{\alpha}{r}=2
$$

所以：

$$
\boxed{W'=W+2BA}
$$

可以把 alpha 理解成 LoRA 分支的“音量”。

## dropout

例如：

```text
lora_dropout = 0.04
```

LoRA 分支训练时进行少量 dropout，用于正则化。

## LoRA+

让 A、B 使用不同学习率。

若：

$$
\lambda=\frac{\eta_B}{\eta_A}
$$

则：

$$
\eta_B=\lambda\eta_A
$$

## rsLoRA

普通 LoRA：

$$
\frac{\alpha}{r}
$$

rsLoRA：

$$
\frac{\alpha}{\sqrt r}
$$

大 rank 时通常更稳定。

## DoRA

将权重的方向与大小分开处理。

## PiSSA

利用原权重的 SVD 信息初始化 LoRA，让初始低秩方向更有信息。

## LoRA target

决定哪些模块插 LoRA，例如：

```text
q_proj,v_proj
```

或：

```text
all
```

## 附加模块

例如：

```text
embed_tokens,lm_head
```

表示这些原生模块也一起训练。

---

# 9. RLHF

推荐阅读：

1. **Deep Reinforcement Learning from Human Preferences**（2017）
2. **Proximal Policy Optimization Algorithms**（PPO，2017）
3. **Learning to Summarize from Human Feedback**（2020）
4. **Training language models to follow instructions with human feedback**（InstructGPT，2022）
5. **Direct Preference Optimization**（DPO，2023）

经典 RLHF：

$$
\boxed{
SFT
\rightarrow
Reward\ Model
\rightarrow
PPO
}
$$

DPO：

> 不单独训练 Reward Model、不跑 PPO，直接使用 chosen / rejected 偏好数据优化模型。

常见：

```text
beta = 0.1
```

用于控制偏好优化强度与参考模型约束尺度。

---

# 10. 多模态参数

典型结构：

```text
图片
↓
视觉编码器
↓
多模态投影器
↓
语言模型
↓
答案
```

## 冻结视觉编码器

Vision Encoder 继续参与前向，但参数不更新。

## 冻结多模态投影器

投影器负责把视觉特征映射到语言模型 hidden space。

例如：

$$
1024\rightarrow4096
$$

冻结后 projector 仍工作，但权重不更新。

## 冻结语言模型

勾选后 LLM 主干参数不更新。

## 图像最大/最小像素

例如：

```text
最大：768*768
最小：32*32
```

分辨率越高：

- 细节越多
- 视觉 token 越多
- 显存与计算量越大

## 视频像素

视频还有时间维度：

$$
T\times H\times W
$$

所以即使单帧分辨率较低，总计算量仍可能很大。

---

# 11. GaLore

GaLore 的核心：

> 仍然更新原模型参数，但把梯度/优化器状态投影到低秩空间中处理，以节省显存。

## 与 LoRA 的区别

LoRA：

```text
原始 W 冻结
↓
训练 A、B
↓
W' = W + BA
```

GaLore：

```text
原始 W 仍然训练
↓
完整梯度 G
↓
低秩投影
↓
优化器在低维空间工作
↓
更新映射回 W
```

## GaLore rank

假设：

$$
G\in\mathbb R^{4096\times4096}
$$

若：

```text
rank = 16
```

则低秩梯度可类似表示成：

$$
G_{low}\in\mathbb R^{16\times4096}
$$

## 更新间隔

例如：

```text
update_interval = 200
```

表示：

> 每 200 step 重新计算一次低秩投影方向。

## scale

控制低秩梯度的缩放强度。

## target

例如：

```text
all
```

表示所有符合条件的线性模块应用 GaLore。

---

# 12. GaLore + Adam：3×3 完整例子

这是理解 GaLore 最关键的一部分。

设完整梯度：

$$
G=
\begin{bmatrix}
3&0&0\\
4&0&0\\
0&0&0
\end{bmatrix}
$$

GaLore：

$$
rank=1
$$

## 12.1 找主要方向

第一列：

$$
\begin{bmatrix}
3\\
4\\
0
\end{bmatrix}
$$

长度：

$$
\sqrt{3^2+4^2}=5
$$

单位方向：

$$
\boxed{
P=
\begin{bmatrix}
0.6\\
0.8\\
0
\end{bmatrix}
}
$$

## 12.2 投影到低秩空间

$$
\boxed{
G_{low}=P^TG
}
$$

其中：

$$
P^T=
\begin{bmatrix}
0.6&0.8&0
\end{bmatrix}
$$

所以：

$$
G_{low}
=
\begin{bmatrix}
0.6&0.8&0
\end{bmatrix}
\begin{bmatrix}
3&0&0\\
4&0&0\\
0&0&0
\end{bmatrix}
$$

得到：

$$
\boxed{
G_{low}
=
\begin{bmatrix}
5&0&0
\end{bmatrix}
}
$$

即：

$$
3\times3\rightarrow1\times3
$$

---

## 12.3 Adam 的 m 和 v

设：

$$
g_1=
\begin{bmatrix}
5&0&0
\end{bmatrix}
$$

Adam 常用：

$$
\beta_1=0.9
$$

$$
\beta_2=0.999
$$

初始：

$$
m_0=0,\qquad v_0=0
$$

### 一阶矩

$$
m_1
=
\beta_1m_0+(1-\beta_1)g_1
$$

所以：

$$
m_1
=
0.1
\begin{bmatrix}
5&0&0
\end{bmatrix}
$$

得到：

$$
\boxed{
m_1=
\begin{bmatrix}
0.5&0&0
\end{bmatrix}
}
$$

### 二阶矩

$$
v_1
=
\beta_2v_0+(1-\beta_2)g_1^2
$$

因为：

$$
g_1^2=
\begin{bmatrix}
25&0&0
\end{bmatrix}
$$

所以：

$$
v_1
=
0.001
\begin{bmatrix}
25&0&0
\end{bmatrix}
$$

得到：

$$
\boxed{
v_1=
\begin{bmatrix}
0.025&0&0
\end{bmatrix}
}
$$

---

## 12.4 偏差修正

$$
\hat m_1
=
\frac{m_1}{1-\beta_1}
$$

所以：

$$
\hat m_1
=
\frac{0.5}{0.1}
=
5
$$

即：

$$
\boxed{
\hat m_1=[5,0,0]
}
$$

同理：

$$
\hat v_1
=
\frac{0.025}{0.001}
=
25
$$

即：

$$
\boxed{
\hat v_1=[25,0,0]
}
$$

---

## 12.5 Adam 更新量 Δ

Adam 的参数更新量：

$$
\boxed{
\Delta
=
-\eta
\frac{\hat m}
{\sqrt{\hat v}+\epsilon}
}
$$

注意：

> $\Delta$ 不是“更新后的梯度”，而是最终要加到参数上的更新量。

假设：

$$
\eta=0.1
$$

则：

$$
\Delta_{low}
=
-0.1\times\frac{5}{\sqrt{25}}
$$

忽略极小的 $\epsilon$：

$$
\boxed{
\Delta_{low}
=
[-0.1,0,0]
}
$$

---

## 12.6 映射回原空间

$$
\boxed{
\Delta W
=
P\Delta_{low}
}
$$

代入：

$$
\Delta W
=
\begin{bmatrix}
0.6\\
0.8\\
0
\end{bmatrix}
\begin{bmatrix}
-0.1&0&0
\end{bmatrix}
$$

得到：

$$
\boxed{
\Delta W=
\begin{bmatrix}
-0.06&0&0\\
-0.08&0&0\\
0&0&0
\end{bmatrix}
}
$$

最终：

$$
\boxed{
W_{new}=W_{old}+\Delta W
}
$$

---

## 12.7 完整链路

$$
\boxed{
G
\rightarrow
G_{low}
\rightarrow
m,v
\rightarrow
\hat m,\hat v
\rightarrow
\Delta_{low}
\rightarrow
\Delta W
\rightarrow
W_{new}
}
$$

对应：

```text
完整梯度 G
    ↓
GaLore 投影
    ↓
低秩梯度 G_low
    ↓
Adam 计算 m、v
    ↓
得到 Δ_low
    ↓
映射回原空间
    ↓
得到 ΔW
    ↓
W_new = W_old + ΔW
```

---

## 12.8 GaLore 为什么对 Adam 特别省显存

普通 Adam：

$$
G:3\times3
$$

因此：

$$
m:3\times3
$$

$$
v:3\times3
$$

状态值：

$$
9+9=18
$$

GaLore rank=1：

$$
G_{low}:1\times3
$$

因此：

$$
m:1\times3
$$

$$
v:1\times3
$$

状态值：

$$
3+3=6
$$

所以玩具例子：

$$
18\rightarrow6
$$

真实模型里差距更明显。

例如：

$$
4096\times4096
$$

普通 Adam 的 $m,v$ 都非常大。

GaLore rank=16 后，优化器状态可以放在类似：

$$
16\times4096
$$

的低秩空间。

核心一句话：

> **GaLore 重点节省的是 Adam/AdamW 的优化器状态，并不是把原始模型权重本身删掉。**

---

# 13. 最终记忆图

```text
LoRA：
原始 W 冻结
   ↓
训练 A、B
   ↓
W' = W + BA


GaLore：
原始 W 仍然训练
   ↓
完整梯度 G
   ↓
低秩投影 G_low
   ↓
Adam 的 m、v 在低维空间工作
   ↓
得到 Δ_low
   ↓
映射回 ΔW
   ↓
更新真正的 W
```

最核心区别：

$$
\boxed{
LoRA：压缩“可训练参数”
}
$$

$$
\boxed{
GaLore：压缩“梯度 / 优化器状态”
}
$$

---

# 14. 常用记忆口诀

- `cutoff_len`：一条样本最多多长
- `batch_size`：一次往 GPU 塞几条
- `gradient_accumulation`：算几次再更新
- `warmup`：起步慢多久
- `scheduler`：学习率怎么随训练变化
- `max_grad_norm`：梯度限速器
- `seed`：控制随机性
- `LoRA rank`：LoRA 用多少个低秩方向
- `LoRA alpha`：LoRA 分支的缩放强度
- `GaLore rank`：梯度保留多少个主要方向
- `GaLore update interval`：多久重新找一次主要梯度方向
- `Adam m`：梯度的一阶移动平均
- `Adam v`：梯度平方的二阶移动平均
- `Δ`：最终参数更新量，不是梯度
