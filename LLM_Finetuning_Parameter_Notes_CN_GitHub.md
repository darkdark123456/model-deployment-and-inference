# 大模型微调参数学习笔记（GitHub 完全兼容版）

> 说明：本版本**不使用 LaTeX 数学公式渲染**，所有公式和矩阵都使用普通文本/代码块表示。  
> 因此无论 GitHub、VS Code、Typora 还是普通 Markdown 阅读器，都能直接看懂，不会出现 `\begin{bmatrix}`、`$$` 等无法渲染的问题。

---

## 目录

1. RoPE 与长上下文扩展
2. 训练加速方式
3. 梯度裁剪与随机种子
4. 截断长度、Batch Size 与梯度累积
5. 学习率调节器与 Warmup
6. 其他常用训练参数
7. Freeze 部分参数微调
8. LoRA 参数
9. RLHF
10. 多模态参数
11. GaLore
12. GaLore + Adam：3×3 完整计算

---

# 1. RoPE 与长上下文扩展

## 1.1 RoPE 的频率公式

RoPE 每一组旋转频率：

```text
ω_i = 1 / θ^(2i/d)
```

其中：

- `θ`：RoPE base，例如 10000
- `d`：参与 RoPE 的维度
- `i`：第几组二维旋转
- `ω_i`：该组旋转频率

某个位置 `p` 的旋转角：

```text
φ_i(p) = p × ω_i
```

### 统一例子

模型原生上下文：

```text
8000 token
```

目标上下文：

```text
32000 token
```

扩展倍数：

```text
factor = 32000 / 8000 = 4
```

为了方便手算，设：

```text
θ = 10000
d = 8
```

那么：

```text
ω_0 = 1
ω_1 = 0.1
ω_2 = 0.01
ω_3 = 0.001
```

最终：

```text
ω = [1, 0.1, 0.01, 0.001]
```

可以粗略理解：

- 高频：更擅长区分相邻 token
- 低频：更适合描述长距离位置

---

## 1.2 none

不进行任何 RoPE 扩展：

```text
ω'_i = ω_i
```

特点：

- 原生短距离位置能力完全保留
- 超过 8000 token 后容易进入模型没训练过的位置范围

---

## 1.3 linear

公式：

```text
ω'_i = ω_i / factor
```

这里：

```text
factor = 4
```

所以：

```text
原来：
[1, 0.1, 0.01, 0.001]

Linear 后：
[0.25, 0.025, 0.0025, 0.00025]
```

它也可以理解为：

```text
p' = p / 4
```

所以：

```text
32000 -> 8000
```

优点：

- 简单
- 能把 32000 的位置尺度压回模型原生 8000 范围

缺点：

- 近距离位置也一起被压缩

---

## 1.4 Dynamic NTK

Dynamic 不直接做：

```text
ω / 4
```

而是先修改 `θ`：

```text
θ' = θ × [ f × (L/L0) - (f-1) ] ^ (d/(d-2))
```

统一参数：

```text
f  = 4
L  = 32000
L0 = 8000
d  = 8
θ  = 10000
```

先算：

```text
L/L0 = 32000/8000 = 4

f × (L/L0) - (f-1)
= 4×4 - 3
= 13
```

于是：

```text
θ' = 10000 × 13^(8/6)
   ≈ 305674
```

然后再利用新的 `θ'` 重新计算每一组频率。

特点：

```text
高频：变化少
中频：变化更多
低频：变化更大
```

所以 Dynamic 不是简单的“动态 Linear”。

---

## 1.5 YaRN

可以粗略理解为：

```text
ω' = α×ω + (1-α)×(ω/factor)
```

其中：

```text
高频：α 接近 1
中频：0 < α < 1
低频：α 接近 0
```

意思：

```text
高频：尽量保留原频率
中频：原频率和 Linear 结果混合
低频：更多采用 Linear 缩放结果
```

核心思想：

> 保护近距离位置能力，同时扩展长距离能力。

---

## 1.6 Llama3 RoPE

先根据频率算波长：

```text
λ = 2π / ω
```

然后根据波长决定：

```text
短波长 / 高频：
基本不改

中间波长：
平滑过渡

长波长 / 低频：
按 factor 缩放
```

---

## 1.7 怎么比较这些方法

不能理解成：

```text
哪个数字更大，哪个就更好
```

真正要同时考虑两个目标。

### 目标 1：近距离

例如：

```text
token 1000
token 1001
```

两者只差 1 token。

希望模型仍然能够精确区分。

### 目标 2：远距离

例如：

```text
token 1000
token 31000
```

两者相距 30000 token。

原模型只有 8000 上下文，因此必须让负责长距离的 RoPE 频率进行扩展。

| 方法 | 近距离 | 长距离 |
|---|---|---|
| none | 保持最好 | 容易超纲 |
| linear | 近距离也被压缩 | 扩展直接 |
| dynamic | 高频少改 | 低频变化大 |
| YaRN | 高频保护好 | 低频逐渐扩展 |
| Llama3 | 高频保护好 | 按波长分段扩展 |

---

# 2. 训练加速方式

这类参数不是长上下文扩展，而是：

> 让训练/推理更快、更省显存。

常见：

```text
auto
flashattn2
unsloth
liger_kernel
```

## 2.1 auto

框架自己判断使用什么加速实现。

适合：

- 第一次跑训练
- 优先考虑兼容性

---

## 2.2 FlashAttention2

Attention 仍然计算：

```text
Attention(Q,K,V)
= softmax(QK^T / sqrt(d)) × V
```

FlashAttention2 不改变数学结果，主要改变 GPU 的计算方式：

```text
普通 Attention：
计算
-> 写显存
-> 再读
-> 再计算
-> 再写

FlashAttention2：
分块
-> 尽量在高速缓存里连续完成
-> 减少显存读写
```

序列越长，通常收益越明显。

---

## 2.3 Unsloth

更像整体训练流程优化：

```text
Attention
LoRA
RoPE
RMSNorm
反向传播
显存管理
```

常用于：

```text
LoRA
QLoRA
单卡训练
```

---

## 2.4 Liger Kernel

主要提供高性能 GPU kernel，例如：

```text
RMSNorm
RoPE
SwiGLU
CrossEntropy
Fused Kernel
```

核心：

> 减少 kernel 启动次数和显存搬运。

---

# 3. 梯度裁剪与随机种子

## 3.1 梯度裁剪

例如：

```text
max_grad_norm = 1.0
```

假设梯度：

```text
g = [3, 4]
```

L2 范数：

```text
||g|| = sqrt(3^2 + 4^2)
      = 5
```

因为：

```text
5 > 1
```

所以整体按比例缩小：

```text
缩放比例 = 1/5 = 0.2

[3,4] -> [0.6,0.8]
```

检查：

```text
sqrt(0.6^2 + 0.8^2)
= 1
```

通用公式：

```text
g' = g × min(1, max_norm / ||g||)
```

注意：

> 梯度裁剪不是把每个梯度强行限制为 1。

真正做的是：

> 保留方向，只限制整个梯度向量的长度。

---

## 3.2 随机种子

例如：

```text
seed = 42
```

会影响：

```text
数据 shuffle
dropout
参数随机初始化
随机采样
```

作用：

> 尽量保证实验可复现。

`42` 并没有特殊训练优势。

---

# 4. 截断长度、Batch Size 与梯度累积

假设：

```text
cutoff_len = 2048
batch_size = 4
gradient_accumulation_steps = 8
```

## 4.1 截断长度 2048

表示：

> 单条训练样本最多 2048 token。

例如：

```text
500 token  -> 全部保留
1500 token -> 全部保留
3000 token -> 超过部分截断
```

---

## 4.2 Batch Size = 4

表示：

> 每张 GPU 一次 forward/backward 同时处理 4 条样本。

---

## 4.3 梯度累积 = 8

表示：

> 连续计算 8 次梯度以后，再执行一次 optimizer.step()。

流程：

```text
第1次：4条样本 -> 算梯度 -> 不更新
第2次：4条样本 -> 累积梯度 -> 不更新
...
第8次：4条样本 -> 累积完成
                    |
                    v
               optimizer.step()
```

单 GPU 有效 Batch：

```text
4 × 8 = 32
```

两张 GPU：

```text
4 × 8 × 2 = 64
```

公式：

```text
effective_batch
= per_device_batch
× gradient_accumulation
× GPU_count
```

注意：

> `4×8×2048` 只是一次参数更新综合了很多 token，不代表模型一次能看到这么长的上下文。

---

# 5. 学习率调节器与 Warmup

## 5.1 linear

Warmup 后，学习率沿直线下降。

---

## 5.2 cosine

大致公式：

```text
LR(t) = LR_max/2 × [1 + cos(πt)]
```

特点：

- 比较平滑
- 大模型训练常见

---

## 5.3 cosine_with_restarts

学习率余弦下降后重新拉高，再进行下一轮下降。

---

## 5.4 polynomial

例如：

```text
LR = LR_max × (1-t)^p
```

---

## 5.5 constant

训练全程保持同一学习率。

---

## 5.6 constant_with_warmup

先预热，再保持固定学习率。

---

## 5.7 inverse_sqrt

大致：

```text
LR ∝ 1 / sqrt(step)
```

---

## 5.8 reduce_lr_on_plateau

当验证集指标长期不改善时再降低学习率。

---

## 5.9 cosine_with_min_lr

使用 cosine，但设置最低学习率，不降到 0。

---

## 5.10 warmup_stable_decay

三段：

```text
Warmup
-> Stable
-> Decay
```

---

## 5.11 Warmup 预热

例如：

```text
learning_rate = 2e-5
warmup_steps = 100
```

大概：

```text
step 0   -> LR ≈ 0
step 25  -> LR ≈ 5e-6
step 50  -> LR ≈ 1e-5
step 75  -> LR ≈ 1.5e-5
step 100 -> LR = 2e-5
```

作用：

> 训练刚开始时先小步走，避免一上来学习率太大导致不稳定。

如果：

```text
warmup_ratio = 0.1
总训练步数 = 1000
```

那么：

```text
warmup_steps = 1000 × 0.1 = 100
```

---

# 6. 其他常用训练参数

## logging_steps

```text
logging_steps = 5
```

每 5 个 global step 打印一次日志。

## save_steps

```text
save_steps = 100
```

每 100 step 保存一次 checkpoint。

## NEFTune

在 embedding 上加入小噪声作为正则化。

```text
neftune_noise_alpha = 0
```

表示关闭。

## packing

把多个短样本打包进一个长训练序列，减少 padding 浪费。

## neat packing

多个样本虽然放进同一个 packed sequence，但避免它们之间互相 Attention。

## train_on_prompt

决定用户 prompt 部分是否也计算 loss。

普通 SFT 常见：

> prompt 不算 loss，只训练 assistant 输出。

## mask_history

多轮对话时：

> 历史轮只作为上下文，通常只训练最后一轮回答。

## resize_vocab

新增 tokenizer token 后，可能需要修改 embedding / lm_head 大小。

---

# 7. Freeze 部分参数微调

假设模型有 32 层。

## 可训练层数 = 2

表示：

```text
Layer 0  ~ Layer 29：冻结
Layer 30             ：训练
Layer 31             ：训练
```

如果：

```text
-2
```

则训练最前面的两层。

## 可训练模块 = all

不是整个模型都训练。

意思是：

> 在已经选中的可训练层内部，所有模块训练。

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

表示：

> 除了指定 Transformer 层，额外训练 lm_head。

---

# 8. LoRA

LoRA 不直接训练整个原权重。

原权重：

```text
W
```

新增两个低秩矩阵：

```text
A
B
```

权重变化：

```text
ΔW = B × A
```

最终：

```text
W' = W + (alpha/r) × B × A
```

---

## 8.1 rank

假设：

```text
W = 4096 × 4096
r = 8
```

那么：

```text
A = 8 × 4096
B = 4096 × 8
```

LoRA 参数数：

```text
8×4096 + 4096×8
= 65536
```

原矩阵参数：

```text
4096×4096
= 16777216
```

所以 LoRA 训练参数少很多。

---

## 8.2 alpha

假设：

```text
r = 8
alpha = 16
```

缩放：

```text
alpha/r = 16/8 = 2
```

最终：

```text
W' = W + 2BA
```

可以把 alpha 理解成：

> LoRA 修改原模型的“音量”。

---

## 8.3 dropout

例如：

```text
lora_dropout = 0.04
```

LoRA 分支训练时随机丢弃少量输入，帮助正则化。

---

## 8.4 LoRA+

让 A、B 使用不同学习率。

例如：

```text
λ = lr_B / lr_A
```

所以：

```text
lr_B = λ × lr_A
```

---

## 8.5 rsLoRA

普通 LoRA 缩放：

```text
alpha / r
```

rsLoRA：

```text
alpha / sqrt(r)
```

大 rank 时通常更稳定。

---

## 8.6 DoRA

把权重的：

```text
方向
大小
```

分开处理。

---

## 8.7 PiSSA

利用原始权重的 SVD 信息初始化 LoRA，让低秩方向一开始更有信息。

---

# 9. RLHF

推荐论文：

1. Deep Reinforcement Learning from Human Preferences
2. Proximal Policy Optimization Algorithms
3. Learning to Summarize from Human Feedback
4. Training language models to follow instructions with human feedback（InstructGPT）
5. Direct Preference Optimization（DPO）

经典 RLHF：

```text
SFT
↓
Reward Model
↓
PPO
```

DPO：

> 不单独训练 Reward Model，也不跑 PPO，直接使用 chosen / rejected 数据优化模型。

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

视觉编码器继续参与前向，但参数不更新。

## 冻结投影器

投影器负责把视觉特征映射到 LLM hidden space。

例如：

```text
1024维视觉特征
↓
Projector
↓
4096维语言模型特征
```

## 冻结语言模型

LLM 主干参与前向，但自身参数不更新。

## 图像最大/最小像素

分辨率越高：

```text
细节更多
视觉 token 更多
显存更高
计算更多
```

## 视频

视频输入还有时间维度：

```text
T × H × W
```

所以即使每帧分辨率较低，总计算仍可能很大。

---

# 11. GaLore

GaLore 的核心：

> 原模型权重 W 仍然真正更新，但梯度和 Adam 优化器状态在低秩空间中处理。

---

## 11.1 LoRA vs GaLore

LoRA：

```text
原始 W 冻结
↓
只训练 A、B
↓
W' = W + BA
```

GaLore：

```text
原始 W 仍然训练
↓
完整梯度 G
↓
压缩为低秩梯度 G_low
↓
Adam 在低维空间计算
↓
再映射回原参数空间
↓
更新 W
```

最核心区别：

```text
LoRA：
压缩“可训练参数”

GaLore：
压缩“梯度 / 优化器状态”
```

---

## 11.2 GaLore rank

例如原梯度：

```text
4096 × 4096
```

如果：

```text
rank = 16
```

低秩梯度可以类似：

```text
16 × 4096
```

大幅减少优化器状态尺寸。

---

## 11.3 更新间隔

例如：

```text
update_interval = 200
```

意思：

```text
step 0   ：计算投影方向 P1
step 1~199：继续使用 P1

step 200 ：重新计算 P2
step 200~399：使用 P2

step 400 ：重新计算 P3
...
```

原因：

> 训练过程中最重要的梯度方向会发生变化。

---

# 12. GaLore + Adam：3×3 完整计算

这是最关键的例子。

假设某层完整梯度：

```text
G =
[ 3  0  0 ]
[ 4  0  0 ]
[ 0  0  0 ]
```

GaLore：

```text
rank = 1
```

---

## 12.1 找最主要方向

第一列：

```text
[3]
[4]
[0]
```

长度：

```text
sqrt(3^2 + 4^2)
= sqrt(25)
= 5
```

单位方向：

```text
P =
[0.6]
[0.8]
[0  ]
```

因为：

```text
3/5 = 0.6
4/5 = 0.8
```

---

## 12.2 将 3×3 梯度压成 1×3

先转置：

```text
P^T = [0.6  0.8  0]
```

公式：

```text
G_low = P^T × G
```

代入：

```text
G_low

= [0.6  0.8  0]

  ×

  [ 3  0  0 ]
  [ 4  0  0 ]
  [ 0  0  0 ]
```

第一项：

```text
0.6×3 + 0.8×4 + 0×0
= 1.8 + 3.2
= 5
```

另外两项为 0。

所以：

```text
G_low = [5  0  0]
```

也就是：

```text
原来：3×3 梯度
↓
GaLore rank=1
↓
现在：1×3 梯度
```

---

# 12.3 Adam 不直接用梯度更新参数

Adam 会保存两个状态：

```text
m：一阶矩，梯度的移动平均
v：二阶矩，梯度平方的移动平均
```

常见：

```text
β1 = 0.9
β2 = 0.999
```

现在：

```text
g1 = [5  0  0]

m0 = [0  0  0]
v0 = [0  0  0]
```

---

## 12.4 计算 m

公式：

```text
m1 = β1×m0 + (1-β1)×g1
```

代入：

```text
m1
= 0.9×[0 0 0]
+ 0.1×[5 0 0]

= [0.5  0  0]
```

所以：

```text
m1 = [0.5  0  0]
```

---

## 12.5 计算 v

公式：

```text
v1 = β2×v0 + (1-β2)×g1^2
```

先平方：

```text
g1^2 = [25  0  0]
```

所以：

```text
v1
= 0.999×[0 0 0]
+ 0.001×[25 0 0]

= [0.025  0  0]
```

得到：

```text
v1 = [0.025  0  0]
```

---

## 12.6 偏差修正

一阶矩：

```text
m_hat = m1 / (1 - β1)
```

代入：

```text
m_hat
= 0.5 / (1-0.9)
= 0.5 / 0.1
= 5
```

所以：

```text
m_hat = [5  0  0]
```

二阶矩：

```text
v_hat = v1 / (1 - β2)
```

代入：

```text
v_hat
= 0.025 / (1-0.999)
= 0.025 / 0.001
= 25
```

所以：

```text
v_hat = [25  0  0]
```

---

## 12.7 Adam 计算参数更新量 Δ

公式：

```text
Δ = -η × m_hat / (sqrt(v_hat) + ε)
```

注意：

> Δ 不是“更新后的梯度”。

它是：

> Adam 最终决定的“参数实际要变化多少”。

假设：

```text
η = 0.1
```

第一项：

```text
sqrt(25) = 5
```

所以：

```text
Δ_low
≈ -0.1 × 5/5
= -0.1
```

得到：

```text
Δ_low = [-0.1  0  0]
```

---

## 12.8 将低维更新量映射回 3×3

公式：

```text
ΔW = P × Δ_low
```

代入：

```text
P =
[0.6]
[0.8]
[0  ]

Δ_low = [-0.1  0  0]
```

矩阵乘法：

```text
ΔW =

[0.6]                    [ -0.06   0   0 ]
[0.8] × [-0.1 0 0]  =   [ -0.08   0   0 ]
[0  ]                    [  0      0   0 ]
```

因此：

```text
ΔW =
[ -0.06   0   0 ]
[ -0.08   0   0 ]
[  0      0   0 ]
```

最终更新原权重：

```text
W_new = W_old + ΔW
```

---

# 12.9 整个 GaLore + Adam 链路

```text
完整梯度 G
3×3
   ↓
GaLore 投影
   ↓
低秩梯度 G_low
1×3
   ↓
Adam 计算
m、v
   ↓
偏差修正
m_hat、v_hat
   ↓
得到低维参数更新量
Δ_low
   ↓
映射回原空间
   ↓
得到 ΔW
3×3
   ↓
W_new = W_old + ΔW
```

也可以记成：

```text
G
-> G_low
-> m,v
-> m_hat,v_hat
-> Δ_low
-> ΔW
-> W_new
```

---

# 12.10 为什么 GaLore + Adam 能省显存

普通 Adam：

```text
G：3×3  -> 9个值
m：3×3  -> 9个值
v：3×3  -> 9个值
```

只看 `m + v`：

```text
9 + 9 = 18 个状态值
```

GaLore rank=1 后：

```text
G_low：1×3
m：1×3 -> 3个值
v：1×3 -> 3个值
```

只看 `m + v`：

```text
3 + 3 = 6 个状态值
```

所以这个玩具例子：

```text
18 -> 6
```

真实模型更明显。

例如完整矩阵：

```text
4096 × 4096
= 16,777,216 个位置
```

GaLore rank=16 后：

```text
16 × 4096
= 65,536 个位置
```

因此 Adam 的 `m` 和 `v` 可以在远小得多的空间中保存。

---

# 13. 最终记忆口诀

```text
cutoff_len
= 一条样本最多多长

batch_size
= 一次往 GPU 塞几条

gradient_accumulation
= 算几次以后才真正更新一次参数

warmup
= 训练开始时慢慢把学习率升上去

scheduler
= 学习率之后怎么变化

max_grad_norm
= 梯度限速器

seed
= 控制随机性

LoRA rank
= LoRA 用多少个低秩方向

LoRA alpha
= LoRA 分支的缩放强度

GaLore rank
= 梯度低秩空间保留多少个主要方向

GaLore update interval
= 多久重新计算一次投影方向

Adam m
= 梯度的一阶移动平均

Adam v
= 梯度平方的二阶移动平均

Δ
= 参数更新量，不是更新后的梯度
```

---

## 最核心的两句话

```text
LoRA：
压缩“可训练参数”

GaLore：
压缩“梯度 / Adam 优化器状态”
```

以及：

```text
梯度 g
-> Adam 的 m、v
-> 参数更新量 Δ
-> 更新模型参数 W
```
