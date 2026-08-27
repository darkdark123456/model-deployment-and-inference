# BAdam 与 APOLLO 学习笔记（Typora 无公式依赖版）

> 这个版本完全不依赖 LaTeX / MathJax。  
> 不使用 `$$`、`\begin{bmatrix}`、`\rightarrow` 等数学渲染语法。  
> 因此即使 Typora 没有开启“数学公式”功能，也可以正常阅读。

---

# 1. BAdam

## 1.1 核心思想

普通全参数 AdamW：

> 所有参数同时参与训练。

假设模型有 4 个 Transformer Block：

```text
Block 0
Block 1
Block 2
Block 3
```

普通 AdamW：

```text
Block 0   训练
Block 1   训练
Block 2   训练
Block 3   训练
```

每个 Block 都需要维护 Adam 的两个主要状态：

```text
m：梯度的一阶移动平均
v：梯度平方的二阶移动平均
```

也就是：

```text
Block 0 → m0、v0
Block 1 → m1、v1
Block 2 → m2、v2
Block 3 → m3、v3
```

BAdam 的核心想法是：

> 不让所有 Block 同时训练，而是一次只训练一部分，训练一段时间后再切换。

例如：

```text
阶段 1：
Block 0   训练
Block 1   冻结
Block 2   冻结
Block 3   冻结

阶段 2：
Block 0   冻结
Block 1   训练
Block 2   冻结
Block 3   冻结

阶段 3：
Block 0   冻结
Block 1   冻结
Block 2   训练
Block 3   冻结
```

因此可以记成：

```text
普通 Adam：
所有参数一起训练

BAdam：
参数分组轮班训练
```

---

## 1.2 Layer 模式

参数：

```text
BAdam 模式 = layer
```

意思：

> 按 Transformer Block / Layer 为单位轮流训练。

假设模型有 4 个 Block，并采用：

```text
ascending
```

训练顺序：

```text
Block 0
   ↓
Block 1
   ↓
Block 2
   ↓
Block 3
   ↓
再回到 Block 0
```

如果：

```text
switch_interval = 50
```

那么：

```text
step 1 ~ 50：
训练 Block 0

step 51 ~ 100：
训练 Block 1

step 101 ~ 150：
训练 Block 2

step 151 ~ 200：
训练 Block 3
```

---

## 1.3 Ratio 模式

如果：

```text
BAdam 模式 = ratio
```

意思：

> 每次只更新一定比例的参数。

例如模型有：

```text
100,000,000 个参数
```

设置：

```text
update_ratio = 0.05
```

计算：

```text
100,000,000 × 0.05
= 5,000,000
```

也就是：

```text
一次大约更新 5% 的参数
```

因此：

```text
layer = 按模型结构分块

ratio = 按参数比例分块
```

---

## 1.4 切换策略

### ascending

从前往后：

```text
Block 0 → Block 1 → Block 2 → Block 3
```

### descending

从后往前：

```text
Block 3 → Block 2 → Block 1 → Block 0
```

### random

随机选择：

```text
例如：
Block 2 → Block 0 → Block 3 → Block 1
```

### fixed

固定在某一个 Block，不自动切换。

---

## 1.5 切换频率

例如：

```text
switch_interval = 50
```

表示：

> 当前活动 Block 连续训练 50 个 step 后，再切换。

注意：

```text
不是：
每 50 步重置 Adam

而是：
每 50 步切换当前参与训练的参数块
```

---

## 1.6 Block 更新比例

例如：

```text
update_ratio = 0.05
```

主要用于：

```text
mode = ratio
```

表示：

```text
每次大约更新 5% 的参数
```

如果当前是：

```text
mode = layer
```

这个参数通常不会主导 layer-wise 的切换。

---

## 1.7 BAdam 为什么省显存

普通 Adam：

```text
所有参数同时训练
      ↓
所有参数同时需要 m、v
      ↓
优化器状态很大
```

BAdam：

```text
当前只训练一部分参数
      ↓
同一时刻需要管理的优化器状态更少
      ↓
降低峰值显存
```

一句话记忆：

> BAdam 不压缩梯度维度，而是减少“同一时刻参与训练的参数数量”。

---

# 2. APOLLO

## 2.1 核心思想

APOLLO 的完整流程：

```text
完整梯度 G
    ↓
随机投影
    ↓
低秩梯度 R
    ↓
低维 Adam 的 m、v
    ↓
得到 Adam 处理后的 R_tilde
    ↓
计算缩放系数
    ↓
组成缩放矩阵 S
    ↓
完整梯度 G × S
    ↓
得到参数更新量 ΔW
    ↓
更新 W
```

简写：

```text
G → R → m,v → R_tilde → S → G×S → ΔW → W_new
```

各符号：

```text
G       = 完整梯度
R       = 随机投影后的低秩梯度
m       = Adam 一阶矩
v       = Adam 二阶矩
R_tilde = Adam 处理后的低维结果
S       = 缩放矩阵
G×S     = 缩放后的完整梯度
ΔW      = 参数更新量
W_new   = 更新后的权重
```

最核心：

> 低维空间负责估计“Adam 应该怎么缩放梯度”，完整空间负责真正更新 W。

---

## 2.2 APOLLO Rank

假设完整梯度尺寸：

```text
G：m × n
```

APOLLO 的 rank：

```text
r
```

随机投影矩阵：

```text
P：r × m
```

计算：

```text
R = P × G
```

因此：

```text
R：r × n
```

例如：

```text
G：4096 × 4096
rank：16
```

得到：

```text
R：16 × 4096
```

如果：

```text
rank = 1
```

那么：

```text
R：1 × n
```

可以记成：

```text
rank 小：
更省显存，但低秩估计更粗

rank 大：
信息更多，但优化器状态更大
```

---

# 3. APOLLO：3×3 + Rank=1 完整计算

## 3.1 原始完整梯度 G

假设：

```text
G =

[ 3  1  2 ]
[ 4  2  0 ]
[ 0  1  1 ]
```

尺寸：

```text
3 × 3
```

设置：

```text
rank = 1
```

为了方便手算，假设随机投影矩阵：

```text
P = [0.6  0.8  0]
```

P 的尺寸：

```text
1 × 3
```

---

## 3.2 第一步：随机投影

公式写成普通文本：

```text
R = P × G
```

代入：

```text
R = [0.6  0.8  0]

    ×

    [ 3  1  2 ]
    [ 4  2  0 ]
    [ 0  1  1 ]
```

第一列：

```text
0.6×3 + 0.8×4 + 0×0
= 1.8 + 3.2
= 5
```

第二列：

```text
0.6×1 + 0.8×2 + 0×1
= 0.6 + 1.6
= 2.2
```

第三列：

```text
0.6×2 + 0.8×0 + 0×1
= 1.2
```

所以：

```text
R = [5  2.2  1.2]
```

尺寸变化：

```text
G：3 × 3
   ↓
R：1 × 3
```

---

## 3.3 第二步：计算 Adam 的 m

设：

```text
β1 = 0.9
```

初始：

```text
m0 = [0  0  0]
```

公式：

```text
m1 = β1×m0 + (1-β1)×R
```

代入：

```text
m1
= 0.9×[0 0 0]
+ 0.1×[5 2.2 1.2]
```

得到：

```text
m1 = [0.5  0.22  0.12]
```

---

## 3.4 第三步：计算 Adam 的 v

设：

```text
β2 = 0.999
```

初始：

```text
v0 = [0  0  0]
```

公式：

```text
v1 = β2×v0 + (1-β2)×R²
```

先平方：

```text
R² = [25  4.84  1.44]
```

因此：

```text
v1
= 0.001 × [25 4.84 1.44]
```

得到：

```text
v1 = [0.025  0.00484  0.00144]
```

---

## 3.5 第四步：偏差修正

一阶矩：

```text
m_hat = m1 / (1-β1)
```

由于：

```text
1 - 0.9 = 0.1
```

所以：

```text
m_hat
= [0.5 0.22 0.12] / 0.1
= [5 2.2 1.2]
```

二阶矩：

```text
v_hat = v1 / (1-β2)
```

由于：

```text
1 - 0.999 = 0.001
```

所以：

```text
v_hat
= [0.025 0.00484 0.00144] / 0.001
= [25 4.84 1.44]
```

---

## 3.6 第五步：得到 Adam 处理后的低维结果

计算：

```text
R_tilde
= m_hat / (sqrt(v_hat) + ε)
```

为了方便手算，忽略非常小的 ε。

第一项：

```text
5 / sqrt(25)
= 5/5
= 1
```

第二项：

```text
2.2 / sqrt(4.84)
= 2.2/2.2
= 1
```

第三项：

```text
1.2 / sqrt(1.44)
= 1.2/1.2
= 1
```

所以：

```text
R_tilde = [1  1  1]
```

到这里：

```text
R
[5  2.2  1.2]

        ↓ Adam

R_tilde
[1  1  1]
```

---

## 3.7 第六步：计算缩放系数

APOLLO 比较：

```text
Adam 处理后的低维结果
÷
原来的低维梯度
```

更一般地说，是比较每一列处理前后的范数。

在 rank=1 的例子里，每一列只有一个数，所以很简单。

第一列：

```text
s1 = 1 / 5
   = 0.2
```

第二列：

```text
s2 = 1 / 2.2
   ≈ 0.4545
```

第三列：

```text
s3 = 1 / 1.2
   ≈ 0.8333
```

得到：

```text
缩放系数：

[0.2, 0.4545, 0.8333]
```

---

## 3.8 第七步：组成缩放矩阵 S

把三个缩放系数放到对角线上：

```text
S =

[ 0.2      0         0      ]
[ 0        0.4545    0      ]
[ 0        0         0.8333 ]
```

S 的尺寸：

```text
3 × 3
```

这里要注意符号：

```text
R = 低秩梯度

S = 缩放矩阵
```

不要把它们混在一起。

---

## 3.9 第八步：重新缩放完整梯度 G

APOLLO 不直接使用 R_tilde 更新 W。

它重新回到完整梯度：

```text
G =

[ 3  1  2 ]
[ 4  2  0 ]
[ 0  1  1 ]
```

然后计算：

```text
G_scaled = G × S
```

因为 S 是对角矩阵，所以等价于：

```text
G 第1列 × 0.2
G 第2列 × 0.4545
G 第3列 × 0.8333
```

第一列：

```text
[3]            [0.6]
[4] × 0.2  =   [0.8]
[0]            [0  ]
```

第二列：

```text
[1]               [0.4545]
[2] × 0.4545  ≈   [0.9090]
[1]               [0.4545]
```

第三列：

```text
[2]               [1.6666]
[0] × 0.8333  ≈   [0     ]
[1]               [0.8333]
```

所以：

```text
G_scaled ≈

[ 0.6    0.4545   1.6666 ]
[ 0.8    0.9090   0      ]
[ 0      0.4545   0.8333 ]
```

---

## 3.10 第九步：得到参数更新量 ΔW

假设学习率：

```text
η = 0.1
```

先设：

```text
APOLLO scale = 1
```

参数更新量：

```text
ΔW = -η × G_scaled
```

所以：

```text
ΔW ≈

[ -0.0600   -0.04545   -0.16666 ]
[ -0.0800   -0.09090    0       ]
[  0        -0.04545   -0.08333 ]
```

最终：

```text
W_new = W_old + ΔW
```

注意：

```text
G  = 梯度

ΔW = 最终参数更新量
```

ΔW 不是“更新后的梯度”。

---

# 4. APOLLO 参数说明

## 4.1 Rank

例如：

```text
rank = 16
```

表示：

> 低秩辅助空间的维度为 16。

简单理解：

```text
rank 小
→ 更省优化器状态
→ 估计更粗

rank 大
→ 保留更多信息
→ 更占显存
```

---

## 4.2 更新间隔

例如：

```text
update_interval = 200
```

表示：

```text
step 0：
生成随机投影 P1

step 1 ~ 199：
使用 P1

step 200：
更新/重新生成投影

step 200 ~ 399：
使用新的投影
```

---

## 4.3 APOLLO Scale

例如：

```text
scale = 32
```

可以粗略理解成：

```text
最终参数更新
=
-learning_rate
× APOLLO_scale
× G_scaled
```

即：

```text
ΔW = -η × α × G_scaled
```

其中：

```text
η = 学习率
α = APOLLO scale
```

---

## 4.4 Target

例如：

```text
target = all
```

表示：

> 对所有符合条件的线性层应用 APOLLO。

常见模块：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

---

# 5. APOLLO 为什么省显存

假设：

```text
G：4096 × 4096
```

普通 Adam：

```text
m：4096 × 4096
v：4096 × 4096
```

单个状态：

```text
4096 × 4096
= 16,777,216 个值
```

m + v：

```text
33,554,432 个值
```

如果：

```text
APOLLO rank = 16
```

低维状态类似：

```text
m_R：16 × 4096
v_R：16 × 4096
```

单个：

```text
16 × 4096
= 65,536
```

两个：

```text
131,072
```

比例：

```text
33,554,432 / 131,072
= 256
```

也就是说，只看这一层的 Adam 状态尺寸：

```text
约相差 256 倍
```

但整卡显存不会真的降低 256 倍，因为还有：

```text
模型参数
完整梯度
激活值
CUDA buffer
临时中间结果
```

---

# 6. APOLLO 与 GaLore 的区别

GaLore：

```text
完整梯度 G
    ↓
低秩投影
    ↓
G_low
    ↓
Adam
    ↓
低维更新 Δ_low
    ↓
映射回原空间
    ↓
ΔW
```

核心：

> 低维空间直接产生最终更新方向。

APOLLO：

```text
完整梯度 G
    ↓
随机投影
    ↓
R
    ↓
低维 Adam
    ↓
R_tilde
    ↓
估计缩放规律 S
    ↓
回到完整梯度 G
    ↓
G × S
    ↓
ΔW
```

核心：

> 低维空间主要用来估计 Adam 对完整梯度的缩放规律。

---

# 7. LoRA / BAdam / GaLore / APOLLO 对比

| 方法 | 原模型参数是否直接更新 | 主要省显存方式 |
|---|---|---|
| LoRA | 大部分不直接更新 | 只训练 A、B |
| BAdam | 是 | 一次只训练一部分参数 |
| GaLore | 是 | 在低秩梯度空间维护优化器状态和更新 |
| APOLLO | 是 | 在低秩空间估计 Adam 的缩放规律 |

---

# 8. 最终一句话记忆

```text
LoRA：
少训练参数

BAdam：
一次少训练一部分参数

GaLore：
把梯度更新放进低秩空间

APOLLO：
在低秩空间观察 Adam 怎么缩放梯度，
再拿这个缩放规律调整完整梯度
```

APOLLO 最核心流程：

```text
G
↓
随机投影
↓
R
↓
m、v
↓
R_tilde
↓
S
↓
G × S
↓
ΔW
↓
W_new
```
