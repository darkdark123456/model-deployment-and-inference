                                    llm compressor  推理实战

------



##### AWQ推理 单卡4090 24G Qwen7B

```python
#AWQ推理 单卡4090 24G Qwen7B
"""下载模型"""
save_path="./models"

from modelscope import snapshot_download
model_dir = snapshot_download(
 'Qwen/Qwen2-7B-Instruct',
 cache_dir=save_path
)



"""加载分词器与模型"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path="autodl-tmp/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"
tokenizer=AutoTokenizer.from_pretrained(
 path,
 trust_remote_code=True
)

model=AutoModelForCausalLM.from_pretrained(
 path,
 torch_dtype=torch.bfloat16,
 device_map={"": "cuda:0"},
)

print("加载成功")

#量化后的模型保存路径
print(torch.cuda.get_device_name())
output_dir="./Qwen2-7B-AWQ-INT4"



#构造数据集 用于寻找激活值的分布
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import Dataset

texts = [
 "人工智能正在改变现代工业。",
 "大型语言模型需要大量计算资源。",
 "Transformer架构是目前主流的大模型架构。",
 "模型压缩可以降低推理成本。",
]*100


dataset = Dataset.from_dict(
 {
 "text": texts
 }
)

print(len(dataset))


from llmcompressor.modifiers.transform.awq import AWQMapping
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


#需要量化的层，根据模型的架构进行实际构造
mappings = [
 AWQMapping(
 "re:.*input_layernorm",
 [
 "re:.*q_proj",
 "re:.*k_proj",
 "re:.*v_proj"
 ]
 ),
    
 AWQMapping(
 "re:.*post_attention_layernorm",
 [
 "re:.*gate_proj",
 "re:.*up_proj"
 ]
 ),
    
 AWQMapping(
 "re:.*up_proj",
 [
 "re:.*down_proj"
 ]
 )
]

#构造量化配置输入文件
recipe=[

 AWQModifier(
 mappings=mappings
 ),
    
 QuantizationModifier(
 ignore=["lm_head"],
 scheme="W4A16_ASYM",
 targets=["Linear"]
 )
]


#进行量化
oneshot(
 model=model,
 tokenizer=tokenizer,
 dataset=dataset,
 recipe=recipe,
 max_seq_length=512,
 num_calibration_samples=128,
 output_dir="./Qwen2-7B-AWQ-INT4"
)

#保存分词器
tokenizer.save_pretrained(
 output_dir
)
```

------

##### lm_eval 模型的能力测试

```html
单卡能力测试
```

```
lm_eval \
--model hf \
--model_args pretrained=/root/autodl-tmp/models/models/Qwen--
Qwen2-7B-Instruct/snapshots/master,dtype=bfloat16 \
--tasks mmlu,gsm8k \
--device cuda:0 \
--batch_size auto \
--apply_chat_template \
--output_path ./fp16_results_test


```

```html
多卡能力测试  lm_eval多卡能力测试命令-将能力测试 数据并行 到4张相同的4090显卡上，最后汇总结果。

实践结果表明：与单卡相比能缩短generate_until requests近1/4
```

```
pip install accelerate
accelerate config
```

```
accelerate launch --num_processes 4 -m lm_
eval \
--model hf \
--model_args pretrained=/root/autodl-tmp/
models/models/Qwen--Qwen2-7B-Instruct/
snapshots/master,dtype=bfloat16 \
--tasks mmlu,gsm8k \
--batch_size auto \
--apply_chat_template \
--output_path ./fp16_results
```

![image-20260819082050432](C:\Users\mzc228699\AppData\Roaming\Typora\typora-user-images\image-20260819082050432.png)

##### vllm benchmark测试

```
模型离线能力测试
```

```
CUDA_VISIBLE_DEVICES=0 \
vllm bench throughput \
--model /root/autodl-tmp/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master \
--dtype bfloat16 \
--dataset-name random \
--num-prompts 100 \
--input-len 512 \
--output-len 128 \
--output-json ./qwen2_7b_bf16_throu
```

```
CUDA_VISIBLE_DEVICES=0 \
vllm bench throughput \
--model /root/Qwen2-7B-AWQ-INT4 \
--dtype auto \
--dataset-name random \
--num-prompts 100 \
--input-len 512 \
--output-len 128 \
--output-json ./qwen2_7b_awq_throughput.json
```

```
模型在线能力测试
```

```
#启动服务
CUDA_VISIBLE_DEVICES=0 \
vllm serve \
/root/autodl-tmp/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master \
--dtype bfloat16 \
--gpu-memory-utilization 0.90 \
--max-model-len 4096 \
--port 8000

#进行测试
vllm bench serve \
--backend openai \
--base-url http://127.0.0.1:8000 \
--model /root/autodl-tmp/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master \
--dataset-name random \
--num-prompts 100 \
--input-len 512 \
--output-len 128 \
--request-rate inf
```

```
#启动服务
CUDA_VISIBLE_DEVICES=0 \
vllm serve \
/root/Qwen2-7B-AWQ-INT4 \
--dtype auto \
--gpu-memory-utilization 0.90 \
--max-model-len 4096 \
--port 8000


#进行测试
vllm bench serve \
--backend openai \
--base-url http://127.0.0.1:8000 \
--model /root/Qwen2-7B-AWQ-INT4 \
--dataset-name random \
--num-prompts 100 \
--input-len 512 \
--output-len 128 \
--request-rate inf
```

![image-20260819083447707](C:\Users\mzc228699\AppData\Roaming\Typora\typora-user-images\image-20260819083447707.png)

![image-20260819083534388](C:\Users\mzc228699\AppData\Roaming\Typora\typora-user-images\image-20260819083534388.png)

------

```
| 类别       | 指标                          | 含义                  | 好坏    |
| -------- | --------------------------- | ------------------- | ----- |
| 测试规模     | Successful requests         | 成功请求数量              | —     |
| 测试规模     | Benchmark duration          | 完成测试总时间             | ↓     |
| 数据量      | Total input tokens          | 总输入 Token           | —     |
| 数据量      | Total generated tokens      | 总生成 Token           | —     |
| **吞吐**   | **Request throughput**      | 每秒完成请求              | **↑** |
| **吞吐**   | **Output token throughput** | 每秒生成 Token          | **↑** |
| 吞吐       | Peak output throughput      | 峰值生成速度              | ↑     |
| 并发       | Peak concurrent requests    | 最大同时请求数量            | —     |
| 吞吐       | Total token throughput      | 输入+输出总处理速度          | ↑     |
| **首包延迟** | **Mean TTFT**               | 平均多久开始回答            | **↓** |
| 首包延迟     | Median TTFT                 | 典型首 Token 延迟        | ↓     |
| 尾延迟      | P99 TTFT                    | 最慢约 1% 的首 Token 情况  | ↓     |
| **生成延迟** | **Mean TPOT**               | 平均每生成一个后续 Token 的时间 | **↓** |
| 生成延迟     | Median TPOT                 | 典型 TPOT             | ↓     |
| 尾延迟      | P99 TPOT                    | TPOT 尾延迟            | ↓     |
| 流式体验     | Mean ITL                    | 平均 Token 间隔         | ↓     |
| 流式体验     | Median ITL                  | 典型 Token 间隔         | ↓     |
| 流式尾延迟    | P99 ITL                     | 偶发卡顿情况              | ↓     |

```

```
| 指标                      |      原模型 BF16 |          AWQ INT4 |       AWQ变化 |
| ----------------------- | ------------: | ----------------: | ----------: |
| Benchmark duration      |       14.64 s |       **13.74 s** |  **↓ 6.1%** |
| Request throughput      |    6.83 req/s |    **7.28 req/s** |  **↑ 6.6%** |
| Output token throughput |  874.57 tok/s |  **931.26 tok/s** |  **↑ 6.5%** |
| Peak output throughput  |    4276 tok/s |    **5500 tok/s** | **↑ 28.6%** |
| Total token throughput  | 4372.87 tok/s | **4656.32 tok/s** |  **↑ 6.5%** |
| Mean TTFT               |    9374.40 ms |    **9327.22 ms** |      ↓ 0.5% |
| Median TTFT             |    9397.68 ms |    **9342.42 ms** |      ↓ 0.6% |
| P99 TTFT                |   11715.61 ms |   **11590.84 ms** |      ↓ 1.1% |
| Mean TPOT               |      39.36 ms |      **33.70 ms** | **↓ 14.4%** |
| Median TPOT             |      39.22 ms |      **33.70 ms** | **↓ 14.1%** |
| P99 TPOT                |      53.46 ms |      **48.37 ms** |      ↓ 9.5% |
| Mean ITL                |      39.36 ms |      **33.70 ms** | **↓ 14.4%** |
| Median ITL              |      23.41 ms |      **18.02 ms** | **↓ 23.0%** |
| P99 ITL                 |     191.19 ms |     **185.41 ms** |      ↓ 3.0% |

```

##### VLLM分布式推理测试与部署

```python
#将大规模模型分布在不同的显卡上，以32B分布在2*A800 40G为 benchmark在线测试为例
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /root/Qwen2-7B-AWQ-INT4 \
--data-parallel-size 2 \
--dtype auto \
--gpu-memory-utilization 0.90 \
--max-model-len 4096 \
--port 8000



```



##### AWQ推理 2XA800 40G Qwen32B

```python
#AWQ推理 双卡A800 40G Qwen3-32B
"""下载模型"""
save_path="./models"

from modelscope import snapshot_download
model_dir = snapshot_download(
 'Qwen/Qwen3-32B',
 cache_dir=save_path
)


"""加载分词器与模型"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path="autodl-tmp/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"
tokenizer=AutoTokenizer.from_pretrained(
 path,
 trust_remote_code=True
)


#模型分片 
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "36GiB",
        1: "36GiB",
        "cpu": "120GiB",
    },
    trust_remote_code=True,
)

print("加载成功")

#量化后的模型保存路径
print(torch.cuda.get_device_name())
output_dir="./Qwen2-7B-AWQ-INT4"



#构造数据集 用于寻找激活值的分布
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import Dataset

texts = [
 "人工智能正在改变现代工业。",
 "大型语言模型需要大量计算资源。",
 "Transformer架构是目前主流的大模型架构。",
 "模型压缩可以降低推理成本。",
]*100


dataset = Dataset.from_dict(
 {
 "text": texts
 }
)

print(len(dataset))


from llmcompressor.modifiers.transform.awq import AWQMapping
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


#在实际操作中 使用2xA800 40G 量化qwen32B的过程中，smothing会导致gpu0溢出

#在 AWQ 的 Smoothing / Grid Search 阶段。AWQ 会临时保存 activation、复制权重、尝试不同 scale，所以即使模型本身已经能放进两张 A800，量化时仍可能因为临时显存不够而爆掉


recipe = [
    AWQModifier(
        offload_device=torch.device("cpu"),#在校准scale时临时的权重放到内存
        n_grid=10,#scale候选网格数量
    ),
    QuantizationModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
]

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=256,
    num_calibration_samples=64,
    pipeline="sequential",#采用顺序切分
    sequential_targets=["Linear"],#顺序处理时，按照linear细粒度切分
    sequential_offload_device="cpu",
    sequential_prefetch=False,#当前模块校准完后，释放，再加载下一个模块
    output_dir=output_dir,
)

#保存分词器
tokenizer.save_pretrained(
 output_dir
)
```

------

#### 优化：分布式oneshot AWQ推理 2XA800 40G Qwen32B 存在Bug！！！！

```
分布式oneshot量化，他不像上面那样直接将模型分片到GPU。而是每一个线程负责一个显卡，将校准数据分成两份。
```

```python
rank0-->gpu0 rank1-->gpu1。rank按需把模型块加载到对应的GPU，每个GPU处理一部分校准数据，之后同步量化统计，gpu卸载当前模块。之后rank再按需加载模型块到对应的GPU。


bug:当rank0 rank1 广播scale的时候，无法访问对方的文件夹


修改为两个rank共享仍然会存在bug
rank0 rank1 进行共享文件夹写入更新scale的时候会出现竞争现象。
offload_dir=f"./offload_cache/rank_"


```

```python
#1 初始化上下文
from compressed_tensors.offload import init_dist
init_dist()


#每一个rank负责自己的卸载目录
import os
rank=os.environ["RANK"]

offload_dir=f"./offload_cache/rank_{rank}"

os.makedirs(
    offload_dir,
    exist_ok=True
)




from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path="models/models/Qwen--Qwen3-32B/snapshots/master"
output_dir="./distribute/qwen32b"
tokenizer=AutoTokenizer.from_pretrained(
 path,
 trust_remote_code=True
)


# 2 修改模型加载方式
from compressed_tensors.offload import load_offloaded_model
with load_offloaded_model():

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto_offload",
        offload_folder=offload_dir,
        trust_remote_code=True,
    )




from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import load_dataset
from llmcompressor.datasets.utils import get_rank_partition
from huggingface_hub import login

#3 为了防止独立进程加载整个数据集并产生额外的工作量或内存占用，我们必须将数据集划分为不相交的集合。对于包含 N 个样本和 R 个进程的数据集，每个进程仅加载 N/R 个样本

#使用官方数据集
login(
    token=""
)

dataset = load_dataset(
    "allenai/c4",
    "en",
    split=get_rank_partition(
        "train",
        128
    )
)









recipe=[

AWQModifier(
    duo_scaling="both",
    n_grid=20,
),

QuantizationModifier(
    targets=["Linear"],
    ignore=[
        "lm_head",
        "embed_tokens"
    ],
    scheme="W4A16_ASYM",
)

]

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=128,
    output_dir=output_dir,
)

tokenizer.save_pretrained(
    output_dir
)

```

```python
torchrun --nproc_per_node=2 YOUR_EXAMPLE.py 运行您的脚本，即可使用两个 GPU 设备运行
```

```python
#使用自定义数据集的分布式oneshot
import os
from compressed_tensors.offload import init_dist
init_dist()

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
offload_dir=f"./offload_cache/rank_{rank}"

os.makedirs(
    offload_dir,
    exist_ok=True
)


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path="models/models/Qwen--Qwen3-32B/snapshots/master"
output_dir="./distribute/qwen32b"
tokenizer=AutoTokenizer.from_pretrained(
 path,
 trust_remote_code=True
)


# 2 修改模型加载方式
from compressed_tensors.offload import load_offloaded_model
with load_offloaded_model():

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto_offload",
        offload_folder=offload_dir,
        trust_remote_code=True,
    )


#构造数据集 用于寻找激活值的分布
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import Dataset

texts = [
 "人工智能正在改变现代工业。",
 "大型语言模型需要大量计算资源。",
 "Transformer架构是目前主流的大模型架构。",
 "模型压缩可以降低推理成本。",
]*100


dataset = Dataset.from_dict(
 {
 "text": texts
 }
)

print(len(dataset))



num_samples = len(dataset)

indices = list(range(num_samples))


# 每个rank拿自己的数据
indices = indices[rank::world_size]
dataset = dataset.select(indices)


print(
    "rank:",
    rank,
    "world_size:",
    world_size,
    "dataset size:",
    len(dataset)
)


recipe=[

AWQModifier(
    duo_scaling="both",
    n_grid=20,
),

QuantizationModifier(
    targets=["Linear"],
    ignore=[
        "lm_head",
        "embed_tokens"
    ],
    scheme="W4A16_ASYM",
)

]

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=128,
    output_dir=output_dir,
)

tokenizer.save_pretrained(
    output_dir
)
```

------



```python
#使用GPTQ进行量化
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier


# ==========================
# 模型
# ==========================

model_path = "./models/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)


# ==========================
# GPTQ recipe
# ==========================

recipe = [

    GPTQModifier(
        block_size=128,
    ),


    QuantizationModifier(
        targets=[
            "Linear"
        ],

        scheme="W4A16_ASYM",

        ignore=[
            "lm_head"
        ],
    ),
]


# ==========================
# calibration dataset
# ==========================

from datasets import Dataset


texts = [
    "人工智能正在改变世界。",
    "Transformer是大语言模型的核心结构。",
    "大型语言模型需要大量计算资源。",
    "模型压缩可以降低部署成本。",
]*32


dataset = Dataset.from_dict(
    {
        "text":texts
    }
)


# ==========================
# GPTQ量化
# ==========================

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,

    recipe=recipe,

    max_seq_length=512,

    num_calibration_samples=128,

    output_dir="./Qwen2-7B-GPTQ-W4A16"
)


tokenizer.save_pretrained(
    "./Qwen2-7B-GPTQ-W4A16"
)
```

```python
#使用oneshot分布式的GPTQ
#每个rank可以量化成功
#！！！！在两个rank之间通信的时候 更新参数的时候挂掉

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

model_path="/root/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"

import os
from compressed_tensors.offload import init_dist
init_dist()

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
offload_dir=f"./offload_cache/rank_{rank}"

os.makedirs(
    offload_dir,
    exist_ok=True
)


import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier



tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


from compressed_tensors.offload import load_offloaded_model
with load_offloaded_model():

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=offload_dir,
        trust_remote_code=True,
    )




recipe = [

    GPTQModifier(
        block_size=128,
    ),


    QuantizationModifier(
        targets=[
            "Linear"
        ],

        scheme="W4A16_ASYM",

        ignore=[
            "lm_head"
        ],
    ),
]



from datasets import Dataset


texts = [
    "人工智能正在改变世界。",
    "Transformer是大语言模型的核心结构。",
    "大型语言模型需要大量计算资源。",
    "模型压缩可以降低部署成本。",
]*32


dataset = Dataset.from_dict(
    {
        "text":texts
    }
)


num_samples = len(dataset)

indices = list(range(num_samples))


# 每个rank拿自己的数据
indices = indices[rank::world_size]
dataset = dataset.select(indices)


oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,

    recipe=recipe,

    max_seq_length=512,

    num_calibration_samples=len(dataset),

    output_dir="./Qwen2-7B-GPTQ-W4A16_dis"
)


tokenizer.save_pretrained(
    "./Qwen2-7B-GPTQ-W4A16_dis"
)




```

```
torchrun --nproc_per_node=2 xx.py
```



------



##### 使用无数据 PTQ (data-free PTQ) 将 Qwen3.5-27B 视觉语言模型量化为 NVFP4A16

##### blackwall架构

```python
#1 加载模型

from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model.
MODEL_ID = "Qwen/Qwen3.5-27B"

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "36GiB",
        1: "36GiB",
        "cpu": "120GiB",
    },
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
```

```python
#2 配置量化方案

# No need to ignore mtp layers as they are not loaded
# through Qwen3_5ForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16", #scheme="W4A16_ASYM", 如果不是blackwall架构可以用这个测试
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
    ],
)

```

```python
#3 应用量化 如果GPU显存不足，会自动使用cpu缓存技术
oneshot(
    model=model,
    recipe=recipe,
    output_dir=output_dir,
)


```

```python
#测试量化后的模型能不能正常使用
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

quant_model_path = "autodl-tmp/models/vis_ptq"

processor = AutoProcessor.from_pretrained(
    quant_model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    quant_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()


print("\n\n========== SAMPLE GENERATION ==============")

messages = [
    {
        "role": "user",
        "content": "Hello my name is"
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = processor(
    text=prompt,
    return_tensors="pt"
)

# 放到模型所在设备
inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}


with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )


print(
    processor.decode(
        output[0],
        skip_special_tokens=True
    )
)

print("==========================================")

```



##### 使用PTQ(带校准数据) 将 Qwen3.6-35B-A3B 稀疏 MoE 模型量化为 NVFP4（权重和激活均量化为 FP4）

```python
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: This example requires transformers >= v5

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)



# No need to ignore mtp layers as they are not loaded
# through Qwen3_5MoeForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)



NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.select_columns(["messages"])
ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": MAX_SEQUENCE_LENGTH,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
    output_dir=output_dir,
)

tokenizer.save_pretrained(
 output_dir
)

```



##### 将 Qwen3.5-27B-A10B 稀疏 MoE 模型量化为 NVFP4（权重和激活均量化为 FP4）。

```python
#1 加载模型
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

```python
#2 加载校准数据集
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.select_columns(["messages"])
ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}
```

```python
#3 配置量化方案
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)
```

```python
#应用量化

oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
    output_dir="./qwen_27b_fpv4"
)
```

------

##### 加入均方权重重要性量化，对重要的权重进行保护 ------- iMatrix重要性加权量化

```python
from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor.modifiers.quantization import QuantizationModifier

scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]
```

##### 量化 input embedding

```

recipe = [

    GPTQModifier(
        block_size=128,
        config_groups={
            "group_0": scheme
        },
        ignore=[
            "lm_head"
        ],
    ),


    QuantizationModifier(
        config_groups={

            "linear": {
                "targets": [
                    "Linear"
                ],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "group",
                    "group_size": 128,
                },
            },


            "embedding": {
                "targets": [
                    "Embedding"
                ],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 64,
                },
            },

        },

        ignore=[
            "lm_head"
        ],
    ),
]
```

##### REAP专家剪枝的MOE压缩

```
from llmcompressor.modifiers.pruning import REAPPruningModifier


recipe = [
    REAPPruningModifier(
        sparsity=0.25,#剪掉1/4的专家
    )
]
```

##### KV cache量化

```
from llmcompressor import oneshot

recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```



#### Kimi-K2.6 NVFP4 

```
原始的 Kimi K2.6 检查点 以量化格式发布，使用 4 位整数权重。为了创建可以利用 NVIDIA 4 位浮点内核的 NVFP4 检查点，我们必须首先反量化到全精度 (bfloat16)，然后量化到所需的 NVFP4 格式。请注意，这需要将全精度模型保存到中间目录。让我们来看看量化过程的主要步骤：1. 模型反量化 2. 对全精度检查点应用量化
```

```python
from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    convert_checkpoint,
)

MODEL_ID = "moonshotai/Kimi-K2.6"
DEQUANTIZED_SAVE_DIR = "Kimi-K2.6-bf16"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*self_attn.*",
    "re:.*embed_tokens$",
    # ignore anything not in language_model
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

# Convert to dense bfloat16 format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=DEQUANTIZED_SAVE_DIR,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=4,
)
```

```
一旦反量化，模型可以通过 oneshot 量化为 NVFP4。NVFP4 使用静态激活量化，因此 oneshot 需要校准数据集。由于模型具有一万亿个参数，我们利用带有磁盘卸载的 compressed_tensors.offload 模块来通过模型运行校准数据集。以下代码片段已在单个
```

```python
from compressed_tensors.offload import load_offloaded_model
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SAVE_DIR = "Kimi-K2.6-NVFP4"

# Quantize bfloat16 checkpoint to NVFP4, limiting CPU RAM usage to 500GB
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        DEQUANTIZED_SAVE_DIR,
        device_map="auto_offload",
        max_memory={"cpu": 500e9},
        trust_remote_code=True,
        offload_folder="./offload_folder",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        DEQUANTIZED_SAVE_DIR, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        DEQUANTIZED_SAVE_DIR, trust_remote_code=True
    )

# Select calibration dataset.
DATASET_ID = "ultrachat-200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 20 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=ignore,
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

##### Kimi-K2.6 FP8 块示例 无数据校验

```
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K2.6"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*kv_a_proj_with_mqa$",
    "re:.*q_a_proj$",
    "re:.*vision_tower.*",
    "re:.*embed_tokens$",
    # ignore anything not in language_model
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=2,
    device="cuda:0",
)
```

------

##### Meta-Llama量化

```python
#FP8无数据量化

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

oneshot(
    model=model,
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="Meta-Llama-3-8B-Instruct-FP8",
)
```

```python
#GPTQ W4A16

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

oneshot(
    model=model,
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    num_calibration_samples=512,
    max_seq_length=2048,
    output_dir="Meta-Llama-3-8B-Instruct-W4A16-GPTQ",
)
```

```python
#具有全专家校准的 MoE 模型

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modeling.llama4 import SequentialLlama4TextMoe  # noqa: F401
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

oneshot(
    model=model,
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*lm_head",
            "re:.*self_attn",
            "re:.*router",
            "re:.*vision_model.*",
            "re:.*multi_modal_projector.*",
            "Llama4TextAttention",
        ],
    ),
    num_calibration_samples=20,
    max_seq_length=2048,
    moe_calibrate_all_experts=True,
    output_dir="Llama-4-Scout-17B-NVFP4",
)
```

------



##### 添加新的 Observer

```

```

##### 添加新的Modifier

```

```

