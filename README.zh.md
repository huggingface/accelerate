<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/accelerate/main/docs/source/imgs/accelerate_logo.png" width="400"/>
    <br>
<p>

<p align="center">
    <a href="https://github.com/huggingface/accelerate/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/accelerate.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/accelerate/index.html"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/accelerate/index.html.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/accelerate/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/accelerate.svg"></a>
    <a href="https://github.com/huggingface/accelerate/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<h3 align="center">
<p>在任意类型的计算设备上运行*原生* PyTorch 训练脚本</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://raw.githubusercontent.com/huggingface/accelerate/main/docs/source/imgs/course_banner.png"></a>
</h3>

---

## ⚡ 极简集成 (Easy to integrate)

🤗 Accelerate 是专为喜爱原生编写 PyTorch 模型训练循环、但又不愿维护多 GPU / TPU / 混合精度繁琐样板代码的开发者打造的。

🤗 Accelerate 仅对多 GPU、TPU 以及 FP16/BF16/FP8 相关的分布式底层样板代码进行精准抽象，其余所有代码完全保持不变。

示例如下：

```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

+ accelerator = Accelerator()
- device = 'cpu'
+ device = accelerator.device

  model = torch.nn.Transformer().to(device)
  optimizer = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
```

如上所示，只需向任何标准 PyTorch 训练脚本中加入 **5 行代码**，即可在单 CPU、单 GPU、多机多 GPU 以及 Google TPU 等任意单机或分布式集群节点上运行，并无缝支持多种混合精度模式（FP8、FP16、BF16）。

更关键的是：同一套代码无需任何改动，既能在本机快速单步调试，也能无缝部署到超大规模云端集群进行全量训练。

🤗 Accelerate 甚至可以自动帮您托管设备分配（Device Placement），进一步消除样板代码，让训练代码更加精简清爽：

```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

- device = 'cpu'
+ accelerator = Accelerator()

- model = torch.nn.Transformer().to(device)
+ model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
-         source = source.to(device)
-         targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
```

想要深入了解？欢迎查阅 [官方技术文档](https://huggingface.co/docs/accelerate) 或浏览 [示例代码库 (examples)](https://github.com/huggingface/accelerate/tree/main/examples)。

---

## 🚀 启动训练脚本 (Launching script)

🤗 Accelerate 提供了一个强大的可选命令行工具（CLI），让您在启动训练前能够快速配置并测试训练集群环境。无需记忆复杂的 `torch.distributed.run` 参数，也不用单独为 TPU 编写专用启动器！

只需在您的机器或集群上执行：

```bash
accelerate config
```

根据终端交互提示回答几个简明问题，系统将自动生成全局配置文件，并在后续执行以下命令时自动加载默认选项：

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，在仓库根目录下运行 MRPC 任务的 GLUE 范例：

```bash
accelerate launch examples/nlp_example.py
```

该 CLI 工具是**完全可选的**，您依然可以自由使用原生的 `python my_script.py` 或 `python -m torchrun my_script.py`。

如果您不想交互式运行 `accelerate config`，也可以直接向 `accelerate launch` 透传所有 `torchrun` 支持的参数。例如在两块 GPU 上启动：

```bash
accelerate launch --multi_gpu --num_processes 2 examples/nlp_example.py
```

更多进阶用法，请参阅 [CLI 命令参考文档](https://huggingface.co/docs/accelerate/package_reference/cli) 或查阅 [配置模板中心 (Config Zoo)](https://github.com/huggingface/accelerate/blob/main/examples/config_yaml_templates/)。

---

## 🖥️ 基于 MPI 启动多 CPU 分布式训练

🤗 Accelerate 支持基于 MPI 的多 CPU 训练模式。关于如何安装与构建 Open MPI，可查阅 [官方指南](https://www.open-mpi.org/faq/?category=building#easy-build)；您同样可以使用 Intel MPI 或 MVAPICH。

在集群上完成 MPI 安装后，运行：
```bash
accelerate config
```
在交互式提示中选择使用多 CPU（multi-CPU），并在询问是否使用 accelerate 启动 mpirun 时选择 "yes"。

随后直接运行您的训练脚本：
```bash
accelerate launch examples/nlp_example.py
```

您也可以直接通过底层 `mpirun` 命令免 CLI 启动：
```bash
mpirun -np 2 python examples/nlp_example.py
```

---

## 🔥 搭配 DeepSpeed 进行大规模训练

🤗 Accelerate 原生支持通过 DeepSpeed 在单卡或多卡 GPU 上加速训练。使用时无需修改训练代码业务逻辑，直接通过 `accelerate config` 开启即可。如果希望直接在 Python 脚本中自定义 DeepSpeed 的进阶参数，可通过 `DeepSpeedPlugin` 进行精细化配置：

```python
from accelerate import Accelerator, DeepSpeedPlugin

# DeepSpeed 需要提前获取梯度累积步数，请务必传入
# 注意：代码中仍需保留与普通训练一致的梯度累积逻辑
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)

# 如何安全保存 🤗 Transformer 模型权重？
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    save_dir, 
    save_function=accelerator.save, 
    state_dict=accelerator.get_state_dict(model)
)
```

> **提示**：如在 DeepSpeed 实验性特性中遇到任何边缘异常，欢迎提交 GitHub Issue 共同交流！

---

## 📓 在 Jupyter Notebook 中启动训练

🤗 Accelerate 还提供了 `notebook_launcher` 函数，可在 Notebook 环境中直接启动分布式多卡/多核训练。对于使用 TPU 运行的 Google Colab 或 Kaggle 环境尤为便捷。只需将训练流程封装在 `training_function` 中，并在末尾单元格调用：

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

实战案例可参阅官方 [Jupyter 示例笔记本](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)。 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)

---

## ❓ 为什么选择 🤗 Accelerate？

如果您希望将训练脚本轻松运行在各种异构或大规模分布式环境中，同时**绝不剥夺对底层训练循环（Training Loop）的完全控制权**，那么 🤗 Accelerate 是最理想的选择。它不是 PyTorch 之上的重型高层框架，而是一层极薄的封装（Thin Wrapper），让您无需额外学习复杂的全新 DSL 抽象。事实上，🤗 Accelerate 的全部核心功能都浓缩在单一的 `Accelerator` 类中！

## 🚫 什么时候不推荐使用 🤗 Accelerate？

如果您不想自己手写训练循环代码，那么不推荐使用 🤗 Accelerate。PyTorch 生态中有许多出色的高层封装库（例如 Hugging Face `Trainer`、fastai 等）为您托管完整的训练生命周期，而 🤗 Accelerate 的定位是底层轻量桥梁。

---

## 🌐 基于 🤗 Accelerate 构建的代表性生态项目

如果您喜欢 🤗 Accelerate 的极简理念，但在某些场景下希望使用更高层次的抽象封装，以下基于 🤗 Accelerate 构建的知名开源项目值得探索：

* **[Amphion](https://github.com/open-mmlab/Amphion)**：顶尖音频、音乐与语音生成工具包，致力于支持可复现的研究与全模态语音生成。
* **[Animus](https://github.com/Scitator/animus)**：极简机器学习实验运行框架，统一实验核心生命周期接口。
* **[Catalyst](https://github.com/catalyst-team/catalyst#getting-started)**：深度学习研发与高可复现实验框架，聚焦工程解耦与代码复用。
* **[fastai](https://github.com/fastai/fastai#installing)**：基于现代最佳实践的深度学习框架，极大简化高性能神经网络训练。
* **[Finetuner](https://github.com/jina-ai/finetuner)**：Jina AI 推出的企业级微调套件，专为语义搜索与跨模态多任务微调高品质 Embedding 向量。
* **[InvokeAI](https://github.com/invoke-ai/InvokeAI)**：业界领先的 Stable Diffusion 创作引擎与商业级图像生成前端。
* **[Kornia](https://kornia.readthedocs.io/en/latest/get-started/introduction.html)**：可微分计算机视觉库，将传统视觉算子与深度学习网络无缝集成。
* **[Open Assistant](https://projects.laion.ai/Open-Assistant/)**：开源会话智能体平台，具备强大的任务理解与外部工具调用能力。
* **[pytorch-accelerated](https://github.com/Chris-hughes10/pytorch-accelerated)**：极度强调透明度与简洁性的轻量级 PyTorch 训练库。
* **[Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**：全球最流行的开源 Gradio 绘图客户端。
* **[torchkeras](https://github.com/lyhue1991/torchkeras)**：Keras 风格的 PyTorch 训练工具，自带动态训练曲线可视化。
* **[transformers](https://github.com/huggingface/transformers)**：全球最具影响力的开源大模型基准库（Accelerate 为其 PyTorch 侧的核心分布式基础设施）。

---

## 📦 安装说明 (Installation)

本代码库在 **Python 3.8+** 与 **PyTorch 1.10.0+** 环境下经过全量自动化 CI 测试。

强烈建议在 [Python 虚拟环境](https://docs.python.org/3/library/venv.html) 中安装 🤗 Accelerate：

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 用户请执行: .venv\Scripts\activate
```

首先参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/#start-locally) 安装适配您硬件架构的 PyTorch，随后通过 pip 安装 🤗 Accelerate：

```bash
pip install accelerate
```

---

## 🧩 支持的集成特性矩阵 (Supported integrations)

- 仅单 CPU 运行
- 单节点多 CPU（多进程并发）
- 跨节点集群多 CPU 协同计算
- 单物理 GPU
- 单节点多物理 GPU
- 跨节点多物理 GPU 算力集群
- Google TPU 芯片
- FP16 / BFloat16 混合精度加速
- 基于 [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 或 [MS-AMP](https://github.com/Azure/MS-AMP/) 的最新 FP8 超低精度混合训练
- DeepSpeed 全系加速特性（ZeRO-1/2/3 等）
- PyTorch 原生 Fully Sharded Data Parallel (FSDP) 完全分片数据并行
- Megatron-LM 超大规模模型并行支持

---

## 📑 论文与学术引用 (Citing 🤗 Accelerate)

如果您在科研、学术成果或商业产品中使用了 🤗 Accelerate，请引用以下 BibTeX 条目：

```bibtex
@Misc{accelerate,
  title =        {Accelerate: Training and inference at scale made simple, efficient and adaptable.},
  author =       {Sylvain Gugger and Lysandre Debut and Thomas Wolf and Philipp Schmid and Zachary Mueller and Sourab Mangrulkar and Marc Sun and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/accelerate}},
  year =         {2022}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月2日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
