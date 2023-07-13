<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Example Zoo

Below contains a non-exhuastive list of tutorials and scripts showcasing 🤗 Accelerate

## Official Accelerate Examples:

### Basic Examples

These examples showcase the base features of Accelerate and are a great starting point

- [Barebones NLP example](https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py)
- [Barebones distributed NLP example in a Jupyter Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)
- [Barebones computer vision example](https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py)
- [Barebones distributed computer vision example in a Jupyter Notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb)
- [Using Accelerate in Kaggle](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate)

### Feature Specific Examples

These examples showcase specific features that the Accelerate framework offers

- [Automatic memory-aware gradient accumulation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/automatic_gradient_accumulation.py)
- [Checkpointing states](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py)
- [Cross validation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/cross_validation.py)
- [DeepSpeed](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py)
- [Fully Sharded Data Parallelism](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/fsdp_with_peak_mem_tracking.py)
- [Gradient accumulation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation.py)
- [Memory-aware batch size finder](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/memory.py)
- [Metric Computation](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py)
- [Using Trackers](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/tracking.py)
- [Using Megatron-LM](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/megatron_lm_gpt_pretraining.py)

### Full Examples 

These examples showcase every feature in Accelerate at once that was shown in "Feature Specific Examples"

- [Complete NLP example](https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py)
- [Complete computer vision example](https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py)
- [Very complete and extensible vision example showcasing SLURM, hydra, and a very extensible usage of the framework](https://github.com/yuvalkirstain/PickScore)
- [Causal language model fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py)
- [Masked language model fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py)
- [Speech pretraining example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py)
- [Translation fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py)
- [Text classification fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py)
- [Semantic segmentation fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/semantic-segmentation/run_semantic_segmentation_no_trainer.py)
- [Question answering fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py)
- [Beam search question answering fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py)
- [Multiple choice question answering fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py)
- [Named entity recognition fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py)
- [Image classification fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification_no_trainer.py)
- [Summarization fine-tuning example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py)
- [End-to-end examples on how to use AWS SageMaker integration of Accelerate](https://github.com/huggingface/notebooks/blob/main/sagemaker/22_accelerate_sagemaker_examples/README.md)
- [Megatron-LM examples for various NLp tasks](https://github.com/pacman100/accelerate-megatron-test) 

## Integration Examples 

These are tutorials from libraries that integrate with 🤗 Accelerate: 

> Don't find your integration here? Make a PR to include it!

### Catalyst

- [Distributed training tutorial with Catalyst](https://catalyst-team.github.io/catalyst/tutorials/ddp.html)

### DALLE2-pytorch 

- [Fine-tuning DALLE2](https://github.com/lucidrains/DALLE2-pytorch#usage)

### 🤗 diffusers

- [Performing textual inversion with diffusers](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion)
- [Training DreamBooth with diffusers](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)

### fastai 

- [Distributed training from Jupyter Notebooks with fastai](https://docs.fast.ai/tutorial.distributed.html)
- [Basic distributed training examples with fastai](https://docs.fast.ai/examples/distributed_app_examples.html)

### GradsFlow

- [Auto Image Classification with GradsFlow](https://docs.gradsflow.com/en/latest/examples/nbs/01-ImageClassification/)

### imagen-pytorch 

- [Fine-tuning Imagen](https://github.com/lucidrains/imagen-pytorch#usage)

### Kornia

- [Fine-tuning vision models with Kornia's Trainer](https://kornia.readthedocs.io/en/latest/get-started/training.html)

### PyTorch Accelerated 

- [Quickstart distributed training tutorial with PyTorch Accelerated](https://pytorch-accelerated.readthedocs.io/en/latest/quickstart.html)

### PyTorch3D

- [Perform Deep Learning with 3D data](https://pytorch3d.org/tutorials/)

### Stable-Dreamfusion

- [Training with Stable-Dreamfusion to convert text to a 3D model](https://colab.research.google.com/drive/1MXT3yfOFvO0ooKEfiUUvTKwUkrrlCHpF?usp=sharing)

### Tez 

- [Leaf disease detection with Tez and Accelerate](https://www.kaggle.com/code/abhishek/tez-faster-and-easier-training-for-leaf-detection/notebook)

### trlx 

- [How to implement a sentiment learning task with trlx](https://github.com/CarperAI/trlx#example-how-to-add-a-task)

### Comfy-UI

- [Enabling using large Stable Diffusion Models in low-vram settings using Accelerate](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py#L291-L296)


## In Science

Below contains a non-exhaustive list of papers utilizing 🤗 Accelerate. 

> Don't find your paper here? Make a PR to include it!

* Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy: “Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation”, 2023; [arXiv:2305.01569](http://arxiv.org/abs/2305.01569).
* Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim: “Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models”, 2023; [arXiv:2305.04091](http://arxiv.org/abs/2305.04091).
* Arthur Câmara, Claudia Hauff: “Moving Stuff Around: A study on efficiency of moving documents into memory for Neural IR models”, 2022; [arXiv:2205.08343](http://arxiv.org/abs/2205.08343).
* Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang: “High-throughput Generative Inference of Large Language Models with a Single GPU”, 2023; [arXiv:2303.06865](http://arxiv.org/abs/2303.06865).
* Peter Melchior, Yan Liang, ChangHoon Hahn, Andy Goulding: “Autoencoding Galaxy Spectra I: Architecture”, 2022; [arXiv:2211.07890](http://arxiv.org/abs/2211.07890).
* Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, Diyi Yang: “A Cheaper and Better Diffusion Language Model with Soft-Masked Noise”, 2023; [arXiv:2304.04746](http://arxiv.org/abs/2304.04746).
* Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, Angjoo Kanazawa: “Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions”, 2023; [arXiv:2303.12789](http://arxiv.org/abs/2303.12789).
* Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, Andrea Vedaldi: “RealFusion: 360° Reconstruction of Any Object from a Single Image”, 2023; [arXiv:2302.10663](http://arxiv.org/abs/2302.10663).
* Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, Hongsheng Li: “Better Aligning Text-to-Image Models with Human Preference”, 2023; [arXiv:2303.14420](http://arxiv.org/abs/2303.14420).
* Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang: “HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace”, 2023; [arXiv:2303.17580](http://arxiv.org/abs/2303.17580).
* Yue Yang, Wenlin Yao, Hongming Zhang, Xiaoyang Wang, Dong Yu, Jianshu Chen: “Z-LaVI: Zero-Shot Language Solver Fueled by Visual Imagination”, 2022; [arXiv:2210.12261](http://arxiv.org/abs/2210.12261).
* Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho: “How to Backdoor Diffusion Models?”, 2022; [arXiv:2212.05400](http://arxiv.org/abs/2212.05400).
* Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, Seungryong Kim: “Let 2D Diffusion Model Know 3D-Consistency for Robust Text-to-3D Generation”, 2023; [arXiv:2303.07937](http://arxiv.org/abs/2303.07937).
* Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, Daniel Cohen-Or: “Localizing Object-level Shape Variations with Text-to-Image Diffusion Models”, 2023; [arXiv:2303.11306](http://arxiv.org/abs/2303.11306).
* Dídac Surís, Sachit Menon, Carl Vondrick: “ViperGPT: Visual Inference via Python Execution for Reasoning”, 2023; [arXiv:2303.08128](http://arxiv.org/abs/2303.08128).
* Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, Qifeng Chen: “FateZero: Fusing Attentions for Zero-shot Text-based Video Editing”, 2023; [arXiv:2303.09535](http://arxiv.org/abs/2303.09535).
* Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, Yejin Choi: “NaturalProver: Grounded Mathematical Proof Generation with Language Models”, 2022; [arXiv:2205.12910](http://arxiv.org/abs/2205.12910).
* Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or: “TEXTure: Text-Guided Texturing of 3D Shapes”, 2023; [arXiv:2302.01721](http://arxiv.org/abs/2302.01721).
* Puijin Cheng, Li Lin, Yijin Huang, Huaqing He, Wenhan Luo, Xiaoying Tang: “Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement”, 2023; [arXiv:2303.04603](http://arxiv.org/abs/2303.04603).
* Shun Shao, Yftah Ziser, Shay Cohen: “Erasure of Unaligned Attributes from Neural Representations”, 2023; [arXiv:2302.02997](http://arxiv.org/abs/2302.02997).
* Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, Minjoon Seo: “In-Context Instruction Learning”, 2023; [arXiv:2302.14691](http://arxiv.org/abs/2302.14691).
* Shikun Liu, Linxi Fan, Edward Johns, Zhiding Yu, Chaowei Xiao, Anima Anandkumar: “Prismer: A Vision-Language Model with An Ensemble of Experts”, 2023; [arXiv:2303.02506](http://arxiv.org/abs/2303.02506 ).
* Haoyu Chen, Zhihua Wang, Yang Yang, Qilin Sun, Kede Ma: “Learning a Deep Color Difference Metric for Photographic Images”, 2023; [arXiv:2303.14964](http://arxiv.org/abs/2303.14964).
* Van-Hoang Le, Hongyu Zhang: “Log Parsing with Prompt-based Few-shot Learning”, 2023; [arXiv:2302.07435](http://arxiv.org/abs/2302.07435).
* Keito Kudo, Yoichi Aoki, Tatsuki Kuribayashi, Ana Brassard, Masashi Yoshikawa, Keisuke Sakaguchi, Kentaro Inui: “Do Deep Neural Networks Capture Compositionality in Arithmetic Reasoning?”, 2023; [arXiv:2302.07866](http://arxiv.org/abs/2302.07866).
* Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, Prithviraj Ammanabrolu: “Behavior Cloned Transformers are Neurosymbolic Reasoners”, 2022; [arXiv:2210.07382](http://arxiv.org/abs/2210.07382).
* Martin Wessel, Tomáš Horych, Terry Ruas, Akiko Aizawa, Bela Gipp, Timo Spinde: “Introducing MBIB -- the first Media Bias Identification Benchmark Task and Dataset Collection”, 2023; [arXiv:2304.13148](http://arxiv.org/abs/2304.13148 ). DOI: [https://dx.doi.org/10.1145/3539618.3591882 10.1145/3539618.3591882].
* Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, Daniel Cohen-Or: “Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models”, 2023; [arXiv:2301.13826](http://arxiv.org/abs/2301.13826).
* Marcio Fonseca, Yftah Ziser, Shay B. Cohen: “Factorizing Content and Budget Decisions in Abstractive Summarization of Long Documents”, 2022; [arXiv:2205.12486](http://arxiv.org/abs/2205.12486).
* Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, Daniel Cohen-Or: “TEXTure: Text-Guided Texturing of 3D Shapes”, 2023; [arXiv:2302.01721](http://arxiv.org/abs/2302.01721).
* Tianxing He, Jingyu Zhang, Tianle Wang, Sachin Kumar, Kyunghyun Cho, James Glass, Yulia Tsvetkov: “On the Blind Spots of Model-Based Evaluation Metrics for Text Generation”, 2022; [arXiv:2212.10020](http://arxiv.org/abs/2212.10020).
* Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham: “In-Context Retrieval-Augmented Language Models”, 2023; [arXiv:2302.00083](http://arxiv.org/abs/2302.00083).
* Dacheng Li, Rulin Shao, Hongyi Wang, Han Guo, Eric P. Xing, Hao Zhang: “MPCFormer: fast, performant and private Transformer inference with MPC”, 2022; [arXiv:2211.01452](http://arxiv.org/abs/2211.01452).
* Baolin Peng, Michel Galley, Pengcheng He, Chris Brockett, Lars Liden, Elnaz Nouri, Zhou Yu, Bill Dolan, Jianfeng Gao: “GODEL: Large-Scale Pre-Training for Goal-Directed Dialog”, 2022; [arXiv:2206.11309](http://arxiv.org/abs/2206.11309).
* Egil Rønningstad, Erik Velldal, Lilja Øvrelid: “Entity-Level Sentiment Analysis (ELSA): An exploratory task survey”, 2023, Proceedings of the 29th International Conference on Computational Linguistics, 2022, pages 6773-6783; [arXiv:2304.14241](http://arxiv.org/abs/2304.14241).
* Charlie Snell, Ilya Kostrikov, Yi Su, Mengjiao Yang, Sergey Levine: “Offline RL for Natural Language Generation with Implicit Language Q Learning”, 2022; [arXiv:2206.11871](http://arxiv.org/abs/2206.11871).
* Zhiruo Wang, Shuyan Zhou, Daniel Fried, Graham Neubig: “Execution-Based Evaluation for Open-Domain Code Generation”, 2022; [arXiv:2212.10481](http://arxiv.org/abs/2212.10481).
* Minh-Long Luu, Zeyi Huang, Eric P. Xing, Yong Jae Lee, Haohan Wang: “Expeditious Saliency-guided Mix-up through Random Gradient Thresholding”, 2022; [arXiv:2212.04875](http://arxiv.org/abs/2212.04875).
* Jun Hao Liew, Hanshu Yan, Daquan Zhou, Jiashi Feng: “MagicMix: Semantic Mixing with Diffusion Models”, 2022; [arXiv:2210.16056](http://arxiv.org/abs/2210.16056).
* Yaqing Wang, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, Jianfeng Gao: “LiST: Lite Prompted Self-training Makes Parameter-Efficient Few-shot Learners”, 2021; [arXiv:2110.06274](http://arxiv.org/abs/2110.06274).