<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Amazon SageMaker

Hugging Face and Amazon introduced new [Hugging Face Deep Learning Containers (DLCs)](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers) to
make it easier than ever to train Hugging Face Transformer models in [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

## Getting Started

### Setup & Installation


Before you can run your 🤗 Accelerate scripts on Amazon SageMaker you need to sign up for an AWS account. If you do not
have an AWS account yet learn more [here](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html).

After you have your AWS Account you need to install the `sagemaker` sdk for 🤗 Accelerate with:

```bash
pip install "accelerate[sagemaker]" --upgrade
```

🤗 Accelerate currently uses the 🤗 DLCs, with `transformers`, `datasets` and `tokenizers` pre-installed. 🤗
Accelerate is not in the DLC yet (will soon be added!) so to use it within Amazon SageMaker you need to create a
`requirements.txt` in the same directory where your training script is located and add it as dependency:

```
accelerate
```

You should also add any other dependencies you have to this `requirements.txt`.


### Configure 🤗 Accelerate

You can configure the launch configuration for Amazon SageMaker the same as you do for non SageMaker training jobs with
the 🤗 Accelerate CLI:

```bash
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 1
```

🤗 Accelerate will go through a questionnaire about your Amazon SageMaker setup and create a config file you can edit.

<Tip>

    🤗 Accelerate is not saving any of your credentials.

</Tip>

### Prepare a 🤗 Accelerate fine-tuning script

The training script is very similar to a training script you might run outside of SageMaker, but to save your model
after training you need to specify either `/opt/ml/model` or use `os.environ["SM_MODEL_DIR"]` as your save
directory. After training, artifacts in this directory are uploaded to S3:


```diff
- torch.save('/opt/ml/model`)
+ accelerator.save('/opt/ml/model')
```

<Tip warning={true}>

    SageMaker doesn’t support argparse actions. If you want to use, for example, boolean hyperparameters, you need to
    specify type as bool in your script and provide an explicit True or False value for this hyperparameter. [[REF]](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#prepare-a-pytorch-training-script).

</Tip>

### Launch Training

You can launch your training with 🤗 Accelerate CLI with:

```
accelerate launch path_to_script.py --args_to_the_script
```

This will launch your training script using your configuration. The only thing you have to do is provide all the
arguments needed by your training script as named arguments.

**Examples**

<Tip>

    If you run one of the example scripts, don't forget to add `accelerator.save('/opt/ml/model')` to it.

</Tip>

```bash
accelerate launch ./examples/sagemaker_example.py
```

Outputs:

```
Configuring Amazon SageMaker environment
Converting Arguments to Hyperparameters
Creating Estimator
2021-04-08 11:56:50 Starting - Starting the training job...
2021-04-08 11:57:13 Starting - Launching requested ML instancesProfilerReport-1617883008: InProgress
.........
2021-04-08 11:58:54 Starting - Preparing the instances for training.........
2021-04-08 12:00:24 Downloading - Downloading input data
2021-04-08 12:00:24 Training - Downloading the training image..................
2021-04-08 12:03:39 Training - Training image download completed. Training in progress..
........
epoch 0: {'accuracy': 0.7598039215686274, 'f1': 0.8178438661710037}
epoch 1: {'accuracy': 0.8357843137254902, 'f1': 0.882249560632689}
epoch 2: {'accuracy': 0.8406862745098039, 'f1': 0.8869565217391304}
........
2021-04-08 12:05:40 Uploading - Uploading generated training model
2021-04-08 12:05:40 Completed - Training job completed
Training seconds: 331
Billable seconds: 331
You can find your model data at: s3://your-bucket/accelerate-sagemaker-1-2021-04-08-11-56-47-108/output/model.tar.gz
```

## Advanced Features

### Distributed Training: Data Parallelism

Set up the accelerate config by running `accelerate config` and answer the SageMaker questions and set it up.
To use SageMaker DDP, select it when asked 
`What is the distributed mode? ([0] No distributed training, [1] data parallelism):`.
Example config below:
```yaml
base_job_name: accelerate-sagemaker-1
compute_environment: AMAZON_SAGEMAKER
distributed_type: DATA_PARALLEL
ec2_instance_type: ml.p3.16xlarge
iam_role_name: xxxxx
image_uri: null
mixed_precision: fp16
num_machines: 1
profile: xxxxx
py_version: py38
pytorch_version: 1.10.2
region: us-east-1
transformers_version: 4.17.0
use_cpu: false
```

### Distributed Training: Model Parallelism

*currently in development, will be supported soon.*

### Python packages and dependencies

🤗 Accelerate currently uses the 🤗 DLCs, with `transformers`, `datasets` and `tokenizers` pre-installed. If you
want to use different/other Python packages you can do this by adding them to the `requirements.txt`. These packages
will be installed before your training script is started.

### Local Training: SageMaker Local mode

The local mode in the SageMaker SDK allows you to run your training script locally inside the HuggingFace DLC (Deep Learning container) 
or using your custom container image. This is useful for debugging and testing your training script inside the final container environment.
Local mode uses Docker compose (*Note: Docker Compose V2 is not supported yet*). The SDK will handle the authentication against ECR
to pull the DLC to your local environment. You can emulate CPU (single and multi-instance) and GPU (single instance) SageMaker training jobs.

To use local mode, you need to set your `ec2_instance_type` to `local`.

```yaml
ec2_instance_type: local
```

### Advanced configuration

The configuration allows you to override parameters for the [Estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html).
These settings have to be applied in the config file and are not part of `accelerate config`. You can control many additional aspects of the training job, e.g. use Spot instances, enable network isolation and many more.

```yaml
additional_args:
  # enable network isolation to restrict internet access for containers
  enable_network_isolation: True
```

You can find all available configuration [here](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html).

### Use Spot Instances

You can use Spot Instances e.g. using (see [Advanced configuration](#advanced-configuration)):
```yaml
additional_args:
  use_spot_instances: True
  max_wait: 86400
```

*Note: Spot Instances are subject to be terminated and training to be continued from a checkpoint. This is not handled in 🤗 Accelerate out of the box. Contact us if you would like this feature.*

### Remote scripts: Use scripts located on Github

*undecided if feature is needed. Contact us if you would like this feature.*