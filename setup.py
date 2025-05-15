# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


extras = {}
extras["quality"] = [
    "black ~= 23.1",  # hf-doc-builder has a hidden dependency on `black`
    "hf-doc-builder >= 0.3.0",
    "ruff ~= 0.11.2",
]
extras["docs"] = []
extras["test_prod"] = ["pytest>=7.2.0,<=8.0.0", "pytest-xdist", "pytest-subtests", "parameterized", "pytest-order"]
extras["test_dev"] = [
    "datasets",
    "diffusers",
    "evaluate",
    "torchdata>=0.8.0",
    "torchpippy>=0.2.0",
    "transformers",
    "scipy",
    "scikit-learn",
    "tqdm",
    "bitsandbytes",
    "timm",
]
extras["testing"] = extras["test_prod"] + extras["test_dev"]
extras["deepspeed"] = ["deepspeed"]
extras["rich"] = ["rich"]

extras["test_fp8"] = ["torchao"]  # note: TE for now needs to be done via pulling down the docker image directly
extras["test_trackers"] = ["wandb", "comet-ml", "tensorboard", "dvclive", "mlflow", "matplotlib"]
extras["dev"] = extras["quality"] + extras["testing"] + extras["rich"]

extras["sagemaker"] = [
    "sagemaker",  # boto3 is a required package in sagemaker
]

setup(
    name="accelerate",
    version="1.7.0",
    description="Accelerate",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="zach.mueller@huggingface.co",
    url="https://github.com/huggingface/accelerate",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "console_scripts": [
            "accelerate=accelerate.commands.accelerate_cli:main",
            "accelerate-config=accelerate.commands.config:main",
            "accelerate-estimate-memory=accelerate.commands.estimate:main",
            "accelerate-launch=accelerate.commands.launch:main",
            "accelerate-merge-weights=accelerate.commands.merge:main",
        ]
    },
    python_requires=">=3.9.0",
    install_requires=[
        "numpy>=1.17,<3.0.0",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=2.0.0",
        "huggingface_hub>=0.21.0",
        "safetensors>=0.4.3",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Release checklist
# 1. Checkout the release branch (for a patch the current release branch, for a new minor version, create one):
#      git checkout -b vXX.xx-release
#    The -b is only necessary for creation (so remove it when doing a patch)
# 2. Change the version in __init__.py and setup.py to the proper value.
# 3. Commit these changes with the message: "Release: v<VERSION>"
# 4. Add a tag in git to mark the release:
#      git tag v<VERSION> -m 'Adds tag v<VERSION> for pypi'
#    Push the tag and release commit to git: git push --tags origin vXX.xx-release
# 5. Run the following commands in the top-level directory:
#      make prepare_release
# 6. Upload the package to the pypi test server first:
#      make target=testpypi upload_release
# 7. Check that you can install it in a virtualenv by running:
#      make install_test_release
#      accelerate env
#      accelerate test
# 8. Upload the final version to actual pypi:
#      make target=pypi upload_release
# 9. Add release notes to the tag in github once everything is looking hunky-dory.
# 10. Go back to the main branch and update the version in __init__.py, setup.py to the new version ".dev" and push to
#     main.
