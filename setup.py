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
    "ruff ~= 0.2.1",
]
extras["docs"] = []
extras["test_prod"] = ["pytest>=7.2.0,<=8.0.0", "pytest-xdist", "pytest-subtests", "parameterized"]
extras["test_dev"] = [
    "datasets",
    "evaluate",
    "torchpippy>=0.2.0",
    "transformers",
    "scipy",
    "scikit-learn",
    "deepspeed<0.13.0",
    "tqdm",
    "bitsandbytes",
    "timm",
]
extras["testing"] = extras["test_prod"] + extras["test_dev"]
extras["rich"] = ["rich"]

extras["test_trackers"] = ["wandb", "comet-ml", "tensorboard", "dvclive"]
extras["dev"] = extras["quality"] + extras["testing"] + extras["rich"]

extras["sagemaker"] = [
    "sagemaker",  # boto3 is a required package in sagemaker
]

setup(
    name="accelerate",
    version="0.28.0",
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
        ]
    },
    python_requires=">=3.8.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.10.0",
        "huggingface_hub",
        "safetensors>=0.3.1",
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
        "Programming Language :: Python :: 3.8",
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
#      rm -rf dist
#      rm -rf build
#      python setup.py bdist_wheel
#      python setup.py sdist
# 6. Upload the package to the pypi test server first:
#      twine upload dist/* -r testpypi
# 7. Check that you can install it in a virtualenv by running:
#      pip install accelerate
#      pip uninstall accelerate
#      pip install -i https://testpypi.python.org/pypi accelerate
#      accelerate env
#      accelerate test
# 8. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 9. Add release notes to the tag in github once everything is looking hunky-dory.
# 10. Go back to the main branch and update the version in __init__.py, setup.py to the new version ".dev" and push to
#     main.
