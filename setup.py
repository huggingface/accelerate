from setuptools import setup
from setuptools import find_packages

extras = {}
extras["quality"] = ["black >= 20.8b1", "isort >= 5.5.4", "flake8 >= 3.8.3"]
extras["docs"] = ["recommonmark", "sphinx==3.2.1", "sphinx-markdown-tables", "sphinx-rtd-theme==0.4.3", "sphinx-copybutton"]

setup(
    name="accelerate",
    version="0.0.1",
    description="Accelerate",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="sylvain@huggingface.co",
    url="https://github.com/huggingface/accelerate",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": [
        "accelerate=accelerate.commands.accelerate_cli:main",
        "accelerate-config=accelerate.commands.config:main",
        "accelerate-launch=accelerate.commands.launch:main",
    ]},
    python_requires=">=3.6.0",
    install_requires=["torch"],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
