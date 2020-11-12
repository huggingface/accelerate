"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py. Remove the master from the links in
   the new models of the README:
   (https://huggingface.co/transformers/master/model_doc/ -> https://huggingface.co/transformers/model_doc/)
   then run `make fix-copies` to fix the index of the documentation.

2. Unpin specific versions from setup.py that use a git install.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

8. Add the release version to docs/source/_static/js/custom.js and .circleci/deploy.sh

9. Update README.md to redirect to correct documentation.
"""

from setuptools import setup
from setuptools import find_packages

extra = {}
extras["quality"] = ["black >= 20.8b1", "isort >= 5.5.4", "flake8 >= 3.8.3"]

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
    python_requires=">=3.6.0",
    install_requires=["torch"],
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
