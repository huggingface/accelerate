<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# How to contribute to 🤗 Accelerate?

Everyone is welcome to contribute, and we value everybody's contribution. Code
is thus not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply star the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/accelerate/blob/main/CODE_OF_CONDUCT.md).

## You can contribute in so many ways!

Some of the ways you can contribute to Accelerate:
* Fixing outstanding issues with the existing code;
* Contributing to the examples or to the documentation;
* Submitting issues related to bugs or desired new features.

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The 🤗 Accelerate library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

Did not find it? :( So we can act quickly on it, please follow these steps:

* Include your **OS type and version**, the versions of **Python** and **PyTorch**.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s;
* Provide the with your Accelerate configuration (located by default in `~/.cache/huggingface/accelerate/default_config.yaml`)

### Do you want a new feature?

A good feature request addresses the following points:

1. Motivation first:
* Is it related to a problem/frustration with the library? If so, please explain
  why. Providing a code snippet that demonstrates the problem is best.
* Is it related to something you would need for a project? We'd love to hear
  about it!
* Is it something you worked on and think could benefit the community?
  Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

## Submitting a pull request (PR)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
🤗 Accelerate. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/huggingface/accelerate) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote. The following command
   assumes you have your public SSH key uploaded to GitHub. See the following guide for more
   [information](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

   ```bash
   $ git clone git@github.com:<your Github handle>/accelerate.git
   $ cd accelerate
   $ git remote add upstream https://github.com/huggingface/accelerate.git
   ```

3. Create a new branch to hold your development changes, and do this for every new PR you work on.

   Start by synchronizing your `main` branch with the `upstream/main` branch (ore details in the [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)):

   ```bash
   $ git checkout main
   $ git fetch upstream
   $ git merge upstream/main
   ```

   Once your `main` branch is synchronized, create a new branch from it:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a conda or a virtual environment you've created for working on this library:

   ```bash
   $ pip install -e ".[quality]"
   ```

   (If accelerate was already installed in the virtual environment, remove
   it with `pip uninstall accelerate` before reinstalling it in editable
   mode with the `-e` flag.)

   Alternatively, if you are using [Visual Studio Code](https://code.visualstudio.com/Download), the fastest way to get set up is by using
   the provided Dev Container. Documentation on how to get started with dev containers is available [here](https://code.visualstudio.com/docs/remote/containers).

5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this (see 
   below an explanation regarding the environment variable):

   ```bash
   $ pytest tests/<TEST_TO_RUN>.py
   ```
   
   > For the following commands leveraging the `make` utility, we recommend using the WSL system when running on
   > Windows. More information [here](https://docs.microsoft.com/en-us/windows/wsl/about).

   You can also run the full suite with the following command.

   ```bash
   $ make test
   ```

   `accelerate` relies on `black` and `ruff` to format its source code
   consistently. After you make changes, apply automatic style corrections and code verifications
   that can't be automated in one go with:

   This target is also optimized to only work with files modified by the PR you're working on.

   If you prefer to run the checks one after the other, the following command apply the
   style corrections:

   ```bash
   $ make style
   ```

   `accelerate` also uses a few custom scripts to check for coding mistakes. Quality
   control runs in CI, however you can also run the same checks with:

   ```bash
   $ make quality
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`, or mark
   the PR as a draft PR. These are useful to avoid duplicated work, and to differentiate
   it from PRs ready to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests. No quality testing = no merge.

See an example of a good PR here: https://github.com/huggingface/accelerate/pull/255

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests folder](https://github.com/huggingface/accelerate/tree/main/tests).

We use `pytest` in order to run the tests. From the root of the
repository, here's how to run tests with `pytest` for the library:

```bash
$ python -m pytest -sv ./tests
```

In fact, that's how `make test` is implemented (sans the `pip install` line)!

You can specify a smaller set of tests in order to test only the feature
you're working on.