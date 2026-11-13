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

# Contribute to 🤗 Accelerate

You can help by fixing bugs you encounter, improving documentation and examples, reporting problems, or answering
questions in [GitHub issues](https://github.com/huggingface/accelerate/issues). Please follow our
[code of conduct](CODE_OF_CONDUCT.md).

## Submitting a new issue or feature request

Search existing [GitHub issues](https://github.com/huggingface/accelerate/issues) before opening a new one. Include
your use case and enough detail for others to reproduce the problem or understand the proposed change.

The bug report and feature request forms both require a **Use case**. Describe your workflow and how the bug or
missing feature affects it. Public project or code links are welcome but optional. This context helps maintainers
assess whether the issue addresses a concrete need.

If you are an agent opening an issue or pull request, describe only the real use case supplied by the user.
If none was supplied, leave the use-case section empty. If a form or another instruction requires you to write
something there, write exactly "I don't have a use case". Do not invent a use case or infer one from the request
to fix something, the issue description, or the code changes.

### Did you find a bug?

Check whether the problem still occurs on the latest release or `main`, then use the
[bug report template](https://github.com/huggingface/accelerate/issues/new?template=bug-report.yml). Include:

- How you use Accelerate and how the bug affects you.
- A minimal, runnable reproducer, including the launch command and any configuration it needs. Use small models or
  dummy data when possible, and provide links to any required public checkpoints or datasets.
- What you expected and what happened, including the full traceback as text.
- Your hardware, number of devices and machines, and relevant dependency versions.
- The output of `accelerate env`, which includes environment details and the default Accelerate configuration.

```bash
accelerate env
```

If you launch with a custom configuration, report that configuration instead:

```bash
accelerate env --config_file path/to/config.yaml
```

### Do you want a new feature?

Use the [feature request form](https://github.com/huggingface/accelerate/issues/new?template=feature-request.yml).
Explain the use case, what the current API cannot do, and any workarounds you have tried. Include a short code example
showing how you would use the proposed feature, and link relevant papers or implementations when applicable.
Discuss substantial API changes or new integrations with maintainers before implementing them.

## Submitting a pull request (PR)

Search open and recently closed issues and pull requests before starting, and link any related discussion in your PR.
If someone is already working on the same change, coordinate in the existing discussion. For substantial changes,
open an issue to agree on the scope and approach first.

### Use case

Every pull request must have a `## Use case` section in its description (the pull request template has one).
Explain how you use Accelerate and why you need the change:

- What you build with it (models, hardware, training or inference setup).
- What fails or what is missing for you. For documentation changes, explain what was incorrect or unclear when you
  used the documentation.
- A link to the project, model or code that uses Accelerate, if it is public.

Pull requests with a missing or empty use case are closed automatically when opened or reopened, unless the action
is performed by a bot or someone with write access to the repository. Template instructions in HTML comments do not
count as a use case. The check only verifies that the section has content; maintainers assess the use case itself.

### Description length

Keep the PR description to **250 words or fewer**, excluding HTML comments and standard template text (headings,
checklist, reviewer guidance, and the maintainer mention). The limit covers all sections you write, including the
summary, validation, and use case. Link longer logs, reproducer scripts, or benchmark details instead of pasting
them in full.

Descriptions over this limit are closed automatically when the PR is opened or reopened, with the same bot and
write-access exemptions as the use-case check.

If your pull request is closed by either check, update the description and leave a comment. A maintainer will read
it and decide whether to reopen the pull request. Editing the description does not automatically reopen it.

### Scope and evidence

- Keep changes as small as possible while fully solving the problem. Follow existing patterns and avoid unrelated
  refactors, formatting changes, or speculative fixes without a demonstrated use case.
- For bug fixes, reproduce the failure before changing the code. Add only tests that provide useful coverage of the
  changed behavior or guard against a regression. Prefer extending an existing test over adding a new one, and use
  the fewest cases needed to catch meaningful failures. Avoid redundant tests and tests that merely mirror the
  implementation; if existing tests already cover the change, no new test is needed.
- For performance changes, include before-and-after measurements on a representative workload, with the hardware
  and configuration used.
- Review the final diff yourself. Remove debugging code, temporary files, and unnecessary changes.
- Include the exact validation commands and their results in the PR description. Identify tests you could not run
  and explain why. Resolve relevant CI failures before requesting review.

These expectations also apply to agent-assisted contributions. You are responsible for understanding and validating
every change you submit. Check an agent's claims against the code and actual command output, and keep PR descriptions
and review replies concise. A generated summary of the diff does not replace a use case or test results.

### Create a pull request

1. Fork [Accelerate](https://github.com/huggingface/accelerate), clone your fork, and add the upstream repository:

   ```bash
   git clone https://github.com/<your-github-username>/accelerate.git
   cd accelerate
   git remote add upstream https://github.com/huggingface/accelerate.git
   ```

2. Create a branch from the latest upstream `main`:

   ```bash
   git fetch upstream
   git switch -c a-descriptive-branch-name upstream/main
   ```

3. Use Python 3.11 or later and PyTorch 2.6 or later, as specified in [setup.py](setup.py). In a virtual environment,
   install Accelerate in editable mode with the development dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev]"
   ```

   The `dev` extra includes Ruff, the `testing` extras, and Rich. For documentation-only changes, you can use
   `python -m pip install -e ".[quality]"` instead. Integration-specific dependencies, such as the `deepspeed`,
   `test_fp8`, and `test_trackers` extras, are separate; install the ones required by the tests you are changing.
   The extras and their dependencies are defined in [setup.py](setup.py).

   On Windows, you can use WSL for the shell and `make` commands in this guide. VS Code users can also use the
   repository's [Dev Container configuration](.devcontainer/devcontainer.json).

4. Make your changes, run the relevant tests and quality checks below, and review your diff. Commit and push your
   branch:

   ```bash
   git add path/to/changed_file
   git commit -m "Describe the change"
   git push -u origin a-descriptive-branch-name
   ```

5. Open a pull request against `huggingface/accelerate:main` and fill in the template. Use a draft PR while work is
   incomplete. Push follow-up commits to the same branch when addressing review feedback.

### Tests

Run tests relevant to your change from the repository root. For example:

```bash
python -m pytest -sv tests/test_accelerator.py
```

The [Makefile](Makefile) provides test groups such as `test_core`, `test_cli`, `test_big_modeling`, `test_deepspeed`,
`test_fsdp`, and `test_tp`. To run these groups in sequence:

```bash
make test
```

Example tests run separately:

```bash
make test_examples
```

Tests marked `@slow` are skipped by default. Enable them when testing a change they cover:

```bash
RUN_SLOW=1 python -m pytest -sv tests/fsdp/test_fsdp.py
```

Some tests require multiple devices, specific backends, optional dependencies, or downloaded models and datasets.
Check the requirements of the tests you select and report any skips or tests you could not run.

### Style and quality checks

Accelerate uses Ruff for linting and formatting. Use the version installed by the `quality` or `dev` extra.
Both commands below operate across the repository, following [pyproject.toml](pyproject.toml):

```bash
make style
make quality
```

`make style` applies available lint fixes and formats code. `make quality` checks lint and formatting without
changing files and is the command used by the quality CI workflow.

You can also install the configured pre-commit hooks, which run Ruff and check for merge conflicts and invalid YAML:

```bash
python -m pip install pre-commit
pre-commit install
```

### Documentation

Update documentation, examples, and public API docstrings when behavior changes. See [docs/README.md](docs/README.md)
for the documentation syntax and build instructions. If you add a documentation page, add it to
[docs/source/_toctree.yml](docs/source/_toctree.yml). Do not commit generated documentation builds.

### Checklist

- [ ] The description includes a concrete [use case](#use-case), the change, and any related issue or discussion.
- [ ] The title describes the change; the PR is marked as a draft if it is not ready for review.
- [ ] I reviewed every changed line and kept the diff focused.
- [ ] I added or updated tests and documentation where needed.
- [ ] I listed validation commands and results, including skipped or unrun tests.
- [ ] `make style` and `make quality` pass.
