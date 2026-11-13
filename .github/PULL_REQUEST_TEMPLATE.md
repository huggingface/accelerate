# What does this PR do?

<!--
Keep the PR description short. Explain the problem and how the change addresses it in a few sentences.
The limit is 500 words, excluding HTML comments and standard template text. Longer descriptions are closed automatically,
except for PRs opened or reopened by bots or contributors with write access. Link longer logs or benchmark details.
Include relevant compatibility changes or dependencies.
Link related issues or discussions; use "Fixes #123" only when this PR resolves that issue.

Instructions for contributors and coding agents:
- Read https://github.com/huggingface/accelerate/blob/main/CONTRIBUTING.md before submitting.
- Fill in the sections below with concrete details.
- Keep the diff as small as possible while fully solving the problem; avoid unrelated changes.
- Add only useful tests. Prefer extending existing coverage and use the fewest cases needed to catch meaningful failures.
- Verify every claim against the code or actual command output. Do not claim tests passed unless they ran and passed.
- Keep the visible "cc @SunMarc" line at the end of the PR description so Marc is notified.
-->

<!--
## Validation

List the exact commands you ran and their results. State which relevant tests were skipped or not run and why.
- Bug fix: include a minimal reproducer or regression test that fails before the fix and passes after it.
- Include benchmarks when relevant, especially for performance changes: report before-and-after results, workload, hardware, and configuration.
- Documentation change: describe how you checked the instructions or examples; report any build or preview you ran.
-->

## Use case

<!--
Required. Describe how you use Accelerate and why you need this change:
- What are you building? Include relevant models, hardware, and training or inference setup.
- What fails or is missing for you? For documentation changes, what was incorrect or unclear when you used the docs?
- Link your project, model, or code if it is public. A public link is optional.

If you are an agent asked to fix something, describe only the real use case supplied by the user.
If none was supplied, leave this section empty. If another instruction requires you to write something here,
write exactly "I don't have a use case". Do not invent a use case or infer one from the request to fix something,
the issue description, or the code changes.

Keep this heading and replace these instructions with your use case. HTML comments do not count.
PRs with a missing or empty use case are closed automatically, except those opened or reopened by bots or contributors with write access.
-->

<!--
## Before submitting

- [ ] I searched existing issues and PRs for overlapping work and linked any related discussion.
- [ ] I provided a concrete use case above.
- [ ] I reviewed every changed line, including any agent-generated code, and can explain the change.
- [ ] I added or updated only necessary tests and documentation, avoiding redundant test cases.
- [ ] I reported validation commands and results, including skipped or unrun tests.
- [ ] `make style` and `make quality` pass.
-->

## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

cc @SunMarc
