# Copyright 2022 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
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
"""
Script to check if the setup.py was modified during a Pull Request,
and if so will write a comment stating that Docker Images will be 
rebuilt upon a merge. 

Environment variables available:
- PR_NUMBER: The number of the currently open pull request
- REPO: The full name of the repository (such as 'huggingface/accelerate')
"""

import os
import re
from github import Github, PullRequest

def get_setup_diff(pull_request:PullRequest):
    """Checks whether `setup.py` was changed during this pull request, and will 
    return the diff if so

    Args:
        pull_request (`PullRequest`):
            A pull request returned from `Github.repo.get_pull_request`
    """
    files = pull_request.get_files()
    for f in files:
        if f.filename == "setup.py":
            return f'''```diff\n{f.patch}\n```'''
    return None

def does_comment_exist(pull_request:PullRequest):
    """Checks whether the bot has already commented on this pull request

    Args:
        pull_request (`PullRequest`):
            A pull request returned from `Github.repo.get_pull_request`
    """
    comments = pull_request.get_issue_comments()
    for c in comments:
        if c.user.login == "github-actions[bot]" and 'This PR modifies `setup.py`.' in c.body:
            return True
    return False

def write_comment(pull_request:PullRequest, diff:str):
    """Writes a comment stating that the pr modified setup.py, and that new Docker images will be built

    Args:
        pull_request (`PullRequest`):
            A pull request returned from `Github.repo.get_pull_request`
        diff (`str`):
            The diff of the modified setup.py
    """
    s = f'This PR modifies `setup.py`. New latest Docker images will be built and deployed once this has been merged:\n\n{diff}'
    pull_request.create_comment(s)

def update_diff(pull_request:PullRequest, diff:str):
    """Updates the diff of the setup.py in the existing comment

    Args:
        pull_request (`PullRequest`):
            A pull request returned from `Github.repo.get_pull_request`
        diff (`str`):
            The diff of the modified setup.py
    """
    comments = pull_request.get_issue_comments()
    for c in comments:
        if c.user.login == "github-actions[bot]" and 'This PR modifies `setup.py`.' in c.body:
            comment = c
            break
    original_diff = re.search(r'```([^`]*)```', comment.body).group(0)
    new_body = comment.body.replace(original_diff, diff)
    comment.edit(new_body)

def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo(os.environ["REPO"])
    pr = repo.get_pr(os.environ["PR_NUMBER"])
    diff = get_setup_diff(pr)
    if diff is not None:
        if does_comment_exist(pr):
            update_diff(pr, diff)
        else:
            write_comment(pr, diff)

if __name__ == "__main__":
    main()