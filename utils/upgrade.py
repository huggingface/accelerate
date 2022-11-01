# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
""" Util to check if a new version of a library dependency is available. """

import os
from github import Github
from pathlib import Path
import subprocess
import importlib_metadata, requests, re, json
from datetime import datetime
from packaging.requirements  import Requirement

def get_core_requirements(package:str):
    reqs = importlib_metadata.requires(package)
    reqs = list(map(Requirement, reqs))
    reqs = [r.name for r in reqs if r.marker is None]
    return reqs

def get_latest_upload_time(package:str):
    url = f"https://pypi.org/pypi/{package}/json"
    r = requests.get(url)
    data = r.json()
    latest_version = data["info"]["version"]
    latest_upload = data["releases"][latest_version][0]["upload_time"]
    match = re.search(r'\d{4}-\d{2}-\d{2}', latest_upload)
    release = datetime.strptime(match.group(), '%Y-%m-%d').date()
    return release

def post_upgrades():
    upgraded = []
    for package in get_core_requirements("accelerate"):
        latest_release = get_latest_upload_time(package)
        if (latest_release - datetime.now().date()).days > -12:
            upgraded.append(package)

    # Call subprocess 
    if len(upgraded) < 1: 
        upgraded = ''
    cmd = f"echo 'UPGRADES={upgraded}' >> $GITHUB_OUTPUT"
    
    subprocess.run(cmd, check=True, shell=True)

def comment_failures():
    print("Checking for failures...")
    failed = []

    for log in Path("../").glob("*.log"):
        with open(log, "r") as f:
            for line in f:
                line = json.loads(line)
                if line.get("nodeid", "") != "":
                    test = line["nodeid"]
                    if line.get("duration", None) is not None:
                        duration = f'{line["duration"]:.4f}'
                        if line.get("outcome", "") == "failed":
                            failed.append([test, duration, log.name.split('_')[0]])
    print(f"Num failures: {len(failed)}")
    if len(failed) > 0:
        result = "## Failed Tests:\n"
        failed_table = '| Test Location | Test Class | Test Name |\n|---|---|---|\n| '
        for test in failed:
            failed_table += ' | '.join(test[0].split("::"))
        result += failed_table
        g = Github(os.environ["GITHUB_TOKEN"])
        repo = g.get_repo("muellerzr/accelerate")
        issue = repo.create_issue(
            title="New Dependency Version Released, Failed Tests", 
            body=f'A new version of: {os.environ["UPGRADES"]} was released, but the tests failed. Please check the logs for more details [here](https://github.com/muellerzr/accelerate/actions/runs/{os.environ["GITHUB_RUN_ID"]}):\n{result}'
        )