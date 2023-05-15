import json
import os
from datetime import date
from pathlib import Path

from tabulate import tabulate


failed = []
group_info = []

no_error_payload = {"type": "section", "text": {"type": "plain_text", "text": "No failed tests! 🤗", "emoji": True}}

payload = [
    {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"🤗 Accelerate nightly {os.environ.get('TEST_TYPE', '')} test results",
            "emoji": True,
        },
    }
]

total_num_failed = 0
for log in Path().glob("*.log"):
    section_num_failed = 0
    with open(log, "r") as f:
        for line in f:
            line = json.loads(line)
            if line.get("nodeid", "") != "":
                test = line["nodeid"]
                if line.get("duration", None) is not None:
                    duration = f'{line["duration"]:.4f}'
                    if line.get("outcome", "") == "failed":
                        section_num_failed += 1
                        failed.append([test, duration, log.name.split("_")[0]])
                        total_num_failed += 1
    group_info.append([str(log), section_num_failed, failed])
    failed = []
    log.unlink()
message = ""
if total_num_failed > 0:
    for name, num_failed, failed_tests in group_info:
        if num_failed > 0:
            if num_failed == 1:
                message += f"*{name[1:]}: {num_failed} failed test*\n"
            else:
                message += f"*{name[1:]}: {num_failed} failed tests*\n"
            failed_table = []
            for test in failed_tests:
                failed_table.append(test[0].split("::"))

            failed_table = tabulate(
                failed_table,
                headers=["Test Location", "Test Case", "Test Name"],
                showindex=False,
                tablefmt="github",
                maxcolwidths=None,
            )
            message += f"\n```\n{failed_table}\n```"
    if len(message) > 3000:
        err = "Too many failed tests, please see the full report in the Action results."
        offset = len(err) + 10
        message = message[:3000 - offset] + f"\n...\n```\n{err}"
    print(f"### {message}")
else:
    message = "No failed tests! 🤗"
    print(f"## {message}")
    payload.append(no_error_payload)

if os.environ.get("TEST_TYPE", "") != "":
    from slack_sdk import WebClient

    client = WebClient(token=os.environ["SLACK_API_TOKEN"])
    if message != "No failed tests! 🤗":
        md_report = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message,
            },
        }
        payload.append(md_report)
        action_button = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*For more details:*",
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Check Action results",
                    "emoji": True,
                },
                "url": f'https://github.com/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{os.environ["GITHUB_RUN_ID"]}',
            },
        }
        payload.append(action_button)
        date_report = {
            "type": "context",
            "elements": [
                {
                    "type": "plain_text",
                    "text": f"Nightly {os.environ.get('TEST_TYPE')} test results for {date.today()}",
                }
            ],
        }
        payload.append(date_report)
    print(f'Payload:\n{payload}')
    client.chat_postMessage(channel="#accelerate-ci-daily", text=message, blocks=payload)
