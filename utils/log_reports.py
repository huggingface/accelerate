import json
import os
from datetime import date
from pathlib import Path

from tabulate import tabulate, TableFormat, DataRow

hf_table_format = TableFormat(
        lineabove=None,
        linebelowheader=None,
        linebetweenrows=None,
        linebelow=None,
        headerrow=DataRow("", "|", "|"),
        datarow=DataRow("", "|", "|"),
        padding=1,
        with_header_hide=None,
    )


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
detailed_failed = []
if total_num_failed > 0:
    for name, num_failed, failed_tests in group_info:
        if num_failed > 0:
            if num_failed == 1:
                message += f"*{name[1:]}: {num_failed} failed test*\n"
            else:
                message += f"*{name[1:]}: {num_failed} failed tests*\n"
            failed_table = []
            files2failed = {}
            for test in failed_tests:
                data = test[0].split("::")
                data[0] = data[0].split("/")[-1]
                if data[0] not in files2failed:
                    files2failed[data[0]] = [data[1:]]
                else:
                    files2failed[data[0]] += [data[1:]]
                failed_table.append(data)
            detailed_failed.append(failed_table)

            files = [test[0] for test in failed_table]
            individual_files = list(set(files))
            # Count number of instances in failed_tests
            table = []
            for file in individual_files:
                table.append([file, len(files2failed[file])])

            failed_table = tabulate(
                table,
                headers=["Test Location", "Num Failed"],
                tablefmt=hf_table_format,
                stralign="right",
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
    response = client.chat_postMessage(channel="#accelerate-ci-daily", text=message, blocks=payload)
    ts = response.data["ts"]
    for k,v in files2failed.items():
        client.chat_postMessage(
            channel="#accelerate-ci-daily",
            thread_ts=ts,
            text=f"Test location: {k}\n```\n{tabulate(v, headers=['Class', 'Test'], tablefmt=hf_table_format, stralign='right')}\n```",
        )
