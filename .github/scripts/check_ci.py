"""Validate change titles and compile CI Python sources without an accelerator."""

import json
import os
from pathlib import Path
import re
import subprocess


TITLE = re.compile(r"\[(Feat|Fix|Refactor|Doc)\] [a-z](?:.*[^.\s])?\.$")


def validate_title(title):
    return title.isprintable() and bool(TITLE.fullmatch(title))


def check_messages(event, event_name):
    errors = []
    pr = event.get("pull_request") if event_name == "pull_request" else None
    if pr:
        if not validate_title(pr["title"]):
            errors.append(f"Invalid PR title: {pr['title']!r}")
        revision = f"{pr['base']['sha']}..{pr['head']['sha']}"
    else:
        before, after = event.get("before", ""), event.get("after", "HEAD")
        revision = f"{before}..{after}" if before and set(before) != {"0"} else f"{after}^!"
    subjects = subprocess.check_output(
        ["git", "log", "--format=%s", "--reverse", revision, "--"], text=True,
    ).splitlines()
    if not subjects:
        errors.append("No commits found for this change")
    errors.extend(f"Invalid commit subject: {title!r}" for title in subjects if not validate_title(title))
    if errors:
        raise SystemExit("\n".join(errors) + "\nUse [Feat], [Fix], [Refactor] or [Doc], lowercase initial and one final period.")
    print(f"Validated {len(subjects)} commit subjects" + (" and PR title" if pr else ""))


def main():
    check_messages(json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), os.environ["GITHUB_EVENT_NAME"])
    for path in Path(".github/scripts").rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    print("CI Python syntax checks passed")


if __name__ == "__main__":
    main()
