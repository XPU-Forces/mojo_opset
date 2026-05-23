#!/usr/bin/env python3
"""Auto PR review via internal Anthropic-compatible endpoint.

Reads PR context from env, asks the model to review the diff, posts the
result as an issue comment on the PR.
"""
import json
import os
import subprocess
import sys
import urllib.request

ENDPOINT = "https://f7xnt9mg.fn.bytedance.net"
MODEL = "claude-opus-4-7"
MAX_DIFF_BYTES = 200_000


def env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"missing required env var: {name}")
    return val


def main() -> None:
    base = env("BASE_SHA")
    head = env("HEAD_SHA")
    pr = env("PR_NUMBER")
    repo = env("REPO")
    anthropic_token = env("ANTHROPIC_AUTH_TOKEN")
    gh_token = env("GITHUB_TOKEN")
    proxy = os.environ.get("OUTBOUND_PROXY") or None

    # Make sure base is fetched locally; on PR events checkout may not have it.
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=200", "origin", base],
        check=False,
    )
    diff = subprocess.check_output(
        ["git", "diff", f"{base}...{head}"], text=True
    )
    if not diff.strip():
        print("empty diff, nothing to review")
        return

    truncated = len(diff) > MAX_DIFF_BYTES
    if truncated:
        diff = diff[:MAX_DIFF_BYTES]

    prompt = (
        "You are reviewing a pull request for the mojo_opset repository "
        "(GPU/NPU operator kernels in Python/Triton). Focus on:\n"
        "- Bugs, correctness issues, off-by-one errors\n"
        "- Numerical / dtype / shape mismatches\n"
        "- Concurrency / synchronization issues in kernel code\n"
        "- Performance regressions\n"
        "- Security concerns\n"
        "- Maintainability / readability\n\n"
        "Be concise. Use markdown. Reference file paths and line numbers when "
        "possible. If there are no significant issues, say so briefly.\n\n"
        f"```diff\n{diff}\n```"
    )
    if truncated:
        prompt += "\n\n(diff truncated to fit context window)"

    # Internal endpoint - bypass any env-configured proxy.
    internal_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "Authorization": f"Bearer {anthropic_token}",
        },
    )
    with internal_opener.open(req, timeout=300) as resp:
        data = json.loads(resp.read())
    review = "".join(b["text"] for b in data["content"] if b["type"] == "text")
    print(f"got review, {len(review)} chars")

    comment = f"## Claude Code Review\n\n{review}"
    if truncated:
        comment += "\n\n_Note: the diff was truncated for review._"

    # GitHub API needs the outbound proxy.
    if proxy:
        external_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    else:
        external_opener = urllib.request.build_opener()

    gh_req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
        data=json.dumps({"body": comment}).encode(),
        headers={
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with external_opener.open(gh_req, timeout=60) as resp:
        print(f"posted comment: HTTP {resp.status}")


if __name__ == "__main__":
    main()
