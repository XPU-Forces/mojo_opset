#!/usr/bin/env python3
"""Auto PR review via internal Anthropic-compatible endpoint.

Reads PR context from env, asks the model to review the diff, posts the
result as an issue comment on the PR.
"""
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://f7xnt9mg.fn.bytedance.net"
MODEL = "claude-opus-4-7"
MAX_DIFF_BYTES = 200_000
GITHUB_COMMENT_LIMIT_BYTES = 65_000  # GitHub hard limit is 65536 bytes; leave headroom


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

    # actions/checkout@v4 with fetch-depth: 0 already fetched all history,
    # so base should be reachable locally.
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

    data = None
    last_err = None
    for attempt in range(3):
        req = urllib.request.Request(
            f"{ENDPOINT}/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "Authorization": f"Bearer {anthropic_token}",
            },
        )
        try:
            with internal_opener.open(req, timeout=300) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            last_err = e
            print(f"attempt {attempt + 1} HTTP {e.code}: {e!r}", file=sys.stderr)
            # Don't retry hard auth/permission/format failures.
            if 400 <= e.code < 500 and e.code not in (408, 429):
                break
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0, 1))
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            print(f"attempt {attempt + 1} failed: {e!r}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    if data is None:
        sys.exit(f"LLM call failed after retries: {last_err!r}")

    if data.get("type") == "error" or "content" not in data:
        sys.exit(f"endpoint returned error payload: {data}")
    review = "".join(
        b.get("text", "") for b in data["content"] if b.get("type") == "text"
    )
    if not review.strip():
        sys.exit(f"no text content in response: {data}")
    print(f"got review, {len(review)} chars")

    comment = f"## Claude Code Review\n\n{review}"
    if truncated:
        comment += "\n\n_Note: the diff was truncated for review._"
    encoded = comment.encode("utf-8")
    if len(encoded) > GITHUB_COMMENT_LIMIT_BYTES:
        suffix = "\n\n_(comment truncated to fit GitHub limit)_"
        # Slice on bytes; decode with errors='ignore' to avoid splitting a
        # multi-byte char.
        budget = GITHUB_COMMENT_LIMIT_BYTES - len(suffix.encode("utf-8"))
        comment = encoded[:budget].decode("utf-8", errors="ignore") + suffix

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
    try:
        with external_opener.open(gh_req, timeout=60) as resp:
            print(f"posted comment: HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        sys.exit(f"GitHub API error: HTTP {e.code}\n{body}")
    except urllib.error.URLError as e:
        sys.exit(f"GitHub API request failed: {e!r}")


if __name__ == "__main__":
    main()
