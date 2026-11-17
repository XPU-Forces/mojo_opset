"""Check collected accuracy declarations for every public function and module API."""

import os
from pathlib import Path
import sys

# Running by filename otherwise places only .github/scripts on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TORCH_TRANSFER_TO_NPU"] = "0"

import pytest


class Coverage:
    def __init__(self, public, ops):
        self.public, self.ops = set(public), set(ops)
        self.covered, self.errors = set(), []

    def pytest_collection_finish(self, session):
        for item in session.items:
            if item.get_closest_marker("reference"):
                continue
            markers = list(item.iter_markers("api"))
            if not markers:
                self.errors.append(f"{item.nodeid}: missing api mark")
            for marker in markers:
                names = set(marker.args)
                ops = set(marker.kwargs.get("ops", (name.removeprefix("functions.") for name in names)))
                if not names or not names <= self.public:
                    self.errors.append(f"{item.nodeid}: unknown or empty API declaration {sorted(names)}")
                if not ops or not ops <= self.ops:
                    self.errors.append(f"{item.nodeid}: unknown or empty operator declaration {sorted(ops)}")
                if item.get_closest_marker("accuracy"):
                    self.covered.update(names & self.public)
        missing = self.public - self.covered
        self.errors.extend(f"No accuracy test declared for {name}" for name in sorted(missing))
        summary = f"Public API accuracy declarations: {len(self.covered)}/{len(self.public)}"
        print(summary)
        for error in self.errors:
            print(error)
        if path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(path, "a") as stream:
                stream.write(f"### Operator test coverage\n\n{summary}\n\n")
                stream.write("Checks collected test declarations; hardware execution is verified by NPU jobs.\n")
                for error in self.errors:
                    stream.write(f"\n- {error}\n")


def main():
    from mojo_opset import functions, modules
    from mojo_opset.kernels.torch import OPS

    public = {f"{namespace}.{name}" for namespace, module in (("functions", functions), ("modules", modules))
              for name in module.__all__}
    coverage = Coverage(public, OPS)
    status = pytest.main(["--collect-only", "-q", "-o", "log_cli=false", "tests/functions", "tests/modules"],
                         plugins=[coverage])
    return int(status) or int(bool(coverage.errors))


if __name__ == "__main__":
    sys.exit(main())
