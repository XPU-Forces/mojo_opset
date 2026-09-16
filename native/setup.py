"""Package one prebuilt provider; use build.sh for compilation, not pip."""

import json
import os
from pathlib import Path
import re
import sysconfig
from uuid import uuid4

from setuptools import Distribution, setup
from setuptools.command.build import build
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
PROVIDER = os.environ.get("MOJO_LIB_PROVIDER", "")
if not re.fullmatch(r"[a-z][a-z0-9]*_[a-z0-9_]+(?:/sku_[a-z0-9_]+)?", PROVIDER):
    raise ValueError("Set MOJO_LIB_PROVIDER, for example npu_a2 or npu_a5/sku_950pr")
SOURCE = ROOT / "mojo_opset_lib" / PROVIDER
if not (SOURCE / "__init__.py").is_file() or not (ROOT / "src" / PROVIDER / "bindings.cpp").is_file():
    raise ValueError(f"Unsupported MOJO_LIB_PROVIDER: {PROVIDER!r}")
PACKAGE = "mojo_opset_lib." + PROVIDER.replace("/", ".")
SUFFIX = PROVIDER.replace("/sku_", "-").replace("_", "-")
VERSION = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT.parent / "pyproject.toml").read_text(), re.M).group(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # Standard wheel tagging uses this interpreter/host, never py3-none-any.
        return True


class ProviderBuild(build):
    def initialize_options(self):
        super().initialize_options()
        # Two provider builds from the same checkout must not share build_lib.
        self.build_base = str(ROOT.parent / "build" / "lib-packages" / SUFFIX / uuid4().hex)


class PrebuiltBuild(build_py):
    def run(self):
        if not self.editable_mode:
            expected = "_native" + sysconfig.get_config_var("EXT_SUFFIX")
            if not (SOURCE / expected).is_file():
                raise RuntimeError(
                    f"Missing {SOURCE / expected}; run bash native/build.sh {PROVIDER} with this Python before packaging"
                )
            extensions = sorted(p.name for p in SOURCE.glob("_native*.so"))
            if extensions != [expected]:
                raise RuntimeError(f"Unexpected/stale extension ABI files: {extensions}; clean the artifacts and run bash native/build.sh {PROVIDER}")
            receipt = SOURCE / "_build_info.json"
            if not receipt.is_file():
                raise RuntimeError(f"Missing {receipt}; run bash native/build.sh {PROVIDER} before packaging")
            info = json.loads(receipt.read_text())
            if info.get("provider") != PROVIDER or info.get("operator") != "all":
                raise RuntimeError(f"Release wheels require a complete build: bash native/build.sh {PROVIDER}")
            if (info.get("python_soabi") != sysconfig.get_config_var("SOABI")
                    or info.get("platform") != sysconfig.get_platform()):
                raise RuntimeError(f"Build interpreter/platform differs from wheel interpreter/platform; run bash native/build.sh {PROVIDER} with this Python")
            libraries = info.get("libraries", [])
            actual = sorted(p.name for p in SOURCE.glob("*.so"))
            if expected not in libraries or sorted(libraries) != actual:
                raise RuntimeError(f"Missing or stale native libraries: expected {libraries}, found {actual}; run bash native/build.sh {PROVIDER}")
            notices = info.get("notices", [])
            if sorted(notices) != sorted(p.name for p in SOURCE.glob("*_LICENSE.txt")):
                raise RuntimeError(f"Missing or stale native dependency notices; run bash native/build.sh {PROVIDER}")
            for notice in notices:
                if Path(notice).name != notice or not notice.endswith("_LICENSE.txt") or not (SOURCE / notice).is_file():
                    raise RuntimeError(f"Missing native dependency notice: {notice}; run bash native/build.sh {PROVIDER}")
                if (SOURCE / notice).stat().st_size == 0:
                    raise RuntimeError(f"Empty native dependency notice: {notice}; run bash native/build.sh {PROVIDER}")
        super().run()


class NoSourceDistribution(sdist):
    def run(self):
        raise RuntimeError("Lib sdist is unsupported: packages are prebuilt wheels only; build from the repository checkout")


setup(
    name=f"byted-mojo-opset-lib-{SUFFIX}",
    version=VERSION,
    description=f"Prebuilt Mojo libraries for {PROVIDER}",
    python_requires=">=3.9",
    install_requires=[f"byted-mojo-opset=={VERSION}"],
    packages=[PACKAGE],
    package_dir={PACKAGE: str(SOURCE.relative_to(ROOT))},
    package_data={PACKAGE: ["*.so", "_build_info.json", "*_LICENSE.txt"]},
    include_package_data=False,
    distclass=BinaryDistribution,
    cmdclass={"build": ProviderBuild, "build_py": PrebuiltBuild, "sdist": NoSourceDistribution},
    zip_safe=False,
)
