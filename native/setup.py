"""Package one prebuilt provider; compilation belongs to build.sh."""

import json
from pathlib import Path
from runpy import run_path
import sysconfig
from uuid import uuid4

from setuptools import Distribution, setup
from setuptools.command.build import build
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
PACKAGING = run_path(str(ROOT.parent / "_packaging.py"))
PROVIDER = PACKAGING["native_provider"](ROOT.parent)
PACKAGE = "mojo_opset_lib." + PROVIDER.replace("/", ".")
METADATA = PACKAGING["lib_setup_metadata"](ROOT.parent, PROVIDER)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # Standard wheel tagging uses this interpreter/host, never py3-none-any.
        return True


class ProviderBuild(build):
    def initialize_options(self):
        super().initialize_options()
        # Provider selections from the same checkout must not share build_lib.
        self.build_base = str(ROOT.parent / "build" / "lib-packages" / uuid4().hex)


class PrebuiltBuild(build_py):
    def run(self):
        if not self.editable_mode:
            self.validate_provider(PROVIDER)
        super().run()

    def validate_provider(self, provider):
        source = ROOT / "mojo_opset_lib" / provider
        expected = "_native" + sysconfig.get_config_var("EXT_SUFFIX")
        if not (source / expected).is_file():
            raise RuntimeError(
                f"Missing {source / expected}; run bash native/build.sh {provider} with this Python before packaging"
            )
        extensions = sorted(p.name for p in source.glob("_native*.so"))
        if extensions != [expected]:
            raise RuntimeError(f"Unexpected/stale extension ABI files: {extensions}; clean the artifacts and run bash native/build.sh {provider}")
        receipt = source / "_build_info.json"
        if not receipt.is_file():
            raise RuntimeError(f"Missing {receipt}; run bash native/build.sh {provider} before packaging")
        info = json.loads(receipt.read_text())
        if info.get("provider") != provider or info.get("operator") != "all":
            raise RuntimeError(f"Release wheels require a complete build: bash native/build.sh {provider}")
        if (info.get("python_soabi") != sysconfig.get_config_var("SOABI")
                or info.get("platform") != sysconfig.get_platform()):
            raise RuntimeError(f"Build interpreter/platform differs from wheel interpreter/platform; run bash native/build.sh {provider} with this Python")
        libraries = info.get("libraries", [])
        actual = sorted(p.name for p in source.glob("*.so"))
        if expected not in libraries or sorted(libraries) != actual:
            raise RuntimeError(f"Missing or stale native libraries: expected {libraries}, found {actual}; run bash native/build.sh {provider}")
        notices = info.get("notices", [])
        if sorted(notices) != sorted(p.name for p in source.glob("*_LICENSE.txt")):
            raise RuntimeError(f"Missing or stale native dependency notices; run bash native/build.sh {provider}")
        for notice in notices:
            if Path(notice).name != notice or not notice.endswith("_LICENSE.txt") or not (source / notice).is_file():
                raise RuntimeError(f"Missing native dependency notice: {notice}; run bash native/build.sh {provider}")
            if (source / notice).stat().st_size == 0:
                raise RuntimeError(f"Empty native dependency notice: {notice}; run bash native/build.sh {provider}")


class NoSourceDistribution(sdist):
    def run(self):
        raise RuntimeError("Lib sdist is unsupported: release prebuilt wheels, or build from an editable repository checkout")


setup(
    **METADATA,
    packages=[PACKAGE],
    package_dir={PACKAGE: "mojo_opset_lib/" + PROVIDER},
    py_modules=[],
    package_data={PACKAGE: ["*.so", "_build_info.json", "*_LICENSE.txt"]},
    include_package_data=False,
    distclass=BinaryDistribution,
    cmdclass={"build": ProviderBuild, "build_py": PrebuiltBuild, "sdist": NoSourceDistribution},
    zip_safe=False,
)
