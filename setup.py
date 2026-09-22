"""Package Python operators; native extensions are supplied by provider lib wheels."""

from pathlib import Path
from runpy import run_path
from uuid import uuid4

from setuptools import find_namespace_packages, setup
from setuptools.command.build import build
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
PACKAGING = run_path(str(ROOT / "_packaging.py"))


class PureBuild(build):
    def initialize_options(self):
        super().initialize_options()
        # Do not reuse artifacts from an earlier combined Python/native build.
        self.build_base = str(ROOT / "build" / "python-package" / uuid4().hex)


class NoSourceDistribution(sdist):
    def run(self):
        raise RuntimeError("sdist is unsupported: release wheels, or use an editable repository checkout")


setup(
    **PACKAGING["setup_metadata"](ROOT),
    packages=find_namespace_packages(
        include=["mojo_opset", "mojo_opset.*"],
        exclude=["mojo_opset._C", "mojo_opset._C.*", "mojo_opset._native", "mojo_opset._native.*"],
    ),
    py_modules=[],
    include_package_data=False,
    package_data={"mojo_opset.config": ["*.yaml"]},
    exclude_package_data={"": ["*.so", "*.pyd", "*.dll", "*.dylib"]},
    cmdclass={"build": PureBuild, "sdist": NoSourceDistribution},
)
