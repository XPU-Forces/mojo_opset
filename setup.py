"""The main distribution contains Python code only."""
from pathlib import Path
from uuid import uuid4

from setuptools import setup
from setuptools.command.build import build


class PureBuild(build):
    def initialize_options(self):
        super().initialize_options()
        # Never reuse artifacts from an earlier combined Python/native build.
        self.build_base = str(Path(__file__).parent / "build" / "python-package" / uuid4().hex)


setup(cmdclass={"build": PureBuild})
