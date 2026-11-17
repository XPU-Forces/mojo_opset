"""Regression checks for the CPU gate; no accelerator required."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import check_ci
from check_api_coverage import Coverage
import pytest


class PreflightTests(unittest.TestCase):
    def test_title_format(self):
        for title in ('[Feat] a.', '[Fix] support CANN 9.1.', '[Refactor] simplify CI.', '[Doc] explain tests.'):
            self.assertTrue(check_ci.validate_title(title), title)
        for title in ('[Test] add tests.', '[CI] update CI.', '[Perf] improve speed.', '[Fix] Capital.',
                      '[Fix] missing period', '[Fix] double..', '[Fix] space .', '[Fix] tail. ',
                      '[Fix] line.\nbreak.', '[Fix] .'):
            self.assertFalse(check_ci.validate_title(title), title)

    def test_only_change_commits_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(['git', '-C', directory, *args], text=True).strip()
            git('init', '-q')
            git('config', 'user.email', 'ci@example.com')
            git('config', 'user.name', 'CI')
            git('commit', '--allow-empty', '-qm', 'Legacy title')
            base = git('rev-parse', 'HEAD')
            git('commit', '--allow-empty', '-qm', '[Feat] add checks.')
            head = git('rev-parse', 'HEAD')
            event = {'pull_request': {'title': '[Feat] add checks.', 'base': {'sha': base}, 'head': {'sha': head}}}
            with contextlib.chdir(directory):
                check_ci.check_messages(event, 'pull_request')
                check_ci.check_messages({'before': base, 'after': head}, 'push')
                event['pull_request']['title'] = '[Test] add checks.'
                with self.assertRaises(SystemExit):
                    check_ci.check_messages(event, 'pull_request')
                with self.assertRaises(SystemExit):
                    check_ci.check_messages({'before': '0' * 40, 'after': base}, 'push')

    def test_real_parameter_marks_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            test = Path(directory) / 'test_examples.py'
            test.write_text('''import pytest
@pytest.mark.accuracy
@pytest.mark.parametrize("value", [pytest.param(1, marks=pytest.mark.api("functions.foo")),
                                     pytest.param(2, marks=pytest.mark.api("modules.Foo", ops=["foo"]))])
def test_api(value):
    raise AssertionError("collection must not execute tests")
@pytest.mark.reference
@pytest.mark.accuracy
def test_reference():
    pass
''')
            coverage = Coverage({'functions.foo', 'modules.Foo'}, {'foo'})
            with patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': ''}), contextlib.redirect_stdout(io.StringIO()):
                status = pytest.main(['--collect-only', '-q', '-c', '/dev/null', str(test)], plugins=[coverage])
            self.assertEqual(status, 0)
            self.assertEqual(coverage.covered, coverage.public)
            self.assertEqual(coverage.errors, [])
            test.write_text('''import pytest
@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.api("functions.foo")
def test_reference(): pass
@pytest.mark.accuracy
@pytest.mark.api("functions.typo")
def test_typo(): pass
def test_unmarked(): pass
''')
            # A new filename avoids pytest's imported test-module cache.
            other = test.with_name('test_missing.py')
            test.rename(other)
            coverage = Coverage({'functions.foo'}, {'foo'})
            with patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': ''}), contextlib.redirect_stdout(io.StringIO()):
                status = pytest.main(['--collect-only', '-q', '-c', '/dev/null', str(other)], plugins=[coverage])
            self.assertEqual(status, 0)
            self.assertFalse(coverage.covered)
            self.assertTrue(any('missing api mark' in error for error in coverage.errors))
            self.assertTrue(any('unknown or empty API' in error for error in coverage.errors))
            self.assertTrue(any('No accuracy test declared' in error for error in coverage.errors))


if __name__ == '__main__':
    unittest.main()
