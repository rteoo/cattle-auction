"""Release boundaries and package contents are explicit and reviewable."""

import subprocess

import pytest

from release import stage_release_changes


def test_release_stages_only_allowed_paths_in_a_real_git_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "example.py").write_text("value = 1\n")
    (tmp_path / "private-notes.txt").write_text("synthetic unrelated file")

    stage_release_changes(False)

    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True)
    assert staged.splitlines() == ["pipeline/example.py"]


def test_release_refuses_unrelated_prestaged_files_before_touching_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "README.md").write_text("synthetic release documentation")
    (tmp_path / "private-notes.txt").write_text("synthetic unrelated file")
    subprocess.run(["git", "add", "--", "private-notes.txt"], check=True)

    with pytest.raises(SystemExit):
        stage_release_changes(False)

    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True)
    assert staged.splitlines() == ["private-notes.txt"]
