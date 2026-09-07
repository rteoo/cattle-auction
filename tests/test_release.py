"""Release boundaries and package contents are explicit and reviewable."""

import tomllib
from pathlib import Path

from release import is_release_path_allowed


def test_release_staging_allowlist_excludes_generated_and_local_files():
    assert is_release_path_allowed("bench/analyze.py")
    assert is_release_path_allowed("README.md")
    assert not is_release_path_allowed("output/checkpoint.json")
    assert not is_release_path_allowed(".env")
    assert not is_release_path_allowed(".clawpatch/findings/report.json")


def test_build_manifest_includes_cli_and_prompts_and_excludes_local_tooling():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert wheel["force-include"] == {"main.py": "main.py", "prompts": "prompts"}
    assert set(sdist["exclude"]) == {"/.claude", "/.clawpatch"}
