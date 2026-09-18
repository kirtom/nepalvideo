"""The startup script: syntax, idempotence markers, and the contract."""
import pathlib
import shutil
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "cloud" / "bootstrap-gcp.sh"


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_script_reads_its_metadata_and_writes_the_markers():
    s = SCRIPT.read_text()
    for attr in ("nepal-profile", "nepal-branch", "nepal-bucket"):
        assert f"md {attr}" in s
    assert "attributes/$1" in s
    assert "ROOT=/data/projects" in s
    assert "$ROOT/.toolchain" in s                   # toolchain once
    assert "$ROOT/READY" in s                        # what `up` waits for
    assert "nvidia-smi" in s and "install_gpu_driver.py" in s
    assert "gcloud storage rsync" in s
    assert "git fetch" in s and "reset -q --hard" in s   # the box tracks the branch
    assert "ANTHROPIC_API_KEY" in s                   # from metadata, into the environment
