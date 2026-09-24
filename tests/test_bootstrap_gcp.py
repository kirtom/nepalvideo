"""The startup script: syntax, idempotence markers, and the contract."""
import pathlib
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nepal.cloud import watchdog

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


def test_the_key_is_fetched_at_login_and_never_written_by_the_script():
    """The script runs under `set -x`; anything it holds in a variable is
    traced into the bootstrap log, syslog and Cloud Logging. So it never
    reads the key: the profile line it writes does the fetch at login."""
    s = SCRIPT.read_text()
    assert "md anthropic-api-key" not in s
    assert "KEY=" not in s.replace("ANTHROPIC_API_KEY=", "")
    fetch = [l for l in s.splitlines() if "attributes/anthropic-api-key" in l and "curl" in l]
    assert len(fetch) == 1 and "Metadata-Flavor" in fetch[0]
    assert fetch[0].startswith("_k=")             # inside the heredoc, not run by the script
    # the first version's plaintext line is removed, the fetch line is kept
    assert "/^export ANTHROPIC_API_KEY=/{/metadata.google.internal/!d}" in s


def test_the_user_manager_lingers_before_the_first_rsync():
    """A readiness probe's logout stopped the user's systemd manager and
    killed the snap gcloud rsync registered in it (exit 143)."""
    s = SCRIPT.read_text()
    assert s.index("loginctl enable-linger $USER_NAME") < s.index("gcloud storage rsync")


def test_every_boot_installs_the_idle_watchdog_and_stamps_the_boot():
    """The box has to be able to stop itself: the session that should have
    stopped it is exactly what failed in 2026-09. The stamp is touched
    after the bucket rsync, or the clock would start at whatever mtime came
    back with work/."""
    s = SCRIPT.read_text()
    # the module renders its own unit and touches its own stamp: nothing the
    # watchdog reads is spelled out here to drift from it
    assert f"-m {watchdog.__name__} install | bash" in s
    assert f"-m {watchdog.__name__} touch" in s
    assert s.index("rsync --recursive \"$BUCKET/work\" $ROOT/nepal_work") < \
        s.index(f"-m {watchdog.__name__} touch")


def test_a_box_with_state_pushes_work_on_boot_rather_than_pulling():
    """After a preemption the disk is newer than the bucket: a run pushes
    only when it ends, and a preempted run never did."""
    s = SCRIPT.read_text()
    assert 'if [ -f $ROOT/nepal_work/db/nepal.sqlite ]; then' in s
    push = s.index("rsync --recursive $ROOT/nepal_work \"$BUCKET/work\"")
    pull = s.index("rsync --recursive \"$BUCKET/work\" $ROOT/nepal_work")
    assert push < pull
