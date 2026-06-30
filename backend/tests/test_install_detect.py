"""install.sh GPU auto-detect selection — verified via stubbed PATH, no real GPU.

install.sh probes NVIDIA with a throwaway `--gpus` container and AMD with host
`rocminfo`. We stub `docker`/`rocminfo`/`curl`/`nvidia-smi` so the selection logic
runs to the `.env` write without touching real hardware or building an image.
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"


def _run(tmp_path: Path, *, nvidia_ok: bool, rocminfo_present: bool) -> dict[str, str]:
    """Run install.sh in an isolated copy with stubbed external tools; return parsed .env."""
    work = tmp_path / "repo"
    work.mkdir()
    shutil.copy(INSTALL_SH, work / "install.sh")

    bindir = tmp_path / "bin"
    bindir.mkdir()

    # docker: `run ...` is the nvidia probe (exit per scenario); `compose ...` is the
    # build/up + logs (always succeed so install.sh reaches its health check).
    (bindir / "docker").write_text(
        '#!/usr/bin/env bash\nif [[ "$1" == "run" ]]; then exit %d; fi\nexit 0\n' % (0 if nvidia_ok else 1)
    )
    # curl: health endpoint returns the sentinel install.sh greps for.
    (bindir / "curl").write_text('#!/usr/bin/env bash\necho \'{"overall":"ok"}\'\n')
    # nvidia-smi: present-but-no-GPU so the cpu-branch hint path doesn't error.
    (bindir / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 1\n")
    if rocminfo_present:
        (bindir / "rocminfo").write_text("#!/usr/bin/env bash\nexit 0\n")
    for f in bindir.iterdir():
        f.chmod(0o755)

    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(tmp_path)}
    subprocess.run(["bash", "install.sh"], cwd=work, env=env, check=True, capture_output=True)

    parsed = {}
    for line in (work / ".env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k] = v
    return parsed


def test_nvidia_host_selects_cuda(tmp_path):
    env = _run(tmp_path, nvidia_ok=True, rocminfo_present=False)
    assert env["TORCH_VARIANT"] == "cuda"
    assert env["COMPOSE_FILE"] == "compose.yml:compose.cuda.yml"


def test_amd_host_selects_rocm(tmp_path):
    env = _run(tmp_path, nvidia_ok=False, rocminfo_present=True)
    assert env["TORCH_VARIANT"] == "rocm"
    assert env["COMPOSE_FILE"] == "compose.yml:compose.rocm.yml"


def test_plain_host_selects_cpu_no_amd_false_positive(tmp_path):
    env = _run(tmp_path, nvidia_ok=False, rocminfo_present=False)
    assert env["TORCH_VARIANT"] == "cpu"
    assert "rocm" not in env["COMPOSE_FILE"]
