import shlex
from pathlib import Path

from ops.modal_hyperscalees_image import _MODAL_ENV, _REPO_MOUNT_IGNORE
from ops.modal_hyperscalees_pixi_base_image import HYPERSCALEES_PIXI_ENV, PIXI_BIN, PIXI_MANIFEST_PATH, mk_hyperscalees_pixi_base_image

_LASSOBENCH_SPEC = "LassoBench @ git+https://github.com/ksehic/LassoBench.git"


def mk_image(modal):
    project_root = Path(__file__).resolve().parents[1]
    image = mk_hyperscalees_pixi_base_image(modal, project_root).run_commands(_lassobench_extra_command()).env(_MODAL_ENV)
    return _add_repo_mount(image, project_root)


def _lassobench_extra_command() -> str:
    pixi_python = f"{shlex.quote(PIXI_BIN)} run --manifest-path {shlex.quote(PIXI_MANIFEST_PATH)} -e {shlex.quote(HYPERSCALEES_PIXI_ENV)} python"
    check = f"{pixi_python} -c {shlex.quote('import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("LassoBench") else 1)')}"
    install = f"{pixi_python} -m pip install --disable-pip-version-check --no-deps {shlex.quote(_LASSOBENCH_SPEC)}"
    return f"bash -lc {shlex.quote(check + ' || ' + install)}"


def _add_repo_mount(image, project_root: Path):
    return image.add_local_dir(
        str(project_root),
        remote_path="/root",
        ignore=list(_REPO_MOUNT_IGNORE),
    )
