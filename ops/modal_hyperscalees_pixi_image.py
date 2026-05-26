from pathlib import Path

from ops.modal_hyperscalees_image import _MODAL_ENV, _REPO_MOUNT_IGNORE
from ops.modal_hyperscalees_pixi_base_image import mk_hyperscalees_pixi_base_image


def mk_image(modal):
    project_root = Path(__file__).resolve().parents[1]
    image = mk_hyperscalees_pixi_base_image(modal, project_root).env(_MODAL_ENV)
    return _add_repo_mount(image, project_root)


def _add_repo_mount(image, project_root: Path):
    return image.add_local_dir(
        str(project_root),
        remote_path="/root",
        ignore=list(_REPO_MOUNT_IGNORE),
    )
