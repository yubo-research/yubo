from __future__ import annotations

from typing import Any


def cleanup_ray_launch(ray: Any, actors: list[Any], placement_groups: list[Any]) -> None:
    kill_ray_actors(ray, actors)
    for pg in placement_groups:
        try:
            ray.util.remove_placement_group(pg)
        except Exception:
            pass


def kill_ray_actors(ray: Any, actors: list[Any]) -> None:
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass


__all__ = ["cleanup_ray_launch", "kill_ray_actors"]
