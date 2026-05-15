from __future__ import annotations

from typing import Any

from llm.vllm_net import get_ip, get_open_port
from llm.vllm_worker_update import apply_lora_es_update as _apply_lora_es_update


class WorkerExtension:
    def get_transport_info(self) -> dict[str, int | str]:
        return {"tensor_rank": _tensor_parallel_rank(), "host": get_ip(), "port": get_open_port()}

    def init_inter_engine_group(
        self,
        master_by_tensor_rank: dict[int, tuple[str, int]] | tuple[str, int],
        engine_rank: int,
        world_size: int,
    ) -> bool:
        self.device = self.model_runner.device
        self.gpu_rank = int(engine_rank)
        self.world_size = int(world_size)
        self.inter_pg = None
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup
        except ImportError:
            return True

        tensor_rank = _tensor_parallel_rank()
        if isinstance(master_by_tensor_rank, dict):
            master = master_by_tensor_rank.get(tensor_rank) or master_by_tensor_rank.get(str(tensor_rank)) or master_by_tensor_rank.get(0)
            if master is None:
                raise RuntimeError(f"No inter-engine transport info for tensor rank {tensor_rank}.")
            master_address, master_port = master
        else:
            master_address, master_port = master_by_tensor_rank
        pg = StatelessProcessGroup.create(
            host=master_address,
            port=int(master_port),
            rank=int(engine_rank),
            world_size=int(world_size),
        )
        self.inter_pg = PyNcclCommunicator(pg, device=self.device)
        return True

    def apply_lora_es_update(
        self,
        normalized_fitnesses: list[float],
        peft_shapes_dict: dict[str, tuple[int, ...]],
        es_step: int,
        args: Any,
    ) -> bool:
        return _apply_lora_es_update(self, normalized_fitnesses, peft_shapes_dict, es_step, args)

    def apply_universal_es_update(
        self,
        normalized_fitnesses: list[float],
        template_ref: Any | None,
        es_step: int,
        args: Any,
    ) -> bool:
        from llm.vllm_worker_update import apply_universal_es_update as _apply_universal_es_update

        return _apply_universal_es_update(self, normalized_fitnesses, template_ref, es_step, args)

    def apply_subspace_perturbation(
        self,
        template: Any | None,
        seed: int,
        scale: float,
    ) -> bool:
        from llm.vllm_worker_update import apply_subspace_perturbation as _apply_subspace_perturbation

        return _apply_subspace_perturbation(self, template, seed, scale)

    def set_universal_subspace_template(self, template: Any) -> bool:
        # Cached on the vLLM worker process to avoid repeatedly pickling the
        # template on every perturb/unperturb call.
        import torch

        print(f"WORKER: Received universal template (search_dim={getattr(template, 'search_dim', 'unknown')})")
        self._universal_subspace_template = template
        # Also cache a name->Parameter lookup for faster coordinate updates.
        if not getattr(self, "_universal_named_parameters", None):
            print("WORKER: Building parameter lookup cache...")
            self._universal_named_parameters = dict(self.model_runner.model.named_parameters())
            print(f"WORKER: Cached {len(self._universal_named_parameters)} parameters.")

        # Group basis indices by parameter name for vectorized updates
        groups = {}
        for i in range(int(template.search_dim)):
            p_meta = template.parameters[template.basis_leaf[i]]
            groups.setdefault(p_meta.name, []).append(i)

        self._universal_update_groups = {
            name: {
                "subspace_indices": torch.tensor(indices, device=self.device, dtype=torch.long),
                "param_indices": torch.tensor([template.basis_index[i] for i in indices], device=self.device, dtype=torch.long),
                "signs": torch.tensor([template.basis_sign[i] for i in indices], device=self.device, dtype=torch.float32),
            }
            for name, indices in groups.items()
        }
        print(f"WORKER: Built vectorized update groups for {len(self._universal_update_groups)} parameters.")
        return True

    def discover_parameters(self) -> Any:
        from llm.lora import discover_vllm_parameters

        return discover_vllm_parameters(self.model_runner.model)

    def broadcast_all_weights(self, src_rank: int) -> bool:
        try:
            import torch
        except ImportError:
            return False

        if not getattr(self, "inter_pg", None):
            return False
        for _, param in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def get_model_state_dict(self) -> dict[str, Any]:
        return {name: param.detach().cpu().clone() for name, param in self.model_runner.model.named_parameters()}

    def set_model_state_dict(self, state_dict: dict[str, Any] | list[dict[str, Any]]) -> bool:
        import torch

        if isinstance(state_dict, list):
            state_dict = state_dict[_tensor_parallel_rank()]
        params = dict(self.model_runner.model.named_parameters())
        for name, value in state_dict.items():
            if name in params:
                params[name].data.copy_(value.to(params[name].device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True


def _tensor_parallel_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0
