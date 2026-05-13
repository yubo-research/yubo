from __future__ import annotations

from typing import Any

from llm.vllm_net import get_ip, get_open_port
from llm.vllm_worker_update import apply_lora_es_update as _apply_lora_es_update


class WorkerExtension:
    def get_transport_info(self) -> tuple[str, int]:
        return get_ip(), get_open_port()

    def init_inter_engine_group(self, master_address: str, master_port: int, gpu_rank: int, world_size: int) -> bool:
        self.device = self.model_runner.device
        self.gpu_rank = int(gpu_rank)
        self.world_size = int(world_size)
        self.inter_pg = None
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup
        except ImportError:
            return True

        if get_tensor_model_parallel_rank() == 0:
            pg = StatelessProcessGroup.create(
                host=master_address,
                port=int(master_port),
                rank=int(gpu_rank),
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

    def set_model_state_dict(self, state_dict: dict[str, Any]) -> bool:
        import torch

        params = dict(self.model_runner.model.named_parameters())
        for name, value in state_dict.items():
            if name in params:
                params[name].data.copy_(value.to(params[name].device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
