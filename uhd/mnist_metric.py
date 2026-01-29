import torch
from torch import Tensor, nn


class MNISTMetric(nn.Module):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        seed: int,
        train: bool = True,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        assert isinstance(data_root, str) and len(data_root) > 0
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(seed, int)
        assert isinstance(train, bool)
        assert isinstance(num_workers, int) and num_workers >= 0
        from torchvision import datasets, transforms

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        ds = datasets.MNIST(
            root=data_root, train=train, download=True, transform=transform
        )
        pin = torch.cuda.is_available()
        self.loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=num_workers,
            pin_memory=pin,
        )
        self._it = iter(self.loader)
        self.criterion = nn.CrossEntropyLoss()

    def measure(self, controller: nn.Module, full_dataset: bool = False) -> Tensor:
        assert isinstance(controller, nn.Module)
        device = next(controller.parameters()).device

        if not full_dataset:
            try:
                xb, yb = next(self._it)
            except StopIteration:
                self._it = iter(self.loader)
                xb, yb = next(self._it)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = controller(xb)
            loss = self.criterion(logits, yb)
            return -loss

        total_loss = None
        total_examples = 0
        for i_batch, (xb, yb) in enumerate(self.loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = controller(xb)
            loss = self.criterion(logits, yb)
            weight = yb.size(0)
            if total_loss is None:
                total_loss = loss * weight
            else:
                total_loss = total_loss + loss * weight
            total_examples += weight
        mean_loss = total_loss / float(total_examples)
        return -mean_loss
