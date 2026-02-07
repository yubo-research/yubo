#!/usr/bin/env python

import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from problems.mnist_classifier import MnistClassifier

TIMEOUT_SECONDS = 3 * 60


def _train_one_epoch(model, train_loader, optimizer, loss_fn, device, *, deadline, scheduler=None):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        if time.monotonic() >= deadline:
            return None
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def _eval_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(test_loader.dataset)


def fit_mnist(*, num_epochs=3, batch_size=1024, lr=8e-3, weight_decay=1e-2, device=None, timeout_seconds=TIMEOUT_SECONDS):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = MnistClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)

    deadline = time.monotonic() + timeout_seconds
    for epoch in range(1, num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device, deadline=deadline, scheduler=scheduler)
        if train_loss is None:
            print(f"timeout after {timeout_seconds}s (during epoch {epoch})")
            break
        accuracy = _eval_accuracy(model, test_loader, device)
        print(f"epoch {epoch}/{num_epochs}  train_loss={train_loss:.4f}  test_acc={accuracy:.4f}")
        if time.monotonic() >= deadline:
            print(f"timeout after {timeout_seconds}s (after epoch {epoch})")
            break

    return model


if __name__ == "__main__":
    fit_mnist()
