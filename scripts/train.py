from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from robustvision.device import select_device
from robustvision.models.registry import create_model

parser = argparse.ArgumentParser(description="Train baseline CNN on CIFAR-10")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--data-dir", default="data")
parser.add_argument("--output", default="artifacts/checkpoints/baseline_cnn.pt")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

device = select_device()
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
)
data = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
if args.quick:
    data = torch.utils.data.Subset(data, range(min(512, len(data))))
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
model = create_model("baseline_cnn").to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    model.train()
    losses = []
    for x, y in loader:
        opt.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"epoch={epoch + 1} loss={sum(losses) / len(losses):.4f}")
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": "baseline_cnn", "state_dict": model.state_dict()}, out)
print(f"checkpoint={out}")
