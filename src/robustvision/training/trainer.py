from __future__ import annotations

import time

import torch

from robustvision.evaluation.runner import evaluate


def train(
    model,
    train_loader,
    validation_loader,
    device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_state, best_score, history = None, -1.0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(labels)
        validation = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / len(train_loader.dataset),
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        if validation["accuracy"] > best_score:
            best_score = validation["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "best_validation_accuracy": best_score,
    }, evaluate(model, validation_loader, device)
