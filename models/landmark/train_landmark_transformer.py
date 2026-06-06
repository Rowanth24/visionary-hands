#!/usr/bin/env python3
"""Train a lightweight landmark classifier on prepared WLASL arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from landmark_baseline.features import append_frame_deltas_torch
from landmark_baseline.model import build_landmark_model


class LandmarkSplitDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        augment: bool = False,
        input_noise_std: float = 0.0,
        frame_dropout: float = 0.0,
        feature_dropout: float = 0.0,
        append_deltas: bool = False,
    ) -> None:
        payload = np.load(npz_path)
        x = payload['X'].astype(np.float32)
        y = payload['y'].astype(np.int64)
        non_empty = payload['non_empty_frame_idxs'].astype(np.float32)
        self.x = torch.from_numpy(x.reshape(x.shape[0], x.shape[1], -1))
        if append_deltas:
            self.x = append_frame_deltas_torch(self.x)
        self.y = torch.from_numpy(y)
        self.valid_mask = torch.from_numpy(non_empty >= 0)
        self.augment = bool(augment)
        self.input_noise_std = float(max(0.0, input_noise_std))
        self.frame_dropout = float(min(1.0, max(0.0, frame_dropout)))
        self.feature_dropout = float(min(1.0, max(0.0, feature_dropout)))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = self.x[idx]
        valid_mask = self.valid_mask[idx]
        if not self.augment:
            return x, valid_mask, self.y[idx]

        x = x.clone()
        valid_mask = valid_mask.clone()
        valid = valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        if self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std * valid

        if self.feature_dropout > 0.0:
            keep = (torch.rand(x.shape[-1]) > self.feature_dropout).to(dtype=x.dtype)
            x = x * keep.unsqueeze(0)

        if self.frame_dropout > 0.0:
            valid_idxs = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if valid_idxs.numel() > 1:
                drop = torch.rand(valid_idxs.numel()) < self.frame_dropout
                if bool(torch.all(drop)):
                    drop[torch.randint(0, valid_idxs.numel(), (1,))] = False
                drop_idxs = valid_idxs[drop]
                if drop_idxs.numel() > 0:
                    x[drop_idxs] = 0.0
                    valid_mask[drop_idxs] = False

        return x, valid_mask, self.y[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model-type', type=str, choices=('transformer', 'gru'), default='transformer')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gru-hidden-size', type=int, default=96)
    parser.add_argument('--gru-unidirectional', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--input-noise-std', type=float, default=0.0)
    parser.add_argument('--frame-dropout', type=float, default=0.0)
    parser.add_argument('--feature-dropout', type=float, default=0.0)
    parser.add_argument('--append-deltas', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--export-torchscript', action='store_true')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for x, valid_mask, y in loader:
            x = x.to(device)
            valid_mask = valid_mask.to(device)
            y = y.to(device)
            logits = model(x, valid_mask)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += int(y.size(0))
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    labels = json.loads((args.data_root / 'labels.json').read_text(encoding='utf-8'))
    data_summary_path = args.data_root / 'summary.json'
    data_summary = json.loads(data_summary_path.read_text(encoding='utf-8')) if data_summary_path.exists() else {}
    train_ds = LandmarkSplitDataset(
        args.data_root / 'train' / 'dataset.npz',
        augment=(args.input_noise_std > 0.0 or args.frame_dropout > 0.0 or args.feature_dropout > 0.0),
        input_noise_std=args.input_noise_std,
        frame_dropout=args.frame_dropout,
        feature_dropout=args.feature_dropout,
        append_deltas=args.append_deltas,
    )
    val_ds = LandmarkSplitDataset(args.data_root / 'val' / 'dataset.npz', append_deltas=args.append_deltas)
    test_ds = LandmarkSplitDataset(args.data_root / 'test' / 'dataset.npz', append_deltas=args.append_deltas)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_landmark_model(
        model_type=args.model_type,
        feature_dim=int(train_ds.x.shape[-1]),
        num_classes=len(labels),
        max_frames=int(train_ds.x.shape[1]),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gru_hidden_size=args.gru_hidden_size,
        gru_bidirectional=not args.gru_unidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    run_config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'gru_hidden_size': args.gru_hidden_size,
        'gru_bidirectional': not args.gru_unidirectional,
        'label_smoothing': args.label_smoothing,
        'input_noise_std': args.input_noise_std,
        'frame_dropout': args.frame_dropout,
        'feature_dropout': args.feature_dropout,
        'append_deltas': args.append_deltas,
        'target_frames': data_summary.get('target_frames'),
        'xy_only': data_summary.get('xy_only'),
        'layout': data_summary.get('layout', 'dominant_hand'),
        'seed': args.seed,
        'device': device.type,
    }
    (args.output_root / 'config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')

    best_val = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for x, valid_mask, y in train_loader:
            x = x.to(device)
            valid_mask = valid_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, valid_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += int(y.size(0))
        scheduler.step()

        train_loss = total_loss / max(1, total_count)
        train_acc = total_correct / max(1, total_count)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'labels': labels,
                    'history': history,
                    'config': run_config,
                },
                args.output_root / 'best.pt',
            )

    best_payload = torch.load(args.output_root / 'best.pt', map_location=device)
    model.load_state_dict(best_payload['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    metrics = {
        'best_val_acc': best_val,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'epochs': args.epochs,
        'model_type': args.model_type,
        'num_classes': len(labels),
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds),
        'device': device.type,
        'history': history,
    }
    (args.output_root / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    if args.export_torchscript:
        export_model = build_landmark_model(
            model_type=args.model_type,
            feature_dim=int(train_ds.x.shape[-1]),
            num_classes=len(labels),
            max_frames=int(train_ds.x.shape[1]),
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            gru_hidden_size=args.gru_hidden_size,
            gru_bidirectional=not args.gru_unidirectional,
        ).cpu()
        cpu_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        export_model.load_state_dict(cpu_state_dict)
        export_model.eval()
        example_x = train_ds.x[:1].cpu()
        example_mask = train_ds.valid_mask[:1].cpu()
        try:
            exported = torch.jit.script(export_model)
        except Exception:
            exported = torch.jit.trace(export_model, (example_x, example_mask), check_trace=False)
        exported.save(str(args.output_root / 'model.ts'))

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
