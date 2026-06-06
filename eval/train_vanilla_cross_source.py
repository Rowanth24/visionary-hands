import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import videotransforms
from asl_backends import choose_device, extract_state_dict, filter_state_dict_for_model, infer_num_classes, strip_module_prefix
from custom_models import I3DFeatureExtractor, SignLanguageRecognitionModel
from datasets.mixed_sign_dataset import MixedSignDataset
from pytorch_i3d import InceptionI3d


def build_parser():
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fine-tune the vanilla ASL model on a mixed-source isolated-sign dataset."
    )
    parser.add_argument("--train-manifest", required=True, help="Path to the mixed-source training manifest.")
    parser.add_argument("--val-manifest", help="Path to the mixed-source validation manifest.")
    parser.add_argument("--video-dir", required=True, help="Directory containing local mixed-source videos.")
    parser.add_argument("--base-checkpoint", required=True, help="Path to the starting vanilla checkpoint.")
    parser.add_argument(
        "--labels-file",
        default=str(code_dir / "preprocess" / "wlasl_class_list.txt"),
        help="Path to the WLASL 1000-class label file.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--clip-length", type=int, default=64, help="Clip length.")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto", help="Training device.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=20260421, help="Random seed.")
    parser.add_argument("--output-dir", required=True, help="Directory to write checkpoints and training history.")
    parser.add_argument(
        "--train-scope",
        choices=("classifier", "transformer"),
        default="classifier",
        help="How much of the transformer stack to update during adaptation.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing applied to cross-entropy.",
    )
    parser.add_argument(
        "--anchor-l2",
        type=float,
        default=0.1,
        help="Regularization weight that keeps trainable parameters near the base checkpoint.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop after this many non-improving validation epochs. Disabled when no val manifest is provided.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=2,
        help="Minimum epochs to run before early stopping can trigger.",
    )
    return parser


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(checkpoint_path, device):
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_obj)
    num_classes = infer_num_classes(state_dict)

    i3d = InceptionI3d(1000, in_channels=3)
    model = SignLanguageRecognitionModel(I3DFeatureExtractor(i3d), num_classes)
    normalized_state = strip_module_prefix(state_dict)
    filtered_state, missing, unexpected, mismatched = filter_state_dict_for_model(model, normalized_state)
    if mismatched:
        raise RuntimeError(f"Checkpoint has mismatched tensor shapes: {mismatched[:5]}")
    if missing:
        raise RuntimeError(f"Checkpoint is missing required tensors: {missing[:10]}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} unexpected checkpoint tensors")
    model.load_state_dict(filtered_state, strict=True)

    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False

    model.to(device)
    return model, num_classes


def configure_trainable_params(model, scope):
    for param in model.transformer.parameters():
        param.requires_grad = False

    if scope == "classifier":
        trainable = list(model.transformer.classifier.parameters())
    else:
        trainable = list(model.transformer.parameters())

    for param in trainable:
        param.requires_grad = True

    return trainable


def capture_anchor_params(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def anchor_regularization(model, anchor_params):
    penalties = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        penalties.append((param - anchor_params[name]).pow(2).mean())
    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item(), preds


def evaluate(model, dataloader, criterion, device):
    model.eval()
    totals = {"all": {"loss": 0.0, "correct": 0, "count": 0}}

    with torch.no_grad():
        for inputs, labels, _video_names, sources in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            correct, _preds = compute_accuracy(logits, labels)

            batch_size = labels.size(0)
            totals["all"]["loss"] += loss.item() * batch_size
            totals["all"]["correct"] += correct
            totals["all"]["count"] += batch_size

            for source, is_correct in zip(sources, (logits.argmax(dim=1) == labels).detach().cpu().tolist()):
                totals.setdefault(source, {"loss": 0.0, "correct": 0, "count": 0})
                totals[source]["correct"] += int(is_correct)
                totals[source]["count"] += 1

    summary = {}
    for key, values in totals.items():
        count = max(1, values["count"])
        summary[key] = {
            "accuracy": values["correct"] / count,
            "correct": values["correct"],
            "count": values["count"],
            "loss": values["loss"] / count if key == "all" else None,
        }
    return summary


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transforms = transforms.Compose(
        [
            videotransforms.RandomCrop(224),
            videotransforms.RandomHorizontalFlip(),
        ]
    )
    val_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    train_dataset = MixedSignDataset(
        manifest_path=args.train_manifest,
        video_dir=args.video_dir,
        labels_path=args.labels_file,
        split="train",
        transforms=train_transforms,
        clip_length=args.clip_length,
    )
    val_dataset = None
    if args.val_manifest:
        val_dataset = MixedSignDataset(
            manifest_path=args.val_manifest,
            video_dir=args.video_dir,
            labels_path=args.labels_file,
            split="val",
            transforms=val_transforms,
            clip_length=args.clip_length,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    device = choose_device(args.device)
    print(f"Training on device: {device}")
    model, num_classes = build_model(args.base_checkpoint, device)
    print(f"Loaded base checkpoint: {args.base_checkpoint}")
    print(f"Model classes: {num_classes}")

    trainable_params = configure_trainable_params(model, args.train_scope)
    anchor_params = capture_anchor_params(model)
    print(f"Train scope: {args.train_scope} ({sum(param.numel() for param in trainable_params):,} parameters)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_metric = (-1.0, -1.0, -1.0)
    best_path = output_dir / "best_cross_source_model.pth"
    last_path = output_dir / "last_cross_source_model.pth"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_anchor_loss = 0.0
        running_correct = 0
        running_count = 0

        for batch_index, (inputs, labels, _video_names, _sources) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            ce_loss = criterion(logits, labels)
            anchor_loss = anchor_regularization(model, anchor_params)
            loss = ce_loss + (args.anchor_l2 * anchor_loss)
            loss.backward()
            optimizer.step()

            correct, _preds = compute_accuracy(logits, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_ce_loss += ce_loss.item() * batch_size
            running_anchor_loss += anchor_loss.item() * batch_size
            running_correct += correct
            running_count += batch_size

            if batch_index % 10 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"anchor={anchor_loss.item():.4f} acc={correct / batch_size:.3f}"
                )

        train_summary = {
            "loss": running_loss / max(1, running_count),
            "ce_loss": running_ce_loss / max(1, running_count),
            "anchor_loss": running_anchor_loss / max(1, running_count),
            "accuracy": running_correct / max(1, running_count),
            "count": running_count,
        }
        val_summary = evaluate(model, val_loader, criterion, device) if val_loader is not None else None
        record = {
            "epoch": epoch,
            "train": train_summary,
            "val": val_summary,
        }
        history.append(record)

        if val_summary is None:
            print(
                f"epoch={epoch} train_loss={train_summary['loss']:.4f} "
                f"train_ce={train_summary['ce_loss']:.4f} train_anchor={train_summary['anchor_loss']:.4f} "
                f"train_acc={train_summary['accuracy']:.3f}"
            )
        else:
            val_all = val_summary["all"]["accuracy"]
            val_aslense = val_summary.get("aslense", {}).get("accuracy", -1.0)
            val_wlasl = val_summary.get("wlasl", {}).get("accuracy", -1.0)
            metric = (val_all, val_aslense, val_wlasl)

            print(
                f"epoch={epoch} train_loss={train_summary['loss']:.4f} train_ce={train_summary['ce_loss']:.4f} "
                f"train_anchor={train_summary['anchor_loss']:.4f} train_acc={train_summary['accuracy']:.3f} "
                f"val_acc={val_all:.3f} val_wlasl={val_wlasl:.3f} val_aslense={val_aslense:.3f}"
            )

            if metric > best_metric:
                best_metric = metric
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_path)
                print(f"Saved improved checkpoint to {best_path}")
            else:
                epochs_without_improvement += 1
                print(f"No validation improvement for {epochs_without_improvement} epoch(s)")

            if epoch >= args.min_epochs and epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), last_path)
    if val_loader is None:
        best_path = last_path

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "best_metric": (
            {
                "combined_val_accuracy": best_metric[0],
                "aslense_val_accuracy": best_metric[1],
                "wlasl_val_accuracy": best_metric[2],
            }
            if val_loader is not None
            else None
        ),
        "history_path": str(history_path.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
