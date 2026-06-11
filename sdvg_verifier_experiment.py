from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sdvg_inference import (
    build_pipeline,
    commit_clean_block,
    denoise_block,
    ensure_wan_symlinks,
    load_config,
    reset_pipeline_cache,
)
from sdvg_latent_verifier import LatentBlockVerifier, verifier_features
from utils.misc import set_seed
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


def serializable_args(args) -> dict:
    return {
        key: value
        for key, value in vars(args).items()
        if not callable(value)
    }


def load_prompt_file(path: str, start_index: int, max_prompts: int) -> list[tuple[int, str]]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_index or not line.strip():
                continue
            prompts.append((idx, line.strip()))
            if 0 < max_prompts <= len(prompts):
                break
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def collect_pairs(args):
    os.chdir(Path(__file__).parent)
    ensure_wan_symlinks(args.model_root)
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config_path)
    config.denoising_step_list = [1000, 750, 500, 250, 0]
    config.warp_denoising_step = False
    config.num_frame_per_block = 3

    device = torch.device("cuda")
    dtype = torch.bfloat16
    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)

    print("Loading target:", args.target_model_name, args.target_checkpoint_path)
    target_pipeline = build_pipeline(
        config,
        args.target_model_name,
        args.target_checkpoint_path,
        device,
        dtype,
        text_encoder=text_encoder,
        vae=vae,
        use_ema=False,
    )
    print("Loading drafter:", args.draft_model_name, args.draft_checkpoint_path)
    draft_pipeline = build_pipeline(
        config,
        args.draft_model_name,
        args.draft_checkpoint_path,
        device,
        dtype,
        text_encoder=text_encoder,
        vae=vae,
        use_ema=args.use_ema,
    )

    prompts = load_prompt_file(args.prompt_file, args.start_index, args.max_prompts)

    features = []
    labels = []
    metadata = []
    latent_shape = [args.num_blocks * 3, 16, 60, 104]

    for prompt_index, prompt in tqdm(prompts, desc="collect prompts"):
        generator = torch.Generator(device=device).manual_seed(args.seed + prompt_index)
        noise = torch.randn([1, *latent_shape], device=device, dtype=dtype, generator=generator)
        committed = torch.zeros_like(noise)

        reset_pipeline_cache(target_pipeline, 1, dtype, device, total_frames=noise.shape[1])
        reset_pipeline_cache(draft_pipeline, 1, dtype, device, total_frames=noise.shape[1])
        target_cond = target_pipeline.text_encoder([prompt])
        draft_cond = target_cond if draft_pipeline.text_encoder is target_pipeline.text_encoder else draft_pipeline.text_encoder([prompt])

        for block_index in range(args.num_blocks):
            start = block_index * 3
            end = start + 3
            block_noise = noise[:, start:end]
            context = committed[:, :start] if start > 0 else None

            target_latents = denoise_block(target_pipeline, block_noise, target_cond, start)

            if block_index > 0:
                draft_latents = denoise_block(draft_pipeline, block_noise, draft_cond, start)
                for source, latents, label in (
                    ("target", target_latents, 1.0),
                    ("draft", draft_latents, 0.0),
                ):
                    feature = verifier_features(latents.detach(), context.detach() if context is not None else None, block_index, args.num_blocks)
                    features.append(feature.cpu())
                    labels.append(torch.tensor([label], dtype=torch.float32))
                    metadata.append(
                        {
                            "prompt_index": prompt_index,
                            "block_index": block_index,
                            "source": source,
                            "label": label,
                            "prompt": prompt,
                        }
                    )

            committed[:, start:end] = target_latents
            commit_clean_block(target_pipeline, target_latents, target_cond, start)
            commit_clean_block(draft_pipeline, target_latents, draft_cond, start)

    payload = {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "metadata": metadata,
        "args": serializable_args(args),
    }
    torch.save(payload, output_path)
    print(f"Wrote {len(labels)} verifier samples to {output_path}")


def train_verifier(args):
    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    features = data["features"].float()
    labels = data["labels"].float()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = LatentBlockVerifier(features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch_features.shape[0]
            preds = (logits.detach() > 0).float()
            total_correct += int((preds == batch_labels).sum().item())
            total += batch_features.shape[0]

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={total_loss / total:.6f} acc={total_correct / total:.4f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "hidden_dim": args.hidden_dim,
            "feature_mean": features.mean(dim=0),
            "feature_std": features.std(dim=0, unbiased=False).clamp_min(1e-6),
            "train_args": serializable_args(args),
            "data_args": data.get("args", {}),
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "samples": int(features.shape[0]),
                "input_dim": int(features.shape[1]),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    print(f"Wrote verifier checkpoint: {output_path}")


def train_acceptance_verifier(args):
    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    pairs = data["pairs"]
    if not pairs:
        raise ValueError(f"No target-regeneration pairs found in {args.data_path}")

    features = []
    labels = []
    metadata = []
    num_blocks = int(data.get("args", {}).get("num_blocks", args.num_blocks))
    threshold = args.tau_delta
    if threshold is None:
        threshold = float(pairs[0].get("threshold"))

    for pair in tqdm(pairs, desc="build acceptance features"):
        context_latents = pair.get("context_latents")
        feature = verifier_features(
            pair["draft_latents"],
            context_latents,
            int(pair["block_index"]),
            num_blocks,
        )
        label = float(pair["delta"] < threshold)
        features.append(feature.cpu())
        labels.append(label)
        metadata.append(
            {
                "prompt_index": pair.get("prompt_index"),
                "block_index": pair["block_index"],
                "delta": pair["delta"],
                "threshold": threshold,
                "label": label,
                "prompt": pair.get("prompt"),
            }
        )

    features = torch.cat(features, dim=0).float()
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = LatentBlockVerifier(features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total = 0
        progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for batch_features, batch_labels in progress:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch_features.shape[0]
            preds = (logits.detach() > 0).float()
            total_correct += int((preds == batch_labels).sum().item())
            total += batch_features.shape[0]
            progress.set_postfix(loss=total_loss / max(1, total), acc=total_correct / max(1, total))

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={total_loss / total:.6f} acc={total_correct / total:.4f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "hidden_dim": args.hidden_dim,
            "feature_mean": features.mean(dim=0),
            "feature_std": features.std(dim=0, unbiased=False).clamp_min(1e-6),
            "train_args": serializable_args(args),
            "data_args": data.get("args", {}),
            "label_threshold": threshold,
            "label_semantics": "1 means draft-target agreement delta is below label_threshold",
        },
        output_path,
    )
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "samples": int(features.shape[0]),
                "positives": int(labels.sum().item()),
                "positive_rate": float(labels.mean().item()),
                "input_dim": int(features.shape[1]),
                "tau_delta": threshold,
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    print(f"Wrote acceptance verifier checkpoint: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect and overfit a latent target-vs-draft verifier for SDVG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    collect.add_argument("--model_root", default="/mnt/lanxiangh/models")
    collect.add_argument("--draft_model_name", default="Wan2.1-T2V-1.3B")
    collect.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    collect.add_argument("--draft_checkpoint_path", default="/mnt/lanxiangh/models/Self-Forcing/checkpoints/self_forcing_dmd.pt")
    collect.add_argument("--target_checkpoint_path", default="/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors")
    collect.add_argument("--prompt_file", default="prompts/MovieGenVideoBench.txt")
    collect.add_argument("--start_index", type=int, default=0)
    collect.add_argument("--max_prompts", type=int, default=2)
    collect.add_argument("--num_blocks", type=int, default=3)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--use_ema", action="store_true", default=True)
    collect.add_argument("--output_path", default="outputs/sdvg/verifier/moviegen_pairs.pt")
    collect.set_defaults(func=collect_pairs)

    train = subparsers.add_parser("train")
    train.add_argument("--data_path", default="outputs/sdvg/verifier/moviegen_pairs.pt")
    train.add_argument("--output_path", default="outputs/sdvg/verifier/moviegen_verifier.pt")
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--hidden_dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=0.0)
    train.add_argument("--log_every", type=int, default=25)
    train.add_argument("--cpu", action="store_true")
    train.set_defaults(func=train_verifier)

    train_accept = subparsers.add_parser("train_acceptance")
    train_accept.add_argument("--data_path", default="outputs/sdvg/verifier/target_regen_pairs.pt")
    train_accept.add_argument("--output_path", default="outputs/sdvg/verifier/moviegen_acceptance_verifier.pt")
    train_accept.add_argument("--tau_delta", type=float, default=None)
    train_accept.add_argument("--num_blocks", type=int, default=9)
    train_accept.add_argument("--epochs", type=int, default=500)
    train_accept.add_argument("--batch_size", type=int, default=32)
    train_accept.add_argument("--hidden_dim", type=int, default=256)
    train_accept.add_argument("--dropout", type=float, default=0.0)
    train_accept.add_argument("--lr", type=float, default=1e-3)
    train_accept.add_argument("--weight_decay", type=float, default=0.0)
    train_accept.add_argument("--log_every", type=int, default=25)
    train_accept.add_argument("--cpu", action="store_true")
    train_accept.set_defaults(func=train_acceptance_verifier)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
