"""Fashion CLIP image preprocessing and batch helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224


def build_transform(augment: bool = False) -> transforms.Compose:
    """
    augment=False -> deterministic preprocessing for inference/embedding.
    augment=True -> data augmentation pipeline for fine-tuning.
    """
    base = [
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
    ]

    if augment:
        base = [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]

    return transforms.Compose([
        *base,
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def _to_rgb(image: Any, remove_background: bool = False) -> Any:
    if image.mode in ("RGBA", "P", "LA"):
        if remove_background:
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[-1])
            return background
        return image.convert("RGB")

    if image.mode != "RGB":
        return image.convert("RGB")

    return image


def preprocess_pil_image(
    image: Any,
    transform: transforms.Compose | None = None,
    remove_background: bool = False,
) -> torch.Tensor | None:
    """Preprocess an in-memory PIL image and return (1, C, H, W)."""
    if transform is None:
        transform = build_transform(augment=False)

    try:
        img = _to_rgb(image, remove_background=remove_background)
        tensor = transform(img)
        return tensor.unsqueeze(0)
    except (UnidentifiedImageError, OSError) as exc:
        print(f"[WARN] Ignoring image in memory: {exc}")
        return None


def preprocess_image(
    image_path: str | Path,
    transform: transforms.Compose | None = None,
    remove_background: bool = False,
) -> torch.Tensor | None:
    """Load and preprocess a single image from disk."""
    if transform is None:
        transform = build_transform(augment=False)

    try:
        img = Image.open(image_path)
        return preprocess_pil_image(
            image=img,
            transform=transform,
            remove_background=remove_background,
        )
    except (UnidentifiedImageError, FileNotFoundError, OSError) as exc:
        print(f"[WARN] Ignoring {image_path}: {exc}")
        return None


class FashionImageDataset(Dataset):
    """Simple dataset to load images from a directory or explicit file list."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        source: str | Path | list[str | Path],
        augment: bool = False,
    ):
        if isinstance(source, list):
            self.paths = [Path(p) for p in source]
        else:
            root = Path(source)
            self.paths = [
                p for p in sorted(root.rglob("*")) if p.suffix.lower() in self.VALID_EXTENSIONS
            ]

        self.transform = build_transform(augment=augment)
        print(f"Dataset: {len(self.paths)} images found.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        tensor = preprocess_image(path, self.transform)

        if tensor is None:
            tensor = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)

        return {"pixel_values": tensor.squeeze(0), "path": str(path)}


def generate_embeddings(
    model: Any,
    dataset: FashionImageDataset,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Iterate the dataset and return (normalized embeddings, paths)."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    model = model.to(device).eval()

    all_embeddings: list[torch.Tensor] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.get_image_features(pixel_values=pixel_values)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu())
            all_paths.extend(batch["path"])

    return torch.cat(all_embeddings, dim=0), all_paths
