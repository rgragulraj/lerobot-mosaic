"""ResNet18 feature-similarity detector for grasp completion.

Loads "done" keyframes at startup, embeds them with a pretrained ResNet18,
and compares each new overhead frame against the mean done-embedding.

No training required — uses ImageNet features out of the box.

Usage:
    detector = GraspDoneDetector(Path("data/grasp_keyframes/done"))
    done, sim = detector.is_done(obs_raw["overhead"])  # frame is RGB numpy array
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from PIL import Image
from torchvision import models, transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)


class GraspDoneDetector:
    """Detects grasp completion via cosine similarity to pre-collected done frames."""

    def __init__(self, done_dir: Path, threshold: float = 0.85, device: str = "cpu"):
        self.threshold = threshold
        self.device = torch.device(device)

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone.eval()
        self.model = backbone.to(self.device)

        done_dir = Path(done_dir)
        overhead_images = sorted(done_dir.glob("*_overhead.jpg"))
        if not overhead_images:
            raise FileNotFoundError(f"No *_overhead.jpg files found in {done_dir.resolve()}")

        print(f"GraspDoneDetector: loading {len(overhead_images)} done-keyframe(s) from {done_dir.resolve()}")

        embeddings = []
        with torch.inference_mode():
            for path in overhead_images:
                img = Image.open(path).convert("RGB")
                tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)
                emb = self.model(tensor)
                embeddings.append(emb)

        mean_emb = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
        self.mean_embedding = func.normalize(mean_emb, dim=1)

    def is_done(self, frame: np.ndarray) -> tuple[bool, float]:
        """Check if the current overhead frame matches the done state.

        Args:
            frame: RGB numpy array from obs_raw["overhead"].

        Returns:
            (done, similarity) where similarity is 0.0–1.0.
        """
        img = Image.fromarray(frame.astype(np.uint8))
        tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            emb = self.model(tensor)
            emb = func.normalize(emb, dim=1)
            sim = func.cosine_similarity(emb, self.mean_embedding, dim=1).item()

        return sim >= self.threshold, sim
