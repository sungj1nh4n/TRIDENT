"""SigLIP2 embedder with two modes:
  - text_cpu : api 컨테이너에서 쿼리 텍스트 임베딩(가벼움, GPU 필요 없음)
  - full_gpu : worker에서 이미지/텍스트 모두(GPU 7에 로드, 컨테이너 내부 cuda:0)

validate_dim()은 full_gpu 모드에서 vision_config hidden_size 검증에 사용.
text_cpu는 config만 로드돼 있으면 OK.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from PIL import Image

from srpost.config import settings

logger = logging.getLogger(__name__)

EmbedderMode = Literal["text_cpu", "full_gpu", "image_cpu"]

_ADAPTER_SENTINEL = object()


class Embedder:
    def __init__(
        self,
        mode: EmbedderMode = "text_cpu",
        *,
        adapter_path: object = _ADAPTER_SENTINEL,
    ) -> None:
        """SigLIP2 embedder.

        adapter_path:
            - default sentinel -> use settings.siglip2_adapter_path (Action LoRA)
            - None            -> no adapter (base model)
            - str path        -> explicit adapter (e.g. siglip2_reid_adapter_path)
        """
        self.mode: EmbedderMode = mode
        self._model = None
        self._processor = None
        device_str = "cpu" if mode in ("text_cpu", "image_cpu") else settings.device_worker
        self._device: torch.device = torch.device(device_str)
        self._adapter_path: str | None
        if adapter_path is _ADAPTER_SENTINEL:
            self._adapter_path = settings.siglip2_adapter_path
        else:
            self._adapter_path = adapter_path  # type: ignore[assignment]

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading embedder %s on %s (mode=%s, adapter=%s)",
                    settings.embedder_model, self._device, self.mode, self._adapter_path)
        self._processor = AutoProcessor.from_pretrained(settings.embedder_model)
        dtype = torch.float32 if self.mode == "text_cpu" else torch.float16
        base = AutoModel.from_pretrained(
            settings.embedder_model, torch_dtype=dtype
        ).to(self._device).eval()

        adapter_path = self._adapter_path
        if adapter_path:
            from pathlib import Path as _Path
            if _Path(adapter_path).exists():
                try:
                    from peft import PeftModel
                    logger.info("Loading SigLIP2 LoRA adapter from %s", adapter_path)
                    base = PeftModel.from_pretrained(base, adapter_path, torch_dtype=dtype)
                    base = base.eval()
                except Exception as e:  # noqa: BLE001
                    logger.warning("PEFT adapter load failed: %s — using base model", e)
            else:
                logger.warning("adapter_path=%s does not exist — using base", adapter_path)

        self._model = base

    @property
    def model(self):
        self._lazy_load()
        return self._model

    @property
    def processor(self):
        self._lazy_load()
        return self._processor

    def validate_dim(self) -> int:
        """Assert model vision hidden_size matches settings.embedding_dim.

        Requires full model loaded (config.vision_config must exist).
        Returns the actual dim on success; raises AssertionError otherwise.
        """
        actual = self.model.config.vision_config.hidden_size
        expected = settings.embedding_dim
        assert actual == expected, (
            f"Embedder dim mismatch: model={actual} settings={expected}. "
            "Update settings.embedding_dim + sql/001_init.sql vector(...) column."
        )
        return int(actual)

    @torch.inference_mode()
    def embed_text(self, text: str) -> np.ndarray:
        # SigLIP2 text encoder has max_position_embeddings=64; long captions must be truncated.
        max_len = int(
            getattr(self.model.config.text_config, "max_position_embeddings", 64)
        )
        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self._device)
        out = self.model.get_text_features(**inputs)
        vec = out[0].detach().cpu().float().numpy()
        return vec

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        if self.mode == "text_cpu":
            raise RuntimeError("embed_image requires mode='full_gpu' or 'image_cpu'")
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt").to(self._device)
        out = self.model.get_image_features(**inputs)
        vec = out[0].detach().cpu().float().numpy()
        return vec

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed N images, return (N, D) array (no pooling, not normalized)."""
        if self.mode == "text_cpu":
            raise RuntimeError("embed_images requires mode='full_gpu' or 'image_cpu'")
        if not images:
            raise ValueError("empty image list")
        rgb = [img.convert("RGB") for img in images]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self._device)
        feats = self.model.get_image_features(**inputs)  # (N, D)
        return feats.detach().cpu().float().numpy()

    @torch.inference_mode()
    def embed_images_mean(self, images: list[Image.Image]) -> np.ndarray:
        """Embed K frames and mean-pool into one scene vector (L2-normalized)."""
        if self.mode == "text_cpu":
            raise RuntimeError("embed_images_mean requires mode='full_gpu' or 'image_cpu'")
        if not images:
            raise ValueError("empty image list")
        rgb = [img.convert("RGB") for img in images]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self._device)
        feats = self.model.get_image_features(**inputs)  # (K, D)
        pooled = feats.mean(dim=0)
        # L2 normalize so cosine is directly dot product; scene-level vector
        pooled = pooled / (pooled.norm() + 1e-8)
        return pooled.detach().cpu().float().numpy()
