"""Central pydantic Settings. All components read from here — never env-var directly."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql://srpost:srpost@localhost:5432/srpost"
    redis_url: str = "redis://localhost:6379/0"

    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

    embedder_model: str = "google/siglip2-so400m-patch14-384"
    embedding_dim: int = 1152

    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "srpost"
    minio_root_password: str = "srpost"
    minio_bucket: str = "srpost"

    hls_volume_root: str = "/data/hls"
    hls_public_base: str = "http://localhost/hls"

    top_k_default: int = 5

    # Device mapping (host GPU indices).
    # 4 GPU layout:
    #   GPU_VLLM=6 : Qwen2.5-VL-7B-AWQ inference (30GB VRAM)
    #   GPU_WORKER=7 : SigLIP2 + arq worker (10GB)
    #   GPU_TRAIN_A=4 : training / LoRA (36GB headroom for Qwen-VL LoRA)
    #   GPU_TRAIN_B=5 : training / LoRA (SigLIP2 + YOLO fine-tune)
    device_api: Literal["cpu", "cuda:0"] = "cpu"
    device_worker: Literal["cpu", "cuda:0"] = "cuda:0"

    # Adapter / checkpoint loading (auto-loaded at startup if present)
    siglip2_adapter_path: str | None = None        # Action retriever LoRA (frame ↔ caption)
    siglip2_reid_adapter_path: str | None = None   # Person ReID LoRA (crop ↔ identity/attribute)
    qwen_vl_lora_path: str | None = None           # served via vLLM --enable-lora
    yolo_weights_path: str | None = None           # person-focused YOLO
    videomae_ckpt_path: str | None = None

    # Scene detection defaults (PySceneDetect AdaptiveDetector)
    # Higher threshold = less sensitive; higher min_scene_len = longer scenes
    scene_adaptive_threshold: float = 3.5
    scene_min_scene_len: int = 30   # frames. ~1.0s at 30fps

    # Adjacent-scene merging: if two consecutive scenes have cosine(emb_a, emb_b) > this,
    # merge them into one (post-process to fight over-segmentation).
    # Higher = more conservative (fewer merges, keep scene granularity).
    scene_merge_cosine: float = 0.85
    # Hard cap on merged scene length (sec) — prevents runaway mergers in CCTV/static camera
    scene_max_merged_length: float = 30.0
    # Also enforce on RAW PySceneDetect output: any scene longer than this gets uniformly split
    scene_max_raw_length: float = 20.0

    # Multi-frame scene representation
    frames_per_scene: int = 3  # K uniformly-sampled frames, mean-pooled into 1 embedding

    # Search minimum score — hits below this cosine get filtered from API output
    search_min_score: float = 0.05

    # Ingest / caption
    caption_prompt: str = (
        "Describe the main action and objects in this frame in one sentence."
    )
    vlm_max_new_tokens: int = 128
    vlm_retries: int = 3
    vlm_timeout_s: float = 30.0

    # FastAPI
    api_prefix: str = "/api"

    # Derived
    @property
    def minio_secure(self) -> bool:
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
