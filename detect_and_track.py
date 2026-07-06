"""영상 인물 감지 + BoT-SORT 트래킹 + (선택) ReID 인물 매칭 — 독립 실행 스크립트.

DB/API 없이 단일 영상 파일만으로 다음을 수행하는 오프라인 러너입니다.
운영 코드(srpost.models.person_tracker.PersonTracker)의 핵심 경로를 그대로
축약해, 등록·검증용으로 곧바로 돌려 볼 수 있게 만든 진입점입니다.

파이프라인
----------
  1) YOLO(class 0=person) + BoT-SORT 로 영상 전체를 스트리밍 추적 →
     track_id 별 (frame_idx, t, bbox, conf) 시퀀스 생성.
  2) min_track_len 미만의 짧은 트랙(YOLO 깜빡임)은 제거.
  3) (선택) --query-image 가 주어지면 SigLIP2-ReID 로 기준 인물 임베딩을
     만들고, 각 트랙에서 N개 크롭을 뽑아 평균 코사인 유사도로 정렬 →
     가장 닮은 인물 트랙(들)을 반환.
  4) 결과 트랙/궤적을 JSON 으로 저장.

사용 예
-------
  # 감지 + 트래킹만 (모든 인물 트랙 추출)
  python detect_and_track.py --video clip.mp4 --out out.json

  # 특정 인물 사진으로 그 사람만 골라 궤적 추적
  python detect_and_track.py --video clip.mp4 --query-image ref.jpg \
         --top-k 1 --out person1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _l2(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


def botsort_video(yolo_model, device, clip: str, imgsz: int, conf: float,
                  iou: float, vid_stride: int):
    """YOLO + BoT-SORT 로 영상 전체를 추적해 track_id → detection 리스트를 만든다.

    Ultralytics 의 track()이 Kalman 모션 모델 + 외형 특징 + IoU 로 프레임 간
    동일 인물을 하나의 track_id 로 묶어 준다. 각 detection 은
    (frame_idx, t_sec, (x1,y1,x2,y2), conf) 튜플.
    """
    import cv2

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracks: dict[int, list] = {}
    results = yolo_model.track(
        source=clip, persist=True, tracker="botsort.yaml",
        classes=[0], conf=conf, iou=iou, imgsz=imgsz,
        device=device, stream=True, vid_stride=vid_stride, verbose=False,
    )
    sample_idx = 0
    last_log = time.monotonic()
    for r in results:
        actual_idx = sample_idx * vid_stride
        if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ts = actual_idx / fps
            for tid, box, c in zip(ids, xyxy, confs):
                tracks.setdefault(int(tid), []).append((
                    int(actual_idx), float(ts),
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    float(c),
                ))
        sample_idx += 1
        if time.monotonic() - last_log > 5.0:
            print(f"  [botsort] frame={sample_idx} t={actual_idx / fps:.1f}s "
                  f"tracks={len(tracks)}")
            last_log = time.monotonic()
    return tracks, fps, total, W, H


def batched_reid_sims(vr, tracks, tids, anchor_emb, reid, n_samples):
    """각 트랙에서 conf 상위 N개 크롭을 뽑아 배치 임베딩 → 앵커와 평균 코사인 유사도."""
    from PIL import Image

    reqs = []  # (frame_idx, bbox, tid)
    for tid in tids:
        dets = tracks[tid]
        if not dets:
            continue
        top = sorted(dets, key=lambda d: d[3], reverse=True)[: max(n_samples * 3, 10)]
        top.sort(key=lambda d: d[0])
        chosen = top if len(top) <= n_samples else [
            top[int(i * len(top) / n_samples)] for i in range(n_samples)
        ]
        for fi, _ts, bb, _c in chosen:
            reqs.append((int(fi), bb, tid))

    crops, owners = [], []
    n_frames = len(vr)
    uniq = sorted({min(max(0, fi), n_frames - 1) for fi, _, _ in reqs})
    frames = {}
    try:
        batch = vr.get_batch(uniq).asnumpy()
        for j, fi in enumerate(uniq):
            frames[fi] = batch[j]
    except Exception:  # noqa: BLE001
        for fi in uniq:
            frames[fi] = vr[fi].asnumpy()
    for fi, bb, tid in reqs:
        arr = frames.get(min(max(0, fi), n_frames - 1))
        if arr is None:
            continue
        H, W = arr.shape[:2]
        x1, y1 = max(0, int(bb[0])), max(0, int(bb[1]))
        x2, y2 = min(W, int(bb[2])), min(H, int(bb[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(Image.fromarray(arr[y1:y2, x1:x2]))
        owners.append(tid)

    out = {tid: 0.0 for tid in tids}
    if not crops:
        return out
    embs = _l2(reid.embed_images(crops))
    sims = embs @ anchor_emb
    counts = {tid: 0 for tid in tids}
    for owner, s in zip(owners, sims.tolist()):
        out[owner] += float(s)
        counts[owner] += 1
    for tid in out:
        if counts[tid]:
            out[tid] /= counts[tid]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="영상 인물 감지 + 트래킹 + (선택) ReID 매칭")
    ap.add_argument("--video", required=True, help="입력 영상 경로")
    ap.add_argument("--out", default="tracks.json", help="결과 JSON 경로")
    ap.add_argument("--query-image", default=None,
                    help="특정 인물 사진 (주면 그 사람만 골라 궤적 추적)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--vid-stride", type=int, default=2, help="N프레임마다 1장 처리")
    ap.add_argument("--min-track-len", type=int, default=5)
    ap.add_argument("--reid-samples", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=1, help="ReID 상위 K개 인물 트랙 반환")
    args = ap.parse_args()

    import torch
    from srpost.config import settings
    from srpost.models.yolo_detector import YOLODetector

    device = torch.device(args.device)
    yolo = YOLODetector(device=device)
    yolo._device = device

    # 1) 감지 + 트래킹
    print("[1/3] YOLO + BoT-SORT 추적 ...")
    tracks, fps, total, W, H = botsort_video(
        yolo._model, device, args.video,
        args.imgsz, args.det_conf, args.iou, args.vid_stride,
    )
    tracks = {t: d for t, d in tracks.items() if len(d) >= args.min_track_len}
    print(f"[filter] {len(tracks)} tracks (len ≥ {args.min_track_len}), "
          f"{W}x{H} @ {fps:.1f}fps")

    selected = list(tracks.keys())
    sims = {t: None for t in selected}

    # 2) (선택) ReID 로 특정 인물만 선별
    if args.query_image:
        from PIL import Image
        from decord import VideoReader, cpu
        from srpost.models.embedder import Embedder

        print("[2/3] SigLIP2-ReID 인물 매칭 ...")
        reid = Embedder(mode="full_gpu", adapter_path=settings.siglip2_reid_adapter_path)
        reid._device = device
        # 기준 사진에서 YOLO 로 인물 크롭 후 임베딩
        ref = Image.open(args.query_image).convert("RGB")
        crops = yolo.crop_persons(ref, conf=0.25)
        anchor_img = crops[0][0] if crops else ref
        anchor_emb = _l2(reid.embed_images([anchor_img])[0])

        vr = VideoReader(args.video, ctx=cpu(0), num_threads=2)
        sims = batched_reid_sims(vr, tracks, selected, anchor_emb, reid, args.reid_samples)
        selected = sorted(sims, key=lambda t: sims[t], reverse=True)[: args.top_k]
        print("[reid] top matches: " +
              ", ".join(f"track{t}={sims[t]:.3f}" for t in selected))

    # 3) 저장
    print("[3/3] 결과 저장 ...")
    payload = {
        "video": args.video,
        "frame_width": W, "frame_height": H, "fps": fps, "n_frames": total,
        "n_tracks_total": len(tracks),
        "tracks": [
            {
                "track_id": t,
                "reid_sim": sims.get(t),
                "n_points": len(tracks[t]),
                "trajectory": [
                    {"t": ts, "frame": fi, "bbox": list(bb), "conf": c}
                    for fi, ts, bb, c in sorted(tracks[t], key=lambda d: d[1])
                ],
            }
            for t in selected
        ],
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"✓ saved → {args.out}  ({len(selected)} track(s))")


if __name__ == "__main__":
    main()
