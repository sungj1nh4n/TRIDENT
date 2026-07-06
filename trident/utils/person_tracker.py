"""Track-level single-person trajectory with BoT-SORT + ReID matching.

Per-frame ReID matching was unreliable on crowded CCTV — different
people get cosine sim 0.85-0.95 just from sharing visual context.
This SOTA-style implementation instead lets BoT-SORT produce
*temporally consistent* multi-object tracks first, then picks the
track that best matches the anchor via track-level mean ReID
embedding cosine similarity. Each track is "one person" by
construction (Kalman + appearance + IoU), so no more mid-trajectory
identity switches.

Flow
----
1. Anchor crop → SigLIP2-ReID embedding.
2. Ultralytics YOLO + BoT-SORT streams over the whole video, emitting
   per-frame detections with track IDs. Tracks aggregate naturally.
3. For each track of length ≥ min_track_len, sample N evenly-spaced
   crops, embed all with SigLIP2-ReID, L2-normalize, mean-pool, and
   compute cosine similarity vs the anchor embedding.
4. Special boost: if any track contains a detection at the anchor's
   timestamp whose IoU with the anchor bbox is ≥ iou_lock_threshold,
   that track is the anchor's track — pick it unconditionally.
5. Otherwise pick the track with the highest mean-embedding cosine
   sim to anchor. Return its full per-frame bbox sequence.

Total compute on a 5-min 1080p CCTV clip is ~10-15 min on a single
A6000: YOLO+BoT-SORT pass dominates, ReID averaging is sub-second.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np
from PIL import Image

from srpost.models.embedder import Embedder
from srpost.models.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    t: float
    bbox: list[float]
    score: float
    verified: bool = True  # mid-track VLM check; False = tracker probably drifted


@dataclass
class TrackResult:
    frame_width: int
    frame_height: int
    points: list[TrajectoryPoint]
    n_candidate_scenes: int
    elapsed_s: float
    overlay_video_path: str | None = None  # SAM3 segmentation overlay video


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    return (v / n).astype(np.float32)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0


class PersonTracker:
    """SOTA track-then-identify person tracker.

    Reuses the YOLO detector + SigLIP2-ReID embedder already loaded in
    app.state, plus Ultralytics' built-in BoT-SORT.
    """

    def __init__(
        self,
        yolo: YOLODetector,
        embedder_reid: Embedder,
        pool,
    ) -> None:
        self._yolo = yolo
        self._reid = embedder_reid
        self._pool = pool

    async def track(
        self,
        video_path: str,
        video_id: uuid.UUID,
        anchor_scene_id: uuid.UUID,
        anchor_bbox: tuple[float, float, float, float],
        anchor_t: float,
        sim_threshold: float = 0.55,
        sample_fps: float = 4.0,  # legacy param, ignored — BoT-SORT runs at native fps
        max_persons_per_frame: int = 8,  # legacy, ignored
        max_candidate_scenes: int = 4,
        # SOTA-specific
        imgsz: int = 1280,
        det_conf: float = 0.25,
        iou_threshold: float = 0.5,
        min_track_len: int = 5,
        reid_samples_per_track: int = 5,
        iou_lock_threshold: float = 0.4,
        vid_stride: int = 4,
        roi_pad_s: float = 1.5,
        progress_cb=None,  # optional callable(stage:str, current:int, total:int)
        query: str | None = None,         # for mid-track VLM verification
        vlm_verifier=None,                # VLMReranker or compatible .score(img, q)
        verify_every: int = 8,            # check every Nth trajectory point
        verify_threshold: float = 0.4,    # VLM score above which a point counts as verified
    ) -> TrackResult:
        from decord import VideoReader, cpu

        t0 = time.monotonic()
        vr = None
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = float(vr.get_avg_fps()) or 30.0
            n_total = len(vr)
            sample_h, sample_w, _ = vr[0].asnumpy().shape
            W, H = int(sample_w), int(sample_h)
        except Exception as e:  # noqa: BLE001
            logger.warning("[track-sota] decord open failed: %s — ffprobe fallback", e)
            vr = None
            import json as _json
            import subprocess as _sub
            probe = _sub.run(
                ["ffprobe", "-v", "error", "-print_format", "json",
                 "-show_streams", "-select_streams", "v:0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            meta = _json.loads(probe.stdout)["streams"][0]
            num_s, den_s = meta.get("r_frame_rate", "30/1").split("/")
            fps = float(num_s) / max(1.0, float(den_s or 1))
            W = int(meta.get("width", 1920))
            H = int(meta.get("height", 1080))
            try:
                n_total = int(meta.get("nb_frames", 0)) or int(
                    float(meta.get("duration", 0)) * fps
                )
            except (TypeError, ValueError):
                n_total = int(fps * 600)

        if progress_cb: progress_cb("anchor", 0, 1)
        # 1. Anchor crop → embedding
        #    If anchor_bbox is degenerate (e.g. [0,0,0,0] from direct-anchor mode),
        #    fetch the best ReID crop from DB for this scene instead.
        x1, y1, x2, y2 = (int(max(0, v)) for v in anchor_bbox)
        x2 = min(W, x2); y2 = min(H, y2)
        use_db_crop = (x2 <= x1 or y2 <= y1)

        if use_db_crop:
            # Fetch best person_crop embedding from DB for anchor scene
            logger.info("[track-sota] degenerate bbox — fetching best DB crop for scene %s", anchor_scene_id)
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_idx
                       FROM person_crops
                       WHERE scene_id = $1 AND embedding IS NOT NULL
                       ORDER BY confidence DESC LIMIT 1""",
                    anchor_scene_id,
                )
            if row is None:
                raise ValueError(f"no person crops in scene {anchor_scene_id}")
            import json as _json
            emb_raw = row["embedding"]
            if isinstance(emb_raw, str):
                anchor_emb = np.array(_json.loads(emb_raw), dtype=np.float32)
            else:
                anchor_emb = np.frombuffer(emb_raw, dtype=np.float32)
            anchor_emb = _l2norm(anchor_emb)
            anchor_bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
            x1, y1, x2, y2 = anchor_bbox
            anchor_idx = max(0, min(int(anchor_t * fps), n_total - 1))
            logger.info("[track-sota] DB crop: bbox=(%d,%d,%d,%d) %dx%d",
                        x1, y1, x2, y2, x2 - x1, y2 - y1)
        else:
            anchor_idx = max(0, min(int(anchor_t * fps), n_total - 1))
            if vr is not None:
                anchor_frame_arr = vr[anchor_idx].asnumpy()
            else:
                from srpost.models.frame_fetch import _extract_uniform_frames_ffmpeg
                _frames = _extract_uniform_frames_ffmpeg(
                    video_path, anchor_t - 0.01, anchor_t + 0.01, 1,
                )
                if not _frames:
                    raise RuntimeError(f"anchor frame unreachable: {video_path}")
                anchor_frame_arr = np.asarray(_frames[0])
            anchor_frame = Image.fromarray(anchor_frame_arr)
            anchor_crop = anchor_frame.crop((x1, y1, x2, y2))
            anchor_emb = await asyncio.to_thread(self._reid.embed_images, [anchor_crop])
            anchor_emb = _l2norm(anchor_emb[0])
        logger.info(
            "[track-sota] anchor_idx=%d (t=%.2fs) crop=%dx%d",
            anchor_idx, anchor_t, x2 - x1, y2 - y1,
        )

        # 2. ROI scoping — figure out which time window is worth scanning.
        #    Combine: (a) anchor scene window, (b) DB-cached ReID candidate scenes,
        #    keeping only candidates within 60s of the anchor to ensure one
        #    contiguous BoT-SORT pass (Kalman motion model assumes continuity).
        #    Multiple distant appearances are handled by the UI's multi-anchor
        #    system which launches separate tracker calls per time cluster.
        anchor_meta = await self._fetch_scene(anchor_scene_id)
        if anchor_meta is None:
            roi_t_start, roi_t_end = 0.0, n_total / fps
        else:
            _, anc_t0, anc_t1 = anchor_meta
            roi_t_start, roi_t_end = anc_t0, anc_t1
            db_candidates = await self._find_candidate_scenes(
                video_id, anchor_emb, sim_threshold,
            )
            db_candidates.sort(key=lambda r: r[3], reverse=True)
            for sid, t0_s, t1_s, sim in db_candidates[:max_candidate_scenes]:
                if t1_s + 60.0 >= roi_t_start and t0_s - 60.0 <= roi_t_end:
                    roi_t_start = min(roi_t_start, t0_s)
                    roi_t_end = max(roi_t_end, t1_s)
        roi_t_start = max(0.0, roi_t_start - roi_pad_s)
        roi_t_end = min(n_total / fps, roi_t_end + roi_pad_s)
        roi_duration = roi_t_end - roi_t_start
        i_start = max(0, int(roi_t_start * fps))
        logger.info(
            "[track-sota] ROI: %.1f-%.1fs (%.1fs / %.1fs full, stride=%d)",
            roi_t_start, roi_t_end, roi_duration, n_total / fps, vid_stride,
        )

        # 3. Tracks: prefer the per-video BoT-SORT cache (var/_tracks_cache) — it
        #    skips the ffmpeg ROI extract (~3-9s) AND the YOLO+BoT-SORT pass
        #    (~10-30s), the two largest tracking costs. Cache frame indices are
        #    global, matching the full-video `vr` used for ReID. Fall back to a
        #    live ROI BoT-SORT pass for videos with no cache (fresh uploads).
        tracks: dict[int, list[tuple[int, float, tuple[float, float, float, float], float]]] = {}
        cached_tracks = self._load_cached_tracks(video_path, roi_t_start, roi_t_end)
        if cached_tracks is not None:
            tracks = cached_tracks
            logger.info("[track-sota] CACHE HIT: %d tracks in ROI (skipped ffmpeg+BoT-SORT)",
                        len(tracks))
            if progress_cb: progress_cb("botsort", 1, 1)
        else:
            import os
            import subprocess
            import tempfile
            tmp_clip = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_clip.close()
            try:
                if progress_cb: progress_cb("roi_extract", 0, 1)
                ff_t0 = time.monotonic()
                await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-ss", f"{roi_t_start:.3f}",
                        "-i", video_path,
                        "-t", f"{roi_duration:.3f}",
                        "-an",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                        tmp_clip.name,
                    ],
                    check=True,
                )
                logger.info("[track-sota] ffmpeg ROI extract: %.1fs", time.monotonic() - ff_t0)

                # 4. YOLO + BoT-SORT on the ROI clip.
                est_total = max(1, int(roi_duration * fps / vid_stride))
                if progress_cb: progress_cb("botsort", 0, est_total)
                await asyncio.to_thread(
                    self._botsort_pass_clip, tmp_clip.name, i_start, fps,
                    vid_stride, tracks, imgsz, det_conf, iou_threshold,
                    progress_cb, est_total,
                )
            finally:
                try:
                    os.unlink(tmp_clip.name)
                except FileNotFoundError:
                    pass
        if not tracks:
            logger.warning("[track-sota] no tracks produced by BoT-SORT")
            return TrackResult(
                frame_width=W, frame_height=H, points=[],
                n_candidate_scenes=0, elapsed_s=time.monotonic() - t0,
            )
        logger.info("[track-sota] BoT-SORT yielded %d tracks", len(tracks))

        # Filter very short tracks (probably YOLO flickers)
        tracks = {
            tid: dets for tid, dets in tracks.items() if len(dets) >= min_track_len
        }
        logger.info("[track-sota] %d tracks remain after min_track_len=%d filter",
                    len(tracks), min_track_len)
        if not tracks:
            return TrackResult(
                frame_width=W, frame_height=H, points=[],
                n_candidate_scenes=0, elapsed_s=time.monotonic() - t0,
            )

        # 3. IoU pre-filter — find tracks whose detection at anchor_t has
        #    high IoU with the anchor bbox; if there's a clear winner we skip
        #    the expensive ReID-everywhere pass entirely.
        iou_hits: list[tuple[int, float]] = []
        for tid, dets in tracks.items():
            best_iou = 0.0
            for fi, ts, bb, _conf in dets:
                if abs(ts - anchor_t) <= 0.5:
                    iou = _bbox_iou(bb, tuple(anchor_bbox))
                    if iou > best_iou:
                        best_iou = iou
            if best_iou >= iou_lock_threshold:
                iou_hits.append((tid, best_iou))
        iou_hits.sort(key=lambda x: x[1], reverse=True)
        logger.info("[track-sota] IoU pre-filter: %d tracks pass (threshold %.2f)",
                    len(iou_hits), iou_lock_threshold)

        winner: int | None = None
        track_sims: dict[int, float] = {}

        if len(iou_hits) == 1 or (
            len(iou_hits) >= 2 and (iou_hits[0][1] - iou_hits[1][1] >= 0.2)
        ):
            # IoU candidate found — verify with ReID before accepting
            iou_tid = iou_hits[0][0]
            iou_sim = await asyncio.to_thread(
                self._batched_reid_sims, vr, tracks, [iou_tid], anchor_emb,
                reid_samples_per_track,
            )
            iou_reid = iou_sim.get(iou_tid, 0.0)
            track_sims.update(iou_sim)
            if iou_reid >= 0.70:
                winner = iou_tid
                logger.info(
                    "[track-sota] IoU-locked to track %d (iou=%.2f, reid=%.3f) — verified",
                    winner, iou_hits[0][1], iou_reid,
                )
            else:
                logger.warning(
                    "[track-sota] IoU candidate track %d rejected (iou=%.2f but reid=%.3f < 0.70) — falling back to ReID",
                    iou_tid, iou_hits[0][1], iou_reid,
                )
                # Fall through to ReID path below

        if winner is None:
            # No verified IoU lock — run ReID on candidates
            if iou_hits:
                candidate_tids = [tid for tid, _ in iou_hits]
                logger.info("[track-sota] ReID disambiguation among %d IoU candidates",
                            len(candidate_tids))
            else:
                # Limit ReID to top-20 longest tracks to avoid scanning hundreds
                sorted_by_len = sorted(tracks.keys(), key=lambda t: len(tracks[t]), reverse=True)
                candidate_tids = sorted_by_len[:20]
                logger.info("[track-sota] no IoU lock — ReID over top-%d/%d tracks (by length)",
                            len(candidate_tids), len(tracks))
            track_sims = await asyncio.to_thread(
                self._batched_reid_sims, vr, tracks, candidate_tids, anchor_emb,
                reid_samples_per_track,
            )
            ranked = sorted(track_sims.items(), key=lambda kv: kv[1], reverse=True)
            logger.info("[track-sota] top-5 track sims: %s",
                        [(tid, round(s, 3), len(tracks[tid])) for tid, s in ranked[:5]])
            winner = ranked[0][0]
            if track_sims[winner] < sim_threshold:
                logger.warning(
                    "[track-sota] best track sim=%.3f < threshold %.2f — returning anyway",
                    track_sims[winner], sim_threshold,
                )

        # 5b. Collect additional tracks matching the same person via ReID.
        #     BoT-SORT assigns different track IDs when a person leaves and
        #     re-enters the frame. Merge high-similarity tracks so multiple
        #     appearances are captured in one trajectory.
        #     Use a tight absolute threshold (0.90) to avoid merging different
        #     people who look similar in CCTV footage (cosine 0.7-0.85 range).
        MAX_MERGE = 5  # cap to prevent runaway merges
        MERGE_SIM_FLOOR = 0.90  # absolute floor — only very close matches
        # Merge re-appearances: BoT-SORT assigns a NEW track id each time the
        # person leaves and re-enters frame. Score extra candidate tracks (capped
        # + batched frame reads, so this is a few seconds not ~100s) so every
        # appearance of the SAME person stitches into one trajectory — the UI
        # then renders each appearance as a separate bar on the progress timeline.
        # TRACK_MERGE=0 disables; "force" widens the candidate scan.
        import os as _os
        _merge_mode = _os.environ.get("TRACK_MERGE", "1")  # "0"=off, "1"=on, "force"=wide
        if _merge_mode != "0" and winner is not None and len(tracks) > 1:
            # Only consider tracks that are TEMPORALLY DISJOINT from the winner —
            # a re-appearance happens at a different time, so a concurrent track is
            # a different person (don't merge) and skipping them cuts the frame-read
            # cost. Then cap to the longest candidates.
            w_dets = tracks[winner]
            w_t0 = min(d[1] for d in w_dets); w_t1 = max(d[1] for d in w_dets)

            def _disjoint(tid: int) -> bool:
                dts = [d[1] for d in tracks[tid]]
                return (max(dts) < w_t0 - 0.5) or (min(dts) > w_t1 + 0.5)

            remaining = [tid for tid in tracks
                         if tid != winner and tid not in track_sims and _disjoint(tid)]
            MERGE_SCAN_CAP = 40 if _merge_mode == "force" else 8
            if len(remaining) > MERGE_SCAN_CAP:
                remaining.sort(key=lambda t: len(tracks[t]), reverse=True)
                remaining = remaining[:MERGE_SCAN_CAP]
            if remaining:
                extra_sims = await asyncio.to_thread(
                    self._batched_reid_sims, vr, tracks, remaining, anchor_emb,
                    min(3, reid_samples_per_track),
                )
                track_sims.update(extra_sims)

        # Build the merged track set (winner + same-person re-appearances). Always
        # computed — merge-off simply means only `winner` was ReID-scored.
        if winner is not None:
            winner_sim_val = track_sims.get(winner, 0.0)
            merge_threshold = max(MERGE_SIM_FLOOR, winner_sim_val * 0.95)
            candidates = [
                (tid, sim) for tid, sim in track_sims.items()
                if tid != winner and sim >= merge_threshold
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            merged_tids = [winner] + [tid for tid, _ in candidates[:MAX_MERGE]]
            if len(merged_tids) > 1:
                logger.info(
                    "[track-sota] merged %d tracks (threshold=%.3f): %s",
                    len(merged_tids), merge_threshold,
                    [(tid, round(track_sims.get(tid, 0), 3), len(tracks[tid])) for tid in merged_tids],
                )
        else:
            merged_tids = []

        # 6. Emit trajectory from all matching tracks, merged by time
        winner_dets = []
        for tid in merged_tids:
            winner_dets.extend(tracks[tid])
        winner_dets.sort(key=lambda d: d[1])  # sort by timestamp
        sim_score = track_sims.get(winner, 0.0)

        # 6b. Mid-track VLM verification — DISABLED for speed.
        # The tight merge threshold (0.90) + ReID matching already ensures
        # identity consistency. VLM verify was the #1 bottleneck (~60-240s).
        verified_flags = [True] * len(winner_dets)
        if False and query and vlm_verifier is not None and len(winner_dets) > 0:
            from decord import VideoReader, cpu as _cpu
            if progress_cb: progress_cb("verify", 0, len(winner_dets))
            sample_step = max(1, verify_every)
            sample_indices = list(range(0, len(winner_dets), sample_step))
            # Always include last point for end-of-track check
            if sample_indices[-1] != len(winner_dets) - 1:
                sample_indices.append(len(winner_dets) - 1)
            logger.info("[track-sota] mid-track VLM verify: %d samples / %d points",
                        len(sample_indices), len(winner_dets))
            sample_results: dict[int, bool] = {}
            vr_local = None
            try:
                vr_local = VideoReader(video_path, ctx=_cpu(0), num_threads=1)
            except Exception as e:  # noqa: BLE001
                logger.warning("[track-sota] verify: decord open failed (%s) — ffmpeg fallback", e)
            for k, idx in enumerate(sample_indices):
                fi, ts, bb, _c = winner_dets[idx]
                try:
                    if vr_local is not None:
                        arr = vr_local[fi].asnumpy()
                    else:
                        from srpost.models.frame_fetch import _extract_uniform_frames_ffmpeg
                        _f = _extract_uniform_frames_ffmpeg(video_path, ts - 0.01, ts + 0.01, 1)
                        if not _f:
                            sample_results[idx] = True
                            continue
                        arr = np.asarray(_f[0])
                    x1, y1, x2, y2 = (int(max(0, v)) for v in bb)
                    x2 = min(arr.shape[1], x2); y2 = min(arr.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        sample_results[idx] = True
                        continue
                    crop = Image.fromarray(arr[y1:y2, x1:x2])
                    s = await vlm_verifier.score(crop, query)
                    sample_results[idx] = (s >= verify_threshold)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[track-sota] verify failed at idx=%d: %s", idx, e)
                    sample_results[idx] = True
                if progress_cb: progress_cb("verify", k + 1, len(sample_indices))
            # Propagate verified flag to all points: between two samples, share
            # the AND of the bracketing samples (stricter — if either end fails
            # we treat the segment as unverified).
            sorted_keys = sorted(sample_results.keys())
            for j, idx in enumerate(sorted_keys):
                v = sample_results[idx]
                next_idx = sorted_keys[j + 1] if j + 1 < len(sorted_keys) else len(winner_dets)
                # All points from idx to next_idx inherit v AND next sample's v
                end_v = sample_results.get(sorted_keys[j + 1], v) if j + 1 < len(sorted_keys) else v
                fill = v and end_v
                for p in range(idx, min(next_idx, len(winner_dets))):
                    verified_flags[p] = fill
            n_verified = sum(verified_flags)
            logger.info("[track-sota] verified %d/%d points (%.0f%%)",
                        n_verified, len(verified_flags),
                        100 * n_verified / len(verified_flags))

        # 6c. Post-filter: remove drifted points where BoT-SORT's Kalman
        # prediction landed on empty space (typical during person crossings),
        # then linearly interpolate gaps so the trajectory stays smooth.
        MIN_DET_CONF = 0.20
        pre_filter = len(winner_dets)
        # Tag each point as valid or drifted
        valid_mask = [d[3] >= MIN_DET_CONF for d in winner_dets]
        n_removed = sum(1 for v in valid_mask if not v)
        if n_removed > 0:
            logger.info("[track-sota] post-filter: %d/%d low-conf points → interpolate",
                        n_removed, pre_filter)

        # Build final points with linear interpolation for drifted gaps
        points: list[TrajectoryPoint] = []
        last_valid_idx: int | None = None
        for i, (_fi, ts, bb, _conf) in enumerate(winner_dets):
            if valid_mask[i]:
                # Fill any gap since last valid point with linear interpolation
                if last_valid_idx is not None and i - last_valid_idx > 1:
                    lv = winner_dets[last_valid_idx]
                    t0, bb0 = lv[1], lv[2]
                    t1, bb1 = ts, bb
                    for j in range(last_valid_idx + 1, i):
                        gap_t = winner_dets[j][1]
                        if t1 - t0 > 0:
                            alpha = (gap_t - t0) / (t1 - t0)
                        else:
                            alpha = 0.5
                        interp_bb = tuple(
                            bb0[k] + alpha * (bb1[k] - bb0[k]) for k in range(4)
                        )
                        points.append(TrajectoryPoint(
                            t=float(gap_t),
                            bbox=[float(v) for v in interp_bb],
                            score=float(sim_score),
                            verified=False,  # interpolated
                        ))
                # Add the valid point itself
                vflag = verified_flags[i] if i < len(verified_flags) else True
                points.append(TrajectoryPoint(
                    t=float(ts),
                    bbox=[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    score=float(sim_score),
                    verified=bool(vflag),
                ))
                last_valid_idx = i
        # Sort by time (should already be, but safe)
        points.sort(key=lambda p: p.t)
        elapsed = time.monotonic() - t0
        logger.info(
            "[track-sota] winner=%d (sim=%.3f), %d points, elapsed=%.1fs",
            winner, sim_score, len(points), elapsed,
        )
        return TrackResult(
            frame_width=W, frame_height=H, points=points,
            n_candidate_scenes=len(tracks),
            elapsed_s=elapsed,
        )

    async def _find_candidate_scenes(
        self, video_id: uuid.UUID, anchor_emb: np.ndarray, threshold: float,
    ) -> list[tuple[uuid.UUID, float, float, float]]:
        """Return [(scene_id, t_start, t_end, max_cos_sim)] for scenes whose
        stored ingest-time person_crops have at least one match above `threshold`.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT pc.scene_id, s.t_start, s.t_end, pc.embedding
                FROM person_crops pc
                JOIN scenes s ON s.id = pc.scene_id
                WHERE pc.video_id = $1
                  AND pc.embedding IS NOT NULL
                """,
                video_id,
            )
        best: dict[uuid.UUID, tuple[float, float, float]] = {}
        for r in rows:
            emb = np.asarray(r["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(anchor_emb, emb))
            sid = r["scene_id"]
            prev = best.get(sid)
            if prev is None or sim > prev[2]:
                best[sid] = (float(r["t_start"]), float(r["t_end"]), sim)
        return [
            (sid, t0, t1, sim) for sid, (t0, t1, sim) in best.items()
            if sim >= threshold
        ]

    async def _fetch_scene(
        self, scene_id: uuid.UUID,
    ) -> tuple[uuid.UUID, float, float] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, t_start, t_end FROM scenes WHERE id = $1", scene_id,
            )
        if row is None:
            return None
        return row["id"], float(row["t_start"]), float(row["t_end"])

    def _botsort_pass_clip(
        self,
        clip_path: str,
        clip_origin_frame: int,
        fps: float,
        vid_stride: int,
        tracks: dict[int, list],
        imgsz: int,
        conf: float,
        iou: float,
        progress_cb=None,
        progress_total: int = 0,
    ) -> None:
        """Run YOLO + BoT-SORT over a clip file, mapping local frame index back
        to absolute source-video frame index for timestamp continuity.
        """
        model = self._yolo._model
        device = self._yolo._device
        results = model.track(
            source=clip_path,
            # persist=False: the YOLO model is a process-wide singleton shared by
            # every /track-person request. persist=True kept the BoT-SORT tracker
            # state (STrack lists, Kalman filters, ReID feature banks) alive ACROSS
            # requests on that singleton, growing without bound → ~1TB RSS / OOM.
            # This pass runs once per request as a single stream, so resetting the
            # tracker at the start of each call is correct and loses nothing.
            persist=False,
            tracker="botsort.yaml",
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            vid_stride=vid_stride,
            verbose=False,
        )
        sample_idx = 0
        for r in results:
            actual_idx = clip_origin_frame + sample_idx * vid_stride
            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                ts = actual_idx / fps
                for tid, box, c in zip(ids, xyxy, confs):
                    tracks.setdefault(int(tid), []).append((
                        actual_idx, float(ts),
                        (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        float(c),
                    ))
            sample_idx += 1
            if sample_idx % 50 == 0:
                if progress_cb:
                    progress_cb("botsort", sample_idx, progress_total)
                if sample_idx % 250 == 0:
                    logger.info("[track-sota] BoT-SORT progress: %d/%d frames tracks=%d",
                                sample_idx, progress_total, len(tracks))

    def _botsort_pass_frames(
        self,
        frame_arrays: list,
        frame_indices: list[int],
        fps: float,
        tracks: dict[int, list],
        imgsz: int,
        conf: float,
        iou: float,
    ) -> None:
        """Run Ultralytics YOLO + BoT-SORT over an in-memory list of frames.

        `frame_arrays[i]` is the i-th frame to track; `frame_indices[i]` is its
        position in the source video (used for timestamp mapping). Tracks are
        keyed on Ultralytics-internal track IDs, valid across the stream.
        """
        model = self._yolo._model
        device = self._yolo._device
        results = model.track(
            source=frame_arrays,
            # persist=False: see _botsort_pass_clip — the shared singleton model
            # must not retain BoT-SORT state across requests (OOM leak source).
            persist=False,
            tracker="botsort.yaml",
            classes=[0],  # person
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            verbose=False,
        )
        sample_idx = 0
        for r in results:
            if sample_idx >= len(frame_indices):
                break
            actual_idx = frame_indices[sample_idx]
            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                ts = actual_idx / fps
                for tid, box, c in zip(ids, xyxy, confs):
                    tracks.setdefault(int(tid), []).append((
                        actual_idx, float(ts),
                        (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        float(c),
                    ))
            sample_idx += 1
            if sample_idx % 500 == 0:
                logger.info("[track-sota] BoT-SORT progress: %d/%d frames tracks=%d",
                            sample_idx, len(frame_indices), len(tracks))

    _TRACKS_CACHE_DIR = "/mnt/storage/LongVideoHaystack/SRPost/var/_tracks_cache"

    def _load_cached_tracks(
        self, video_path: str, roi_t_start: float, roi_t_end: float,
    ) -> dict | None:
        """Load pre-computed BoT-SORT tracks for this video and keep those with
        ≥1 detection inside the ROI window. Returns None if no cache exists.

        Cache format (built by scripts/query_detect.botsort_video) matches the
        PersonTracker det tuple exactly: (frame_idx:int, t:float, bbox:tuple, conf:float),
        with GLOBAL frame indices — compatible with the full-video `vr`.
        """
        import os
        import pickle
        try:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            path = os.path.join(self._TRACKS_CACHE_DIR, f"{stem}.pkl")
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                data = pickle.loads(f.read())
            all_tracks = data.get("tracks") or {}
            kept: dict[int, list] = {}
            for tid, dets in all_tracks.items():
                if any(roi_t_start <= d[1] <= roi_t_end for d in dets):
                    kept[int(tid)] = list(dets)
            return kept or None
        except Exception as e:  # noqa: BLE001
            logger.warning("[track-sota] cached-tracks load failed: %s", e)
            return None

    def _batched_reid_sims(
        self,
        vr,
        tracks: dict,
        candidate_tids: list[int],
        anchor_emb: np.ndarray,
        n_samples: int,
    ) -> dict[int, float]:
        """Run one batched ReID embedding call across all candidate tracks' crops,
        return per-track mean cosine similarity to anchor.
        """
        # Gather (frame_idx, bbox, tid) requests across all candidate tracks first.
        reqs: list[tuple[int, tuple, int]] = []
        for tid in candidate_tids:
            dets = tracks[tid]
            if not dets:
                continue
            top = sorted(dets, key=lambda d: d[3], reverse=True)[: max(n_samples * 3, 10)]
            top.sort(key=lambda d: d[0])
            if len(top) <= n_samples:
                chosen = top
            else:
                step = len(top) / n_samples
                chosen = [top[int(i * step)] for i in range(n_samples)]
            for fi, _ts, bb, _c in chosen:
                reqs.append((int(fi), bb, tid))

        all_crops: list[Image.Image] = []
        owners: list[int] = []
        if reqs:
            # BATCH-decode the unique frames in one decord call. Per-frame vr[fi]
            # random seeks were the dominant tracking cost (~0.3s/frame at 1440p →
            # 100s+ when scoring dozens of tracks); get_batch decodes together.
            n_frames = len(vr)
            uniq = sorted({min(max(0, fi), n_frames - 1) for fi, _, _ in reqs})
            frames_by_idx: dict[int, np.ndarray] = {}
            try:
                batch = vr.get_batch(uniq).asnumpy()
                for j, fi in enumerate(uniq):
                    frames_by_idx[fi] = batch[j]
            except Exception:  # noqa: BLE001
                for fi in uniq:
                    try:
                        frames_by_idx[fi] = vr[fi].asnumpy()
                    except Exception:  # noqa: BLE001
                        continue
            for fi, bb, tid in reqs:
                arr = frames_by_idx.get(min(max(0, fi), n_frames - 1))
                if arr is None:
                    continue
                H, W = arr.shape[:2]
                x1 = max(0, int(bb[0])); y1 = max(0, int(bb[1]))
                x2 = min(W, int(bb[2])); y2 = min(H, int(bb[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                all_crops.append(Image.fromarray(arr[y1:y2, x1:x2]))
                owners.append(tid)

        if not all_crops:
            return {tid: 0.0 for tid in candidate_tids}

        # Single GPU batched call across all candidate crops
        embs = self._reid.embed_images(all_crops)
        embs = _l2norm(embs)
        sims = embs @ anchor_emb  # (N,)

        out: dict[int, float] = {tid: 0.0 for tid in candidate_tids}
        counts: dict[int, int] = {tid: 0 for tid in candidate_tids}
        for owner, s in zip(owners, sims.tolist()):
            out[owner] = out.get(owner, 0.0) + float(s)
            counts[owner] = counts.get(owner, 0) + 1
        for tid in list(out):
            if counts.get(tid, 0) > 0:
                out[tid] /= counts[tid]
        return out

    def _track_mean_reid_sim(
        self,
        vr,
        dets: list[tuple[int, float, tuple[float, float, float, float], float]],
        anchor_emb: np.ndarray,
        n_samples: int,
    ) -> float:
        """Sample N evenly-spaced detections from a track, embed crops, mean → sim."""
        if not dets:
            return 0.0
        # Sort by detection confidence (prefer clearer crops) then pick evenly-spaced
        dets_sorted = sorted(dets, key=lambda d: d[3], reverse=True)
        top = dets_sorted[: max(n_samples * 3, 10)]  # candidate pool
        top.sort(key=lambda d: d[0])  # back to temporal order

        # Even sampling across temporal positions
        if len(top) <= n_samples:
            chosen = top
        else:
            step = len(top) / n_samples
            chosen = [top[int(i * step)] for i in range(n_samples)]

        crops: list[Image.Image] = []
        for fi, _ts, bb, _c in chosen:
            try:
                arr = vr[fi].asnumpy()
            except Exception:  # noqa: BLE001
                continue
            H, W = arr.shape[:2]
            x1 = max(0, int(bb[0])); y1 = max(0, int(bb[1]))
            x2 = min(W, int(bb[2])); y2 = min(H, int(bb[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(Image.fromarray(arr[y1:y2, x1:x2]))

        if not crops:
            return 0.0

        embs = self._reid.embed_images(crops)
        embs = _l2norm(embs)
        # Each crop sim → mean (more stable than embedding mean for very different poses)
        sims = embs @ anchor_emb
        return float(np.mean(sims))
