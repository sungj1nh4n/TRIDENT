# TRIDENT
# PersonTracking
# 영상 내 능동형 인물 감지 및 재식별(ReID) 기반 다중객체 궤적 트래킹

CCTV·긴 영상에서 **사람이 등장하는 구간을 자동 감지**하고, 특정 인물을 **영상 전체에
걸쳐 시간적으로 일관되게 추적**하여 이동 궤적을 산출하는 인물 트래킹 시스템입니다.
프레임마다 검출한 사람을 외형 유사도로 이어 붙이는(per-frame ReID) 방식은 붐비는 CCTV
에서 서로 다른 사람이 배경·조명만 공유해도 코사인 0.85~0.95를 기록해 신원이 자주
뒤바뀝니다. 본 방법은 **감지-후-식별(Track-then-Identify)** 로 이를 해결합니다:
① **BoT-SORT** 로 먼저 시간적으로 일관된 트랙(= 한 사람)을 만들고 → ② 트랙 단위 **평균
ReID 임베딩**으로 대상 인물만 선별 → ③ **재등장 병합·드리프트 보정**으로 하나의 매끄러운
궤적을 완성합니다.

핵심 구현:
- 인물 감지 → [`srpost/models/yolo_detector.py`](PersonTracking_Detector_Tracker/srpost/models/yolo_detector.py) → `YOLODetector`
- 인물 추적 → [`srpost/models/person_tracker.py`](PersonTracking_Detector_Tracker/srpost/models/person_tracker.py) → `PersonTracker`
- ReID 임베더 → [`srpost/models/embedder.py`](PersonTracking_Detector_Tracker/srpost/models/embedder.py) → `Embedder`
- 독립 실행 러너 → [`detect_and_track.py`](PersonTracking_Detector_Tracker/detect_and_track.py)

---

## 왜 감지-후-식별(Track-then-Identify)인가?

| 문제 | 해결 모듈 |
|------|-----------|
| 프레임별 ReID는 붐비는 CCTV에서 서로 다른 사람을 0.85~0.95로 오인 → 신원 뒤바뀜 | **① BoT-SORT 다중객체 추적** (칼만 + 외형 + IoU로 트랙을 먼저 확정) |
| 어떤 트랙이 "그 사람"인가 | **② 트랙 단위 평균 ReID 임베딩** (기준 인물과 코사인 유사도) |
| 화면 이탈·재등장 시 트랙 ID가 갈라짐 | **③ 재등장 병합** (시간적 비중첩 + 엄격 임계값 0.90) |
| 교차 순간 추적기 예측이 빈 공간에 떨어짐 | **④ 드리프트 제거 + 선형 보간** |

---

## 방법 — 4단계 파이프라인

### ① 인물 감지 + BoT-SORT 추적
Ultralytics YOLO(class 0 = person)로 사람을 검출하고, 내장 BoT-SORT가 칼만 모션 모델 +
외형 특징 + IoU 매칭으로 프레임 간 동일 인물을 하나의 `track_id`로 묶습니다. 각 트랙은
구성상 "한 사람"이라 궤적 중간의 신원 뒤바뀜이 원천 차단됩니다.
```python
model.track(source=clip, tracker="botsort.yaml", classes=[0],
            conf=0.25, iou=0.5, imgsz=1280, vid_stride=2, stream=True)
# → track_id 별 (frame_idx, t, bbox, conf) 시퀀스
```

### ② IoU-Lock → 트랙 단위 ReID 선별
기준 인물의 관찰 시점에서 IoU ≥ `iou_lock_threshold`(기본 0.4)인 뚜렷한 트랙이 있으면
그 트랙을 대상으로 확정(ReID 전수 계산 생략), 확정 후에도 ReID로 한 번 더 검증합니다.
IoU-Lock이 없으면 길이 상위 트랙들에 대해서만 ReID를 계산합니다.
```python
anchor_emb = l2norm(reid.embed_images([anchor_crop])[0])   # 기준 인물
track_sim  = mean_j cos(anchor_emb, reid(crop_j))          # 트랙별 평균 유사도
winner     = argmax_track track_sim
```

### ③ 재등장 트랙 병합
인물이 화면을 벗어났다 재등장하면 BoT-SORT가 새 track_id를 부여합니다. **시간적으로
겹치지 않는** 트랙만 후보로 삼아 엄격한 임계값(코사인 0.90)으로 같은 사람의 여러 등장을
하나의 궤적으로 병합합니다(서로 닮은 다른 사람의 오병합 방지).

### ④ 드리프트 보정 + 궤적 평활화
추적기 예측이 빈 공간에 떨어진 저신뢰(conf < 0.20) 지점을 제거하고, 그 구간을 선형
보간으로 메워 매끄러운 프레임별 궤적 `{t, bbox, score}` 을 산출합니다.

---

## 주요 파라미터

| 단계 | 파라미터 | 기본값 | 의미 |
|------|----------|--------|------|
| ① 감지 | `det_conf`, `iou`, `imgsz` | 0.25, 0.5, 1280 | YOLO 신뢰도/IoU/입력 해상도 |
| ① 추적 | `vid_stride`, `min_track_len` | 2~4, 5 | N프레임당 1장 / 최소 트랙 길이(깜빡임 제거) |
| ② ReID | `reid_samples_per_track`, `sim_threshold` | 5, 0.55 | 트랙당 크롭 샘플 수 / 최소 유사도 |
| ② IoU-Lock | `iou_lock_threshold` | 0.4 | 앵커 시점 IoU 확정 임계 |
| ③ 병합 | `MERGE_SIM_FLOOR`, `MAX_MERGE` | 0.90, 5 | 재등장 병합 임계/최대 병합 수 |
| ④ 보정 | `MIN_DET_CONF` | 0.20 | 드리프트 제거 임계 |

---

## 저장소 구조

```
PersonTracking_Registration_Package/
├── README.md                              # 본 문서
├── 소프트웨어등록_인물감지_트래킹.md        # 소프트웨어 등록서 (본문)
└── PersonTracking_Detector_Tracker/
    ├── detect_and_track.py                # 독립 실행 러너 (감지+추적+ReID)
    ├── requirements.txt                   # 실행 패키지
    └── srpost/
        ├── config.py                      # 설정(Settings) — 모델 경로/디바이스
        └── models/
            ├── yolo_detector.py           # ① 인물 감지 (YOLODetector)
            ├── person_tracker.py          # ②③④ 트래킹 (PersonTracker, 운영 코드)
            └── embedder.py                # SigLIP2 ReID 임베더 (Embedder)
```

> `person_tracker.py` 는 운영 API(FastAPI)에서 쓰이는 실제 코드로, PostgreSQL 트랙 캐시·
> 후보 장면 검색 등 DB 연동 경로를 포함합니다. DB 없이 곧바로 검증하려면
> `detect_and_track.py` 독립 러너를 사용하세요(감지+추적+ReID 핵심 경로만 축약).

---

## 설치

```bash
# Python 3.10–3.12. GPU(CUDA) 환경 권장.
cd PersonTracking_Detector_Tracker
pip install -r requirements.txt
```

준비물:
- **YOLO 가중치**: 사람 감지용 `yolo26x.pt`(또는 `yolov8n.pt`). `srpost/config.py` 의
  `yolo_weights_path` 로 지정하거나 작업 폴더에 두면 자동 탐색.
- **(선택) SigLIP2 ReID LoRA 어댑터**: `siglip2_reid_adapter_path` 로 지정. 없으면 베이스
  SigLIP2 모델로 동작.

---

## 사용법

### 1) 감지 + 트래킹만 — 모든 인물 트랙 추출

```bash
cd PersonTracking_Detector_Tracker
python detect_and_track.py --video clip.mp4 --out out.json
```

출력 `out.json` 구조:
```json
{
  "frame_width": 1920, "frame_height": 1080, "fps": 30.0,
  "n_tracks_total": 7,
  "tracks": [
    {
      "track_id": 3, "reid_sim": null, "n_points": 214,
      "trajectory": [ {"t": 12.3, "frame": 369, "bbox": [x1,y1,x2,y2], "conf": 0.91}, ... ]
    }
  ]
}
```

### 2) 특정 인물만 추적 — 기준 사진으로 ReID 매칭

```bash
python detect_and_track.py --video clip.mp4 \
    --query-image person.jpg --top-k 1 --out person1.json
```
기준 사진에서 YOLO로 인물을 크롭해 SigLIP2-ReID로 임베딩한 뒤, 각 트랙과의 평균 코사인
유사도 상위 K개 인물의 궤적만 반환합니다.

주요 옵션:

| 옵션 | 기본값 | 의미 |
|------|--------|------|
| `--video` | (필수) | 입력 영상 경로 |
| `--query-image` | 없음 | 특정 인물 사진(주면 그 사람만 추적) |
| `--top-k` | 1 | ReID 상위 K개 인물 트랙 반환 |
| `--device` | cuda:0 | 추론 디바이스 |
| `--imgsz` | 1280 | YOLO 입력 해상도 |
| `--det-conf` | 0.25 | 감지 신뢰도 임계 |
| `--vid-stride` | 2 | N프레임마다 1장 처리(속도↔정밀도) |
| `--min-track-len` | 5 | 최소 트랙 길이(깜빡임 제거) |
| `--reid-samples` | 5 | 트랙당 ReID 크롭 샘플 수 |

### 3) 운영 환경 — FastAPI 엔드포인트

운영 스택에서는 앵커 인물(장면 ID·경계상자·시각)을 REST로 전달해 전체 영상 궤적을
조회합니다. `person_tracker.PersonTracker` 가 트랙 캐시·후보 장면 검색까지 활용합니다.
```
POST /track-person   { video_id, scene_id, anchor_bbox, anchor_t, query }
                     → { trajectory: [{t, bbox, score, verified}], elapsed_s, ... }
GET  /track-progress → { stage, current, total, elapsed_s }   # 진행률 폴링
```

---

## 성능 참고

- 5분·1080p CCTV 클립 기준 단일 A6000에서 약 10~15분(YOLO+BoT-SORT 패스가 지배적,
  ReID 평균화는 1초 미만).
- per-video 트랙 캐시(`var/_tracks_cache`)가 있으면 ffmpeg 추출 + YOLO+BoT-SORT 패스를
  건너뛰어 수 초 내 응답.
- BoT-SORT 는 요청마다 `persist=False` 로 초기화 — 전역 싱글턴 모델에 추적기 상태가
  누적되어 메모리가 무한 증가(OOM)하는 것을 방지.
