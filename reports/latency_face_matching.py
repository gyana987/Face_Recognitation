"""Per-camera latency test for the FULL missing-person matching pipeline
(Wildtrack cam1-cam7): person detection -> per-person face detection +
embedding -> embedding comparison against one reference photo.

For each camera, reads 10 RANDOM frames and times, separately, in seconds:
    - person_detection_s   : YOLO person detection on the whole frame
    - face_detect_embed_s  : face detection + embedding extraction, summed
                              across every detected person crop in that frame
    - embedding_matching_s : cosine-similarity comparison of every embedding
                              found against the one reference embedding
    - total_s              : sum of the three stages above

Reference photo: input/reference/wildtrack_missing_person.jpg (the photo
already used for Wildtrack searches in config/search_cameras.ini).

Writes reports/latency_face_matching.xlsx: one sheet per camera, an
"All Frames" sheet, and a "Summary by Camera" sheet -- all times in seconds.

Does not modify any existing project file -- only imports read-only building
blocks from search_multi_camera.py.

Usage:
    python reports/latency_face_matching.py
"""
import glob
import importlib.util
import os
import random
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)


def _ensure_gpu_libs_on_path():
    if os.environ.get("_PFP_GPU_LIBS_SET") == "1":
        return
    lib_dirs = []
    if importlib.util.find_spec("nvidia") is not None:
        for pkg in ("nvidia.cu13", "nvidia.cudnn"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                lib_dirs.append(os.path.join(list(spec.submodule_search_locations)[0], "lib"))
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
        os.environ["_PFP_GPU_LIBS_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_gpu_libs_on_path()

import cv2
from openpyxl import Workbook
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.styles import Font, PatternFill

from search_multi_camera import resolve_device, load_person_model, FaceMatcher, crop_upscaled, cosine_sim

WILDTRACK_DIR = os.path.join(BASE_DIR, "Wildtrack")
REFERENCE_IMAGE = os.path.join(BASE_DIR, "input", "reference", "wildtrack_missing_person.jpg")
OUT_XLSX = os.path.join(BASE_DIR, "reports", "latency_face_matching.xlsx")

N_FRAMES = 10
RANDOM_SEED = 42

CAMERA_FILES = {}
for path in sorted(glob.glob(os.path.join(WILDTRACK_DIR, "*.mp4"))):
    fname = os.path.basename(path)
    if fname.endswith("_clip.mp4"):
        continue
    m = re.search(r"cam(\d+)", fname, re.IGNORECASE)
    if m:
        CAMERA_FILES[f"cam{m.group(1)}"] = path

CAMERAS = [f"cam{i}" for i in range(1, 8)]
missing = [c for c in CAMERAS if c not in CAMERA_FILES]
if missing:
    raise SystemExit(f"Could not find video files for: {missing}. Found: {CAMERA_FILES}")

print("Camera -> video file mapping:")
for c in CAMERAS:
    print(f"  {c}: {os.path.basename(CAMERA_FILES[c])}")

device = resolve_device("auto")
print(f"\nDevice: {device}")
person_model = load_person_model(device)
matcher = FaceMatcher(device, (640, 640), 28)

# ---------------------------------------------------------------------------
print(f"\nReference image: {os.path.basename(REFERENCE_IMAGE)}")
ref_img_raw = cv2.imread(REFERENCE_IMAGE)
if ref_img_raw is None:
    raise SystemExit(f"Could not read reference image: {REFERENCE_IMAGE}")
ref_img, _ = crop_upscaled(ref_img_raw, 220, 4.0)
ref_embedding, _, ref_score = matcher.best_face_embedding(ref_img)
if ref_embedding is None:
    raise SystemExit("No face found in reference image -- pick a different reference photo.")
print(f"Reference face found (det_score={ref_score:.3f})")

# ---------------------------------------------------------------------------
random.seed(RANDOM_SEED)
all_results = {}

for cam in CAMERAS:
    video_path = CAMERA_FILES[cam]
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= N_FRAMES:
        positions = list(range(frame_count))
    else:
        positions = sorted(random.sample(range(frame_count), N_FRAMES))
    print(f"\n== {cam} ({os.path.basename(video_path)}, {frame_count} frames) -- sampled: {positions} ==")

    frame_results = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {pos}: could not read, skipping")
            continue

        # Stage 1: person detection
        t0 = time.perf_counter()
        dets = person_model.run_inference([frame], {"tracking": False, "confidence": 0.25, "imgsz": 960})[0]
        person_detection_s = time.perf_counter() - t0

        # Stage 2: face detection + embedding extraction (every person crop)
        embeddings = []
        t0 = time.perf_counter()
        for det in dets:
            x1, y1, x2, y2 = map(int, det["xyxy"])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            up, _ = crop_upscaled(crop, 220, 4.0)
            emb, _, _ = matcher.best_face_embedding(up)
            if emb is not None:
                embeddings.append(emb)
        face_detect_embed_s = time.perf_counter() - t0

        # Stage 3: embedding matching against the reference (all faces found).
        # A single cosine_sim call is sub-microsecond, so repeat the whole
        # comparison set many times and average -- gives a real, non-zero
        # measurement instead of a single call the wall clock can't resolve.
        MATCH_REPEATS = 2000
        t0 = time.perf_counter()
        for _ in range(MATCH_REPEATS):
            similarities = [cosine_sim(e, ref_embedding) for e in embeddings]
        embedding_matching_s = (time.perf_counter() - t0) / MATCH_REPEATS

        total_s = person_detection_s + face_detect_embed_s + embedding_matching_s
        row = {
            "frame": pos,
            "persons_detected": len(dets),
            "faces_embedded": len(embeddings),
            "person_detection_s": round(person_detection_s, 6),
            "face_detect_embed_s": round(face_detect_embed_s, 6),
            "embedding_matching_s": round(embedding_matching_s, 6),
            "total_s": round(total_s, 6),
            "person_detection_ms": round(person_detection_s * 1000, 3),
            "face_detect_embed_ms": round(face_detect_embed_s * 1000, 3),
            "embedding_matching_ms": round(embedding_matching_s * 1000, 3),
            "total_ms": round(total_s * 1000, 3),
            "best_similarity": round(max(similarities), 4) if similarities else None,
        }
        frame_results.append(row)
        print(f"  frame {pos}: persons={row['persons_detected']} faces={row['faces_embedded']} "
              f"detect={row['person_detection_s']}s embed={row['face_detect_embed_s']}s "
              f"match={row['embedding_matching_ms']}ms total={row['total_s']}s "
              f"best_sim={row['best_similarity']}")
    cap.release()
    all_results[cam] = frame_results

# ---------------------------------------------------------------------------
# Build Excel workbook
wb = Workbook()
header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
per_cam_headers = ["Frame", "Persons Detected", "Faces Embedded",
                    "Person Detection (s)", "Face Detect+Embed (s)", "Embedding Matching (ms)", "Total (s)",
                    "Best Similarity"]
all_headers = ["Camera"] + per_cam_headers


def style_header(ws, n_cols):
    for col in range(1, n_cols + 1):
        c = ws.cell(row=1, column=col)
        c.font, c.fill = header_font, header_fill


def row_values(row):
    return [row["person_detection_s"], row["face_detect_embed_s"], row["embedding_matching_ms"], row["total_s"],
            row["best_similarity"]]


for cam in CAMERAS:
    ws = wb.create_sheet(title=cam)
    ws.append(per_cam_headers)
    style_header(ws, len(per_cam_headers))
    for row in all_results[cam]:
        ws.append([row["frame"], row["persons_detected"], row["faces_embedded"]] + row_values(row))
    for col, width in zip("ABCDEFGH", (10, 16, 14, 18, 18, 20, 12, 14)):
        ws.column_dimensions[col].width = width

default_sheet = wb["Sheet"]
wb.remove(default_sheet)

ws_all = wb.create_sheet(title="All Frames", index=0)
ws_all.append(all_headers)
style_header(ws_all, len(all_headers))
for cam in CAMERAS:
    for row in all_results[cam]:
        ws_all.append([cam, row["frame"], row["persons_detected"], row["faces_embedded"]] + row_values(row))
for col, width in zip("ABCDEFGHI", (10, 10, 16, 14, 18, 18, 20, 12, 14)):
    ws_all.column_dimensions[col].width = width

last_row = ws_all.max_row
chart = ScatterChart()
chart.title = "Total Latency vs Persons Detected (all cameras)"
chart.x_axis.title = "Persons Detected"
chart.y_axis.title = "Total Latency (s)"
chart.style = 13
xvalues = Reference(ws_all, min_col=3, min_row=2, max_row=last_row)
yvalues = Reference(ws_all, min_col=8, min_row=2, max_row=last_row)  # Total (s)
series = Series(yvalues, xvalues, title="All cameras")
series.marker.symbol = "circle"
series.graphicalProperties.line.noFill = True
chart.series.append(series)
chart.width, chart.height = 18, 12
ws_all.add_chart(chart, "K2")

# Second chart: embedding-matching time vs number of faces (the actual
# question this report answers -- matching cost is real, just tiny)
chart_match = ScatterChart()
chart_match.title = "Embedding Matching Time vs Faces Embedded"
chart_match.x_axis.title = "Faces Embedded"
chart_match.y_axis.title = "Embedding Matching (ms)"
chart_match.style = 13
xvalues_m = Reference(ws_all, min_col=4, min_row=2, max_row=last_row)  # Faces Embedded
yvalues_m = Reference(ws_all, min_col=7, min_row=2, max_row=last_row)  # Embedding Matching (ms)
series_m = Series(yvalues_m, xvalues_m, title="All cameras")
series_m.marker.symbol = "diamond"
series_m.graphicalProperties.line.noFill = True
chart_match.series.append(series_m)
chart_match.width, chart_match.height = 18, 12
ws_all.add_chart(chart_match, "K26")

# Summary by camera: seconds for detection/embed/total, ms for matching
ws_summary = wb.create_sheet(title="Summary by Camera", index=1)
summary_headers = ["Camera", "Frames Tested",
                    "Avg Person Detection (s)", "Avg Face Detect+Embed (s)", "Avg Embedding Matching (ms)",
                    "Avg Total (s)", "Total Time all Frames (s)"]
ws_summary.append(summary_headers)
style_header(ws_summary, len(summary_headers))

grand = {"person_detection_s": 0.0, "face_detect_embed_s": 0.0, "embedding_matching_ms": 0.0, "total_s": 0.0}
grand_n = 0
for cam in CAMERAS:
    rows = all_results[cam]
    n = len(rows)
    avg_pd = sum(r["person_detection_s"] for r in rows) / n
    avg_fde = sum(r["face_detect_embed_s"] for r in rows) / n
    avg_em_ms = sum(r["embedding_matching_ms"] for r in rows) / n
    avg_tot = sum(r["total_s"] for r in rows) / n
    total_all = sum(r["total_s"] for r in rows)
    ws_summary.append([cam, n, round(avg_pd, 6), round(avg_fde, 6), round(avg_em_ms, 4),
                        round(avg_tot, 6), round(total_all, 4)])
    grand["person_detection_s"] += sum(r["person_detection_s"] for r in rows)
    grand["face_detect_embed_s"] += sum(r["face_detect_embed_s"] for r in rows)
    grand["embedding_matching_ms"] += sum(r["embedding_matching_ms"] for r in rows)
    grand["total_s"] += total_all
    grand_n += n

overall_avg_pd = grand["person_detection_s"] / grand_n
overall_avg_fde = grand["face_detect_embed_s"] / grand_n
overall_avg_em_ms = grand["embedding_matching_ms"] / grand_n
overall_avg_tot = grand["total_s"] / grand_n
overall_row = ["Overall", grand_n, round(overall_avg_pd, 6), round(overall_avg_fde, 6),
               round(overall_avg_em_ms, 4), round(overall_avg_tot, 6), round(grand["total_s"], 4)]
ws_summary.append(overall_row)
for col in range(1, len(summary_headers) + 1):
    ws_summary.cell(row=ws_summary.max_row, column=col).font = Font(bold=True)

for col, width in zip("ABCDEFG", (10, 14, 22, 22, 22, 14, 20)):
    ws_summary.column_dimensions[col].width = width

os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
wb.save(OUT_XLSX)
print(f"\n\nSaved {OUT_XLSX}")
