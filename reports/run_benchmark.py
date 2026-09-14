"""Runs the real performance benchmark suite for the missing-person search
pipeline (model size/footprint, resolution latency, single vs multi-camera
scaling, per-face latency, person-count dependency, minimum face pixel
size) on THIS machine's actual GPU, using real footage from Wildtrack/.

Does not modify any existing project file -- only imports read-only building
blocks from search_multi_camera.py (crop_upscaled, FaceMatcher, etc.).

Usage:
    python reports/run_benchmark.py
    (then run reports/build_benchmark_report.py to turn the results into a
     Word document)

Output: reports/benchmark_results.json
"""
import glob
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)


def _ensure_gpu_libs_on_path():
    if os.environ.get("_PFP_GPU_LIBS_SET") == "1":
        return
    lib_dirs = []
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
import numpy as np

from search_multi_camera import resolve_device, load_person_model, FaceMatcher, crop_upscaled

RESULTS_PATH = os.path.join(BASE_DIR, "reports", "benchmark_results.json")

# Auto-discover Wildtrack video files rather than hardcoding names -- these
# get renamed/added/removed over time as the folder is used for demos.
WILDTRACK_VIDEOS = sorted(glob.glob(os.path.join(BASE_DIR, "Wildtrack", "*.mp4")))
if not WILDTRACK_VIDEOS:
    raise SystemExit("No .mp4 files found in Wildtrack/ -- point SOURCE_VIDEOS at some real footage first.")
SOURCE_VIDEO = WILDTRACK_VIDEOS[0]  # used for the single-source tests (B, D, E, F)
SCALING_VIDEOS = WILDTRACK_VIDEOS[:4] if len(WILDTRACK_VIDEOS) >= 4 else \
    (WILDTRACK_VIDEOS * 4)[:4]  # reuse videos if fewer than 4 are available

results = {}


def gpu_mem_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


# ---------------------------------------------------------------------------
print("== A. Model size / memory footprint ==")
mem_before_any = gpu_mem_used_mb()

device = resolve_device("auto")
results["device"] = device
results["gpu_mem_idle_mb"] = mem_before_any
results["source_videos_used"] = {"single_source": SOURCE_VIDEO, "scaling_test": SCALING_VIDEOS}

t0 = time.time()
person_model = load_person_model(device)
mem_after_yolo = gpu_mem_used_mb()
yolo_load_s = time.time() - t0

t0 = time.time()
matcher = FaceMatcher(device, (640, 640), 28)
mem_after_insightface = gpu_mem_used_mb()
insightface_load_s = time.time() - t0

buffalo_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
buffalo_disk_mb = sum(os.path.getsize(os.path.join(buffalo_dir, f))
                      for f in os.listdir(buffalo_dir) if f.endswith(".onnx")) / (1024 * 1024)
yolo_disk_mb = os.path.getsize(os.path.join(BASE_DIR, "AI_Orchestration", "Detection", "YOLO",
                                             "yolov8n.pt")) / (1024 * 1024)

results["model_size"] = {
    "buffalo_l_disk_mb": round(buffalo_disk_mb, 1),
    "buffalo_l_submodels": {
        f: round(os.path.getsize(os.path.join(buffalo_dir, f)) / 1e6, 1)
        for f in sorted(os.listdir(buffalo_dir)) if f.endswith(".onnx")
    },
    "yolov8n_disk_mb": round(yolo_disk_mb, 1),
    "yolo_load_time_s": round(yolo_load_s, 2),
    "insightface_load_time_s": round(insightface_load_s, 2),
    "gpu_mem_idle_mb": mem_before_any,
    "gpu_mem_after_yolo_mb": mem_after_yolo,
    "gpu_mem_after_insightface_mb": mem_after_insightface,
    "yolo_gpu_footprint_mb": mem_after_yolo - mem_before_any,
    "insightface_gpu_footprint_mb": mem_after_insightface - mem_after_yolo,
}
print(json.dumps(results["model_size"], indent=2))

# ---------------------------------------------------------------------------
print("\n== B. Resolution latency (720p / 1080p / 4K) ==")
cap = cv2.VideoCapture(SOURCE_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
ok, base_frame = cap.read()
cap.release()
if not ok:
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    ok, base_frame = cap.read()
    cap.release()
assert ok, f"Could not read a frame from {SOURCE_VIDEO}"

resolutions = {"720p": (1280, 720), "1080p": (1920, 1080), "4K": (3840, 2160)}
N_WARMUP, N_RUNS = 3, 20

res_results = {}
for name, (w, h) in resolutions.items():
    frame = cv2.resize(base_frame, (w, h))
    imgsz = 960 if name != "4K" else 1280

    for _ in range(N_WARMUP):
        person_model.run_inference([frame], {"tracking": False, "confidence": 0.25, "imgsz": imgsz})

    det_times, face_times, person_counts = [], [], []
    for _ in range(N_RUNS):
        t0 = time.time()
        dets = person_model.run_inference([frame], {"tracking": False, "confidence": 0.25, "imgsz": imgsz})[0]
        det_times.append(time.time() - t0)
        person_counts.append(len(dets))

        t0 = time.time()
        for det in dets:
            x1, y1, x2, y2 = map(int, det["xyxy"])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            up, _ = crop_upscaled(crop, 220, 4.0)
            matcher.best_face_embedding(up)
        face_times.append(time.time() - t0)

    total_times = [d + f for d, f in zip(det_times, face_times)]
    res_results[name] = {
        "resolution": f"{w}x{h}", "yolo_imgsz": imgsz,
        "avg_people_detected": round(sum(person_counts) / len(person_counts), 1),
        "detection_ms_mean": round(1000 * sum(det_times) / len(det_times), 1),
        "face_stage_ms_mean": round(1000 * sum(face_times) / len(face_times), 1),
        "total_ms_mean": round(1000 * sum(total_times) / len(total_times), 1),
        "fps_equivalent": round(1.0 / (sum(total_times) / len(total_times)), 2),
        "gpu_mem_used_mb": gpu_mem_used_mb(),
    }
    print(name, res_results[name])

results["resolution_latency"] = res_results

# ---------------------------------------------------------------------------
print("\n== C. Single vs multi-camera scaling (1-4 concurrent cameras) ==")

def run_n_frames(video_path, n_frames, tracker_key, out):
    cap = cv2.VideoCapture(video_path)
    times = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        dets = person_model.run_inference(
            [frame], {"tracking": True, "confidence": 0.25, "imgsz": 960, "tracker_key": tracker_key})[0]
        for det in dets:
            x1, y1, x2, y2 = map(int, det["xyxy"])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            up, _ = crop_upscaled(crop, 220, 4.0)
            matcher.best_face_embedding(up)
        times.append(time.time() - t0)
    cap.release()
    out[tracker_key] = times


def summarize(times):
    return {"frames": len(times), "mean_ms": round(1000 * sum(times) / len(times), 1),
            "fps": round(len(times) / sum(times), 2)}


N_FRAMES = 40
scaling_results = {}
for n_cams in (1, 2, 3, 4):
    videos = SCALING_VIDEOS[:n_cams]
    out = {}
    threads = [threading.Thread(target=run_n_frames, args=(v, N_FRAMES, f"cam{i+1}", out))
               for i, v in enumerate(videos)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    per_cam = {k: summarize(v) for k, v in out.items()}
    scaling_results[f"{n_cams}_camera(s)"] = {
        "cameras_used": videos, "wall_time_s": round(wall, 2), "per_camera": per_cam,
        "combined_throughput_fps": round((N_FRAMES * n_cams) / wall, 2),
        "avg_per_camera_fps": round(sum(p["fps"] for p in per_cam.values()) / len(per_cam), 2),
    }
    print(f"{n_cams} camera(s):", json.dumps(scaling_results[f"{n_cams}_camera(s)"], indent=2))

results["camera_scaling"] = scaling_results

# ---------------------------------------------------------------------------
print("\n== D. Per-face recognition latency (isolated) ==")
cap = cv2.VideoCapture(SOURCE_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
ok, frame = cap.read()
cap.release()
dets = person_model.run_inference([frame], {"tracking": False, "confidence": 0.25, "imgsz": 960})[0]
crops = []
for det in dets[:10]:
    x1, y1, x2, y2 = map(int, det["xyxy"])
    c = frame[max(0, y1):y2, max(0, x1):x2]
    if c.size > 0:
        up, _ = crop_upscaled(c, 220, 4.0)
        crops.append(up)

face_only_times = []
for c in crops:
    for _ in range(5):
        t0 = time.time()
        matcher.best_face_embedding(c)
        face_only_times.append(time.time() - t0)

results["per_face_latency"] = {
    "num_crops_tested": len(crops), "num_calls": len(face_only_times),
    "mean_ms": round(1000 * sum(face_only_times) / len(face_only_times), 2) if face_only_times else None,
    "min_ms": round(1000 * min(face_only_times), 2) if face_only_times else None,
    "max_ms": round(1000 * max(face_only_times), 2) if face_only_times else None,
}
print(json.dumps(results["per_face_latency"], indent=2))

# ---------------------------------------------------------------------------
print("\n== E. Dependency on person count ==")
sample_positions = [500, 2000, 4000, 6000, 8000, 10000, 12000, 15000]
dep_results = []
cap = cv2.VideoCapture(SOURCE_VIDEO)
for pos in sample_positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, frame = cap.read()
    if not ok:
        continue
    t0 = time.time()
    dets = person_model.run_inference([frame], {"tracking": False, "confidence": 0.25, "imgsz": 960})[0]
    det_t = time.time() - t0

    t0 = time.time()
    n_faces = 0
    for det in dets:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        up, _ = crop_upscaled(crop, 220, 4.0)
        emb, _, _ = matcher.best_face_embedding(up)
        if emb is not None:
            n_faces += 1
    face_t = time.time() - t0

    dep_results.append({"frame": pos, "people_detected": len(dets), "faces_found": n_faces,
                         "detection_ms": round(det_t * 1000, 1), "face_stage_ms": round(face_t * 1000, 1),
                         "total_ms": round((det_t + face_t) * 1000, 1)})
cap.release()
results["person_count_dependency"] = dep_results
for r in dep_results:
    print(r)

# ---------------------------------------------------------------------------
print("\n== F. Minimum face pixel size (real background context, not artificial padding) ==")
# A tight face-only crop on black padding gives FALSE negatives (the
# detector relies on surrounding context) -- use a full PERSON crop,
# shrunk as a whole, so context shrinks proportionally like a real person
# walking farther from camera.
matcher_nogate = FaceMatcher(device, (640, 640), 1)  # min_face_px=1 -- see the raw score, not the gate

best_person_crop, best_face_px = None, -1
cap = cv2.VideoCapture(SOURCE_VIDEO)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 18000)
for pos in range(0, frame_count, max(1, frame_count // 12)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, frame = cap.read()
    if not ok:
        continue
    dets = person_model.run_inference([frame], {"tracking": False, "confidence": 0.25, "imgsz": 960})[0]
    for det in dets:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        emb, bbox, score = matcher_nogate.best_face_embedding(crop)
        if emb is None:
            continue
        fx1, fy1, fx2, fy2 = bbox
        face_px = min(fx2 - fx1, fy2 - fy1)
        if face_px > best_face_px:
            best_face_px, best_person_crop = face_px, crop.copy()
cap.release()
print(f"Base test subject: face ~{best_face_px}px, "
      f"person crop {best_person_crop.shape if best_person_crop is not None else None}")

min_face_pixel = {"base_face_px": best_face_px,
                   "base_person_crop_shape": list(best_person_crop.shape) if best_person_crop is not None else None,
                   "shrink_test": []}
if best_person_crop is not None:
    orig_h, orig_w = best_person_crop.shape[:2]
    for shrink_factor in [1.0, 0.8, 0.6, 0.5, 0.4, 0.32, 0.25, 0.2, 0.15, 0.1, 0.07]:
        new_w, new_h = max(1, int(orig_w * shrink_factor)), max(1, int(orig_h * shrink_factor))
        shrunk = cv2.resize(best_person_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        approx_face_px = round(best_face_px * shrink_factor, 1)

        emb_raw, bbox_raw, score_raw = matcher_nogate.best_face_embedding(shrunk)
        upscaled, scale = crop_upscaled(shrunk, 220, 4.0)
        emb_up, bbox_up, score_up = matcher_nogate.best_face_embedding(upscaled)

        min_face_pixel["shrink_test"].append({
            "shrink_factor": shrink_factor, "person_crop_size": f"{new_w}x{new_h}",
            "approx_face_px_before_upscale": approx_face_px,
            "raw_detected": emb_raw is not None,
            "raw_det_score": round(float(score_raw), 3) if score_raw else None,
            "with_pipeline_upscale_gate": {
                "upscale_applied": round(scale, 2), "detected": emb_up is not None,
                "det_score": round(float(score_up), 3) if score_up else None,
                "resulting_face_px": (min(bbox_up[2] - bbox_up[0], bbox_up[3] - bbox_up[1]) if bbox_up else None),
            },
        })
results["min_face_pixel"] = min_face_pixel
for r in min_face_pixel["shrink_test"]:
    print(r)

# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nSaved {RESULTS_PATH}")
print("Now run: python reports/build_benchmark_report.py")
