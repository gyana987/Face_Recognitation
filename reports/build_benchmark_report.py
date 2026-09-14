"""Builds the performance/benchmark Word report from reports/benchmark_results.json
(produced by reports/run_benchmark.py). Plain/no-color style.

Usage:
    python reports/run_benchmark.py          # measures real numbers on this machine
    python reports/build_benchmark_report.py  # turns them into a .docx
"""
import json
import os

from docx import Document
from docx.shared import Pt, RGBColor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "reports", "benchmark_results.json")
OUT_DOCX = os.path.join(BASE_DIR, "reports", "performance_benchmark_report.docx")
BLACK = RGBColor(0, 0, 0)

if not os.path.exists(RESULTS_PATH):
    raise SystemExit(f"{RESULTS_PATH} not found -- run reports/run_benchmark.py first.")

with open(RESULTS_PATH) as f:
    R = json.load(f)

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.font.color.rgb = BLACK


def h1(t):
    p = doc.add_heading(t, level=1)
    for r in p.runs:
        r.font.color.rgb = BLACK


def h2(t):
    p = doc.add_heading(t, level=2)
    for r in p.runs:
        r.font.color.rgb = BLACK


def para(t, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    r.font.color.rgb = BLACK
    return p


def bullet(t):
    p = doc.add_paragraph(t, style="List Bullet")
    for r in p.runs:
        r.font.color.rgb = BLACK


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = BLACK
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    doc.add_paragraph()


# ============================================================================
title = doc.add_heading("Performance Test Report", level=0)
for r in title.runs:
    r.font.color.rgb = BLACK
para("Missing-person search pipeline (YOLOv8 + InsightFace buffalo_l) -- measured, not estimated, "
     "on this machine's actual GPU using real footage.", italic=True)
doc.add_paragraph()

h1("Test environment")
src = R.get("source_videos_used", {})
table(
    ["Item", "Value"],
    [
        ["Compute device used", R["device"]],
        ["Idle GPU memory (before any model loaded)", f"{R['gpu_mem_idle_mb']} MB"],
        ["Test footage (single-source tests)", src.get("single_source", "n/a")],
        ["Test footage (multi-camera scaling test)", ", ".join(src.get("scaling_test", []))],
        ["Person detector", "YOLOv8n (yolov8n.pt)"],
        ["Face model", "InsightFace buffalo_l (all 5 sub-models)"],
    ],
)
para("Caveat: results depend on this machine's specific GPU (check `nvidia-smi` for its model/VRAM). A "
     "GPU with more VRAM or more compute will be meaningfully faster, especially for the multi-camera "
     "scaling results in section C, where several camera threads share one GPU.", italic=True)

# ============================================================================
h1("A. Model size and footprint")
ms = R["model_size"]
table(
    ["Component", "Disk size"],
    [[k, f"{v} MB"] for k, v in ms["buffalo_l_submodels"].items()] +
    [["buffalo_l TOTAL", f"{ms['buffalo_l_disk_mb']} MB"],
     ["YOLOv8n (yolov8n.pt)", f"{ms['yolov8n_disk_mb']} MB"]],
)

table(
    ["Measurement", "Value"],
    [
        ["YOLOv8n load time (cold start)", f"{ms['yolo_load_time_s']} s"],
        ["InsightFace buffalo_l load time (cold start)", f"{ms['insightface_load_time_s']} s"],
        ["GPU memory -- idle", f"{ms['gpu_mem_idle_mb']} MB"],
        ["GPU memory -- after YOLOv8n loaded", f"{ms['gpu_mem_after_yolo_mb']} MB"],
        ["GPU memory -- after buffalo_l also loaded", f"{ms['gpu_mem_after_insightface_mb']} MB"],
        ["YOLOv8n GPU footprint", f"{ms['yolo_gpu_footprint_mb']} MB"],
        ["buffalo_l GPU footprint", f"{ms['insightface_gpu_footprint_mb']} MB"],
    ],
)
cold_start = round(ms["yolo_load_time_s"] + ms["insightface_load_time_s"], 1)
para(f"Takeaway: buffalo_l is the heavier model by far on disk ({ms['buffalo_l_disk_mb']} MB vs. "
     f"YOLOv8n's {ms['yolov8n_disk_mb']} MB) and in GPU memory (~{ms['insightface_gpu_footprint_mb']} MB "
     f"vs. ~{ms['yolo_gpu_footprint_mb']} MB for YOLO). Total cold-start time to be ready for the first "
     f"frame is about {cold_start} seconds.")

# ============================================================================
h1("B. Resolution vs. performance/latency")
rl = R["resolution_latency"]
table(
    ["Resolution", "YOLO imgsz", "Avg. people detected", "Detection (ms)", "Face stage (ms)",
     "Total (ms)", "Effective FPS", "GPU memory (MB)"],
    [[k, v["yolo_imgsz"], v["avg_people_detected"], v["detection_ms_mean"], v["face_stage_ms_mean"],
      v["total_ms_mean"], v["fps_equivalent"], v["gpu_mem_used_mb"]] for k, v in rl.items()],
)
para("Caveat: resolution and person-count are not perfectly isolated here -- YOLO's own working "
     "resolution (imgsz) is raised for 4K to match this project's real config behavior, which can "
     "additionally catch more small/distant people than at 720p/1080p. Check the 'avg. people detected' "
     "column above before comparing rows directly.", italic=True)
para("Takeaway: detection time (YOLO) changes only slightly with resolution. The face-recognition stage "
     "dominates total latency because it runs buffalo_l once per detected person -- see section E for how "
     "strongly that scales with crowd size versus resolution alone.")

# ============================================================================
h1("C. Single-camera vs. multi-camera")
cs = R["camera_scaling"]
rows = []
for key in sorted(cs.keys(), key=lambda k: int(k.split("_")[0])):
    v = cs[key]
    n = key.split("_")[0]
    rows.append([n, v["wall_time_s"], v["avg_per_camera_fps"], v["combined_throughput_fps"]])
table(["Cameras running concurrently", "Wall time for test batch (s)",
       "Avg. per-camera FPS", "Combined throughput (FPS, all cameras)"], rows)

para("Per-camera detail:")
for key in sorted(cs.keys(), key=lambda k: int(k.split("_")[0])):
    v = cs[key]
    per_cam_rows = [[cam, d["frames"], d["mean_ms"], d["fps"]] for cam, d in v["per_camera"].items()]
    para(f"{key.replace('_', ' ')}:", bold=True)
    table(["Camera", "Frames", "Mean latency (ms)", "FPS"], per_cam_rows)

one_cam_fps = cs.get("1_camera(s)", {}).get("avg_per_camera_fps")
max_n = max(cs.keys(), key=lambda k: int(k.split("_")[0]))
max_cam_fps = cs[max_n]["avg_per_camera_fps"]
para(f"Takeaway: going from 1 to {max_n.split('_')[0]} concurrent cameras on this GPU drops average "
     f"per-camera FPS from {one_cam_fps} to {max_cam_fps} -- all cameras share one GPU and queue for its "
     "time. Combined throughput across all cameras is worth comparing to the single-camera number above to "
     "see whether the GPU is doing roughly the same total work either way (flat combined throughput) or "
     "genuinely falling behind as more cameras are added.")

# ============================================================================
h1("D. Per-face recognition time (isolated)")
pf = R["per_face_latency"]
table(
    ["Measurement", "Value"],
    [
        ["Face crops tested", pf["num_crops_tested"]],
        ["Total calls timed", pf["num_calls"]],
        ["Mean latency per face", f"{pf['mean_ms']} ms"],
        ["Fastest call", f"{pf['min_ms']} ms"],
        ["Slowest call", f"{pf['max_ms']} ms"],
    ],
)
para(f"Takeaway: a single InsightFace call (detect + landmarks + age/gender + 512-d embedding, all 5 "
     f"sub-models) on ONE already-cropped person takes about {pf['mean_ms']} ms on average, isolated from "
     "YOLO/tracking overhead. This is the real per-person cost -- section B's much larger face-stage "
     "totals are this number multiplied by however many people are in frame.")

# ============================================================================
h1("E. What latency actually depends on")
h2("E.1 Person count")
pcd = R["person_count_dependency"]
table(
    ["Frame", "People detected", "Faces found", "Detection (ms)", "Face stage (ms)", "Total (ms)"],
    [[d["frame"], d["people_detected"], d["faces_found"], d["detection_ms"], d["face_stage_ms"], d["total_ms"]]
     for d in pcd],
)
para("Takeaway: detection time stays roughly flat regardless of how many people are in frame -- YOLO "
     "processes the whole frame once. The face stage, in contrast, scales directly with how many people "
     "were detected, confirming total latency is overwhelmingly driven by PERSON COUNT, not frame "
     "resolution or pixel count on their own.")

h2("E.2 Face pixel size / how much upscaling was needed")
mfp = R["min_face_pixel"]
para(f"Base test subject: a real detected face, approximately {mfp['base_face_px']} px, from a person crop "
     f"of size {mfp['base_person_crop_shape'][1]}x{mfp['base_person_crop_shape'][0]} pixels with real "
     "background context (not an artificial crop) -- progressively shrunk to simulate the same person "
     "moving farther from the camera.")
table(
    ["Shrink factor", "Person crop size", "Approx. face px (before upscale)", "Detected (raw)",
     "Raw det_score", "Upscale applied", "Detected (with pipeline's upscale gate)",
     "det_score (upscaled)", "Resulting face px"],
    [[d["shrink_factor"], d["person_crop_size"], d["approx_face_px_before_upscale"], d["raw_detected"],
      d["raw_det_score"], d["with_pipeline_upscale_gate"]["upscale_applied"],
      d["with_pipeline_upscale_gate"]["detected"], d["with_pipeline_upscale_gate"]["det_score"],
      d["with_pipeline_upscale_gate"]["resulting_face_px"]]
     for d in mfp["shrink_test"]],
)
para("Takeaway: with REAL background context (as every actual detection has -- a face never appears "
     "alone on a black background), the model keeps detecting a face down to a very small raw pixel size, "
     "with confidence degrading gradually rather than failing sharply at one cutoff. The pipeline's own "
     "upscale gate (UPSCALE_BELOW/MAX_UPSCALE in config) automatically recovers detail for small/distant "
     "crops, keeping the resulting face above the MIN_FACE_PX quality gate for most of the range tested.")

# ============================================================================
h1("F. Minimum detectable face size / distance")
para("From section E.2's real-context test:")
bullet("Practical minimum face size for reliable detection (det_score comfortably high): see the "
       "'raw_det_score' column above for where confidence starts dropping.")
bullet("With upscaling applied (as the real pipeline always does for small crops), usable detections were "
       "obtained at even smaller raw face sizes -- though confidence is lower the smaller the original "
       "face was, so very distant people are still lower-confidence matches even after upscaling.")
bullet("Current pipeline gate: MIN_FACE_PX = 28px (after upscaling) -- compare against the "
       "'resulting_face_px' column above to judge whether this threshold is conservative or aggressive for "
       "your footage.")

para("Translating pixel size to real-world distance requires your camera's actual specs (resolution and "
     "field-of-view/focal length) -- this was not measured because it depends on camera hardware, not the "
     "software. As a worked EXAMPLE only (not a measurement): a typical 1080p camera with a ~90-degree "
     "horizontal field of view resolves roughly 21 pixels per degree; a human face is about 15-16 cm wide, "
     "which subtends roughly 1.5 degrees at 6 meters -- implying a face would be roughly 30 px wide at "
     "about 6 meters under those specific assumptions. This is illustrative only; provide your actual "
     "camera's resolution and FOV/lens spec for a real number.", italic=True)

# ============================================================================
h1("Summary of key numbers")
table(
    ["Question", "Answer"],
    [
        ["buffalo_l total size", f"{ms['buffalo_l_disk_mb']} MB disk, ~{ms['insightface_gpu_footprint_mb']} MB GPU memory"],
        ["Cold start time", f"~{cold_start} s"],
        ["Per-face recognition time", f"~{pf['mean_ms']} ms"],
        ["1 camera throughput", f"{one_cam_fps} FPS"],
        [f"{max_n.split('_')[0]} cameras combined throughput",
         f"{cs[max_n]['combined_throughput_fps']} FPS ({max_cam_fps} FPS/camera)"],
        ["What latency depends on most", "Number of people in frame -- NOT resolution or pixel count directly"],
        ["Minimum usable face size", "See section F -- current MIN_FACE_PX gate is 28px after upscaling"],
    ],
)

doc.save(OUT_DOCX)
print("Saved:", OUT_DOCX)
