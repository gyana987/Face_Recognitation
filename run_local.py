"""Local, no-infra runner for the missing-person face-extraction pipeline.

AI_VMS wires Detection -> BL -> Event stages together across three separate
RabbitMQ-fed containers, each dynamically loading the right module by
folder-name convention (see orchestrator.py / bl_orchestrator.py /
event_orchestrator.py in AI_VMS). This script does the exact same three
dynamic loads and the same run_inference -> run_logic -> run_event_logic call
chain, but in a single process reading a local video file - so
AI_Orchestration/Detection/YOLO/inference.py, BL_Orchestration's
MISSING_PERSON_FACE_EXTRACTION/missing_person_face_extraction.py, and
Event_Manager's MISSING_PERSON_FACE_EXTRACTION/missing_person_face_extraction.py
can be developed/tested here and then copied into AI_VMS UNCHANGED (see
README.md for exact destination paths). Only this wiring file goes away when
porting - AI_VMS's existing orchestrators replace it.

Usage:
    python run_local.py --video input/video.mp4 --output output/annotated_video.mp4
"""

import argparse
import importlib.util
import logging
import os
import sys


def _ensure_gpu_libs_on_path():
    """
    torch/onnxruntime ship their own CUDA/cuDNN .so files inside the
    nvidia.* pip packages, but nothing puts that directory on
    LD_LIBRARY_PATH -- and the dynamic linker only reads that variable at
    process startup, so setting os.environ from inside a running process
    has no effect. Re-exec once with it set correctly so GPU inference
    (cudnn conv kernels, onnxruntime's CUDAExecutionProvider) can find its
    libraries instead of silently falling back to CPU. Must run before any
    torch/ultralytics/onnxruntime import anywhere in the process -- this is
    local-machine-specific and not something that ports into AI_VMS.
    """
    if os.environ.get("_PFP_GPU_LIBS_SET") == "1":
        return
    lib_dirs = []
    for pkg in ("nvidia.cu13", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            lib_dirs.append(os.path.join(list(spec.submodule_search_locations)[0], "lib"))
    if not lib_dirs:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_PFP_GPU_LIBS_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_gpu_libs_on_path()

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All three stages define their own "constant" and "services" packages
# (matching AI_VMS's real per-stage layout, where each stage runs in its own
# container/process and so never collides with the others). This script
# loads all three into ONE process for local testing, so without care,
# Python's module cache would let whichever stage's "constant"/"services"
# loads FIRST silently win for every subsequent `from constant.constants
# import Constants` in the other stages. _load_stage_module loads each
# stage's usecase script with only that stage's folder on sys.path, then
# purges the shared package names from sys.modules and sys.path afterward so
# the next stage starts clean.
_SHARED_PACKAGE_NAMES = ("constant", "constant.constants", "services", "services.config_reader")


def _load_stage_module(stage_dir, rel_module_path, module_name):
    stage_path = os.path.join(BASE_DIR, stage_dir)
    sys.path.insert(0, stage_path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(BASE_DIR, rel_module_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(stage_path)
        for name in _SHARED_PACKAGE_NAMES:
            sys.modules.pop(name, None)


def main():
    parser = argparse.ArgumentParser(description="Local Detection -> BL -> Event pipeline runner.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="output/annotated_video.mp4",
                         help="Path to save the annotated output video")
    parser.add_argument("--camera-id", default="local-cam-1", help="Camera id used for BL/Event per-camera state")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no cv2.imshow window)")
    args = parser.parse_args()

    ai_module = _load_stage_module(
        "AI_Orchestration",
        os.path.join("AI_Orchestration", "Detection", "YOLO", "inference.py"),
        "yolo_inference",
    )
    bl_module = _load_stage_module(
        "BL_Orchestration",
        os.path.join("BL_Orchestration", "Usecases_Core_logics", "MISSING_PERSON_FACE_EXTRACTION",
                     "missing_person_face_extraction.py"),
        "missing_person_face_extraction_bl",
    )
    event_module = _load_stage_module(
        "Event_Manager",
        os.path.join("Event_Manager", "Usecases_Event_logics", "MISSING_PERSON_FACE_EXTRACTION",
                     "missing_person_face_extraction.py"),
        "missing_person_face_extraction_event",
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps / args.every, (frame_w, frame_h))

    frame_idx = 0
    frames_with_face = 0
    total_faces = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.every != 0:
            continue

        detections_per_frame = ai_module.run_inference([frame], {"tracking": True, "tracker_key": args.camera_id})

        batch = {
            "camera_metadata": {"camera_id": args.camera_id},
            "frame_metadata": {"frame": frame, "fps": fps / args.every},
            "inference_results": detections_per_frame,
        }

        bl_module.run_logic(batch)
        event_module.run_event_logic(batch, "MISSING_PERSON_FACE_EXTRACTION")

        # BL re-encodes frame_metadata["frame"] to jpg bytes (matching real
        # AI_VMS message shape) -- decode back to a displayable/writable frame.
        annotated = batch["frame_metadata"]["frame"]
        if isinstance(annotated, (bytes, bytearray)):
            annotated = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)

        faces = batch["bl_results"].get("faces", [])
        if faces:
            frames_with_face += 1
            total_faces += len(faces)

        if writer:
            writer.write(annotated)
        if not args.no_display:
            cv2.imshow("Missing-Person Face Extraction (local)", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logging.info("Done -- frames with a usable face: %d, total faces saved: %d", frames_with_face, total_faces)
    if args.output:
        logging.info("Annotated video saved to %s", args.output)


if __name__ == "__main__":
    main()
