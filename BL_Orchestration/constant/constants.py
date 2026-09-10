"""Config key names for the MISSING_PERSON_FACE_EXTRACTION BL usecase."""


class Constants:
    DEVICE_KEY = "DEVICE"
    DET_SIZE_W_KEY = "DET_SIZE_W"
    DET_SIZE_H_KEY = "DET_SIZE_H"
    MIN_FACE_PX_KEY = "MIN_FACE_PX"
    FACE_DET_CONF_KEY = "FACE_DET_CONF"
    BLUR_MIN_KEY = "BLUR_MIN"
    UPSCALE_BELOW_KEY = "UPSCALE_BELOW"
    MAX_UPSCALE_KEY = "MAX_UPSCALE"
    BEST_FACES_DIR_KEY = "BEST_FACES_DIR"

    # Face-embedding ReID (merges a tracker's unstable track_id into a
    # stable global_person_id via cosine similarity between ArcFace embeddings)
    REID_THRESHOLD_KEY = "REID_THRESHOLD"

    # Composite quality-score weights
    SCORE_W_CONF_KEY = "SCORE_W_CONF"
    SCORE_W_SHARP_KEY = "SCORE_W_SHARP"
    SCORE_W_SIZE_KEY = "SCORE_W_SIZE"
    SCORE_W_FRONTAL_KEY = "SCORE_W_FRONTAL"

    # Score normalization reference points
    SHARP_REF_MAX_KEY = "SHARP_REF_MAX"
    SIZE_REF_MAX_KEY = "SIZE_REF_MAX"
