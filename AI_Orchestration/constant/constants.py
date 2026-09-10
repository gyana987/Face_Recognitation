"""Config key names for the AI (detection) stage. Mirrors AI_VMS's constant/constants.py
-- only the keys this usecase's model actually reads; AI_VMS's real file has many more."""


class Constants:
    DEVICE_KEY = "DEVICE"
    PERSON_CONF_KEY = "PERSON_CONF"
    PERSON_IMGSZ_KEY = "PERSON_IMGSZ"
    TRACKER_KEY = "TRACKER"
