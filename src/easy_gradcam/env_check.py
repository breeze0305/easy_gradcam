# --- OpenCV ---
try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required but not installed. "
        "Please install it via `pip install opencv-python`."
    ) from e

# --- Matplotlib ---
try:
    import matplotlib
except ImportError as e:
    raise ImportError(
        "Matplotlib is required but not installed. "
        "Please install it via `pip install matplotlib`."
    ) from e

# --- Seaborn ---
try:
    import seaborn
except ImportError as e:
    raise ImportError(
        "Seaborn is required but not installed. "
        "Please install it via `pip install seaborn`."
    ) from e
