#!/usr/bin/env python3
"""Measure Boltzmann's constant from Millikan oil-drop Brownian-motion videos."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


cv2: Any = None
np: Any = None
pd: Any = None
plt: Any = None
optimize: Any = None
scipy_stats: Any = None
trackpy_module: Any = None

STANDARD_GRAVITY_M_S2 = 9.80665
TRUE_BOLTZMANN_J_K = 1.380649e-23


class AnalysisError(RuntimeError):
    """Raised when the analysis cannot proceed with the provided inputs."""


@dataclass
class VideoInfo:
    path: str
    fps: float
    native_fps: float
    frame_count: int
    width_px: int
    height_px: int


@dataclass
class PhysicsParams:
    temperature_k: float
    pressure_pa: float
    oil_density_kgm3: float
    air_density_kgm3: float
    eta0_pa_s: float
    eta0_reference_k: float
    sutherland_c_k: float
    viscosity_model: str
    cunningham_b_pa_m: float


@dataclass
class TrackingConfig:
    tracker: str
    roi_size_px: int
    search_range_px: int
    trackpy_diameter_px: int
    trackpy_minmass: float | None
    trackpy_memory_frames: int
    selection_radius_px: float
    particle_polarity: str
    clahe: bool
    blur_kernel_px: int
    max_frames: int | None
    bg_frames: int = 30


@dataclass
class RadiusResult:
    drop_id: str
    fall_time_s: float
    frame_count: int
    vertical_speed_um_s: float
    terminal_velocity_m_s: float
    radius_m: float
    radius_um: float
    cunningham_factor: float
    eta_air_pa_s: float
    eta_slip_corrected_pa_s: float
    fit_r2: float


def load_runtime_dependencies(*, need_video: bool, use_agg_backend: bool) -> None:
    global cv2, np, pd, plt, optimize, scipy_stats, trackpy_module

    missing_modules: list[str] = []
    loaded_modules: dict[str, Any] = {}
    required_modules = ["numpy", "pandas", "scipy.optimize", "scipy.stats", "matplotlib"]
    if need_video:
        required_modules.append("cv2")

    for module_name in required_modules:
        try:
            loaded_modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)

    if missing_modules:
        install_names = ["opencv-python" if module_name == "cv2" else module_name.split(".")[0] for module_name in missing_modules]
        unique_install_names = sorted(set(install_names))
        raise SystemExit(
            "Missing required Python package(s): "
            + ", ".join(missing_modules)
            + "\nInstall them, for example:\n  python -m pip install "
            + " ".join(unique_install_names)
        )

    np = loaded_modules["numpy"]
    pd = loaded_modules["pandas"]
    optimize = loaded_modules["scipy.optimize"]
    scipy_stats = loaded_modules["scipy.stats"]
    if need_video:
        cv2 = loaded_modules["cv2"]

    matplotlib_module = loaded_modules["matplotlib"]
    if use_agg_backend:
        matplotlib_module.use("Agg", force=True)
    plt = importlib.import_module("matplotlib.pyplot")

    try:
        trackpy_module = importlib.import_module("trackpy")
        if hasattr(trackpy_module, "quiet"):
            trackpy_module.quiet()
    except ImportError:
        trackpy_module = None


class VideoSource:
    def __init__(self, path: Path, fps_override: float | None = None) -> None:
        if cv2 is None:
            raise AnalysisError("OpenCV is not loaded.")
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise AnalysisError(f"Could not open video: {path}")

        native_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width_px = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_px = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(fps_override if fps_override else native_fps)
        if fps <= 0:
            raise AnalysisError(f"FPS is unavailable for {path}; pass --fps.")

        self.info = VideoInfo(
            path=str(path),
            fps=fps,
            native_fps=native_fps,
            frame_count=frame_count,
            width_px=width_px,
            height_px=height_px,
        )

    def read_frame(self, frame_index: int = 0) -> Any:
        if frame_index < 0:
            raise AnalysisError("Frame index must be non-negative.")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = self.capture.read()
        if not success or frame is None:
            raise AnalysisError(f"Could not read frame {frame_index} from {self.path}.")
        return frame

    def iter_frames(self, *, start_frame: int = 0, max_frames: int | None = None) -> Iterable[tuple[int, Any]]:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        frame_index = int(start_frame)
        emitted_frames = 0
        while True:
            if max_frames is not None and emitted_frames >= max_frames:
                break
            success, frame = self.capture.read()
            if not success or frame is None:
                break
            yield frame_index, frame
            frame_index += 1
            emitted_frames += 1

    def release(self) -> None:
        self.capture.release()


def parse_point_list(raw_value: str | None) -> list[tuple[float, float]]:
    if not raw_value:
        return []
    points: list[tuple[float, float]] = []
    for item in raw_value.split(";"):
        stripped_item = item.strip()
        if not stripped_item:
            continue
        coordinates = [coordinate.strip() for coordinate in stripped_item.split(",")]
        if len(coordinates) != 2:
            raise AnalysisError(f"Invalid point '{item}'. Expected x,y; separated by semicolons.")
        points.append((float(coordinates[0]), float(coordinates[1])))
    return points


def parse_timer_roi(raw_value: str) -> tuple[int, int, int, int]:
    """Parse a timer ROI string ``'x,y,w,h'`` into a 4-tuple of ints."""
    parts = [v.strip() for v in raw_value.split(",")]
    if len(parts) != 4:
        raise AnalysisError(f"--timer-roi must be 'x,y,w,h'; got: {raw_value!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def parse_drop_ids(raw_value: str | None, count: int) -> list[str]:
    if not raw_value:
        return [str(drop_number) for drop_number in range(1, count + 1)]
    drop_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
    if len(drop_ids) != count:
        raise AnalysisError(f"Expected {count} drop id(s), got {len(drop_ids)}.")
    return drop_ids


def parse_lag_times(raw_value: str | None, max_lag_s: float, step_s: float) -> list[float]:
    if raw_value:
        lag_times = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    else:
        if max_lag_s <= 0 or step_s <= 0:
            raise AnalysisError("--max-lag-s and --dt-step-s must be positive.")
        lag_times = list(np.arange(step_s, max_lag_s + step_s * 0.25, step_s))
    lag_times = sorted({round(lag_time, 9) for lag_time in lag_times if lag_time > 0})
    if not lag_times:
        raise AnalysisError("No positive lag times were provided.")
    return lag_times


def dataframe_to_markdown(dataframe: Any) -> str:
    try:
        return dataframe.to_markdown(index=False)
    except Exception:
        return dataframe.to_string(index=False)


def save_table(dataframe: Any, output_dir: Path, stem: str) -> None:
    dataframe.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(dataframe_to_markdown(dataframe), encoding="utf-8")


def validate_positive(name: str, value: float) -> None:
    if value <= 0 or not math.isfinite(value):
        raise AnalysisError(f"{name} must be positive and finite.")


def validate_odd_positive(name: str, value: int) -> int:
    if value <= 0:
        raise AnalysisError(f"{name} must be positive.")
    if value % 2 == 0:
        value += 1
        print(f"Adjusted {name} to odd value {value} for particle localization.")
    return value


def preprocess_gray(frame: Any, config: TrackingConfig, background: Any = None) -> Any:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    if background is not None:
        # Subtract static background (removes grid lines for moving drops).
        gray_frame = np.clip(
            gray_frame.astype(np.float32) - background, 0.0, 255.0
        ).astype(np.uint8)
    else:
        # Per-row median subtraction: removes horizontal line structures (grid lines)
        # without requiring a pre-computed temporal background.
        row_med = np.median(gray_frame, axis=1, keepdims=True)
        gray_frame = np.clip(
            gray_frame.astype(np.float32) - row_med, 0.0, 255.0
        ).astype(np.uint8)
    if config.clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_frame = clahe.apply(gray_frame)
    blur_kernel_px = int(config.blur_kernel_px)
    if blur_kernel_px > 1:
        if blur_kernel_px % 2 == 0:
            blur_kernel_px += 1
        gray_frame = cv2.GaussianBlur(gray_frame, (blur_kernel_px, blur_kernel_px), 0)
    return gray_frame


def compute_video_background(video: VideoSource, n_frames: int, start_frame: int = 0) -> Any:
    """Return the pixel-wise median gray image over the first *n_frames* frames.

    Used to suppress static features (grid lines, apparatus markings) so that
    moving particles stand out after subtraction.
    """
    frames: list[Any] = []
    for _, frame in video.iter_frames(start_frame=start_frame, max_frames=n_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        frames.append(gray.astype(np.float32))
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)


def detect_timer_start(
    video: VideoSource,
    timer_roi: tuple[int, int, int, int],
    scan_frames: int = 300,
    change_threshold: float = 4.0,
    bright_thresh: int = 80,
) -> int:
    """Return the first frame index where the timer ROI starts changing.

    Uses binary thresholding on bright display pixels (> bright_thresh) to
    compare each frame's timer ROI against the frame-0 baseline.  Thresholding
    suppresses background motion noise so only actual digit changes are
    detected.  The noise level is estimated from the first ~30 idle frames.
    The first frame whose binary-diff-vs-baseline exceeds
    ``idle_mean + max(2*idle_std, 0.02)`` AND whose successor also exceeds
    ``idle_mean + max(idle_std, 0.01)`` is returned as the onset.
    Returns -1 if the onset is not found within *scan_frames*.
    """
    x0, y0, w_roi, h_roi = timer_roi
    # Build the reference ROI from frame 0 (frozen timer state).
    frame0 = video.read_frame(0)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY) if frame0.ndim == 3 else frame0.copy()
    h_img, w_img = gray0.shape[:2]
    y1_c = min(y0 + h_roi, h_img)
    x1_c = min(x0 + w_roi, w_img)
    baseline_bin = (gray0[y0:y1_c, x0:x1_c] > bright_thresh).astype(np.float32)
    diffs: list[float] = []
    frame_indices: list[int] = []
    for frame_index, frame in video.iter_frames(start_frame=0, max_frames=scan_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        roi_bin = (gray[y0:y1_c, x0:x1_c] > bright_thresh).astype(np.float32)
        if roi_bin.shape == baseline_bin.shape:
            diffs.append(float(np.mean(np.abs(roi_bin - baseline_bin))))
            frame_indices.append(frame_index)
    if len(diffs) < 5:
        return -1
    diffs_arr = np.array(diffs)
    n_base = min(30, max(1, len(diffs_arr) // 4))
    idle_mean = float(np.mean(diffs_arr[:n_base]))
    idle_std = max(float(np.std(diffs_arr[:n_base])), 0.001)
    threshold_high = idle_mean + max(2.0 * idle_std, 0.02)
    threshold_low = idle_mean + max(1.0 * idle_std, 0.01)
    for i in range(n_base, len(diffs_arr) - 1):
        if diffs_arr[i] > threshold_high and diffs_arr[i + 1] > threshold_low:
            return int(frame_indices[i])
    return -1


def find_drop_at_zero_line(
    frame: Any,
    zero_line_y_px: float,
    config: "TrackingConfig",
    background: Any = None,
    search_band_px: int = 30,
    x_hint_px: float | None = None,
    x_search_half_width_px: int = 250,
    x_max_px: int | None = None,
) -> tuple[float, float] | None:
    """Locate the oil drop nearest to the zero graduation line.

    Searches within *search_band_px* pixels of *zero_line_y_px* (and
    optionally within *x_search_half_width_px* of *x_hint_px*).  Pass
    *x_max_px* to exclude the display overlay region (e.g. x_max_px=950).
    Returns the sub-pixel centroid ``(x_px, y_px)`` of the brightest
    feature found, or ``None`` if no bright spot is detected.
    """
    gray = polarity_adjusted(preprocess_gray(frame, config, background), config.particle_polarity)
    h_img, w_img = gray.shape[:2]
    y1 = max(0, int(zero_line_y_px) - search_band_px)
    y2 = min(h_img, int(zero_line_y_px) + search_band_px + 1)
    if x_hint_px is not None:
        x1 = max(0, int(x_hint_px) - x_search_half_width_px)
        x2 = min(w_img, int(x_hint_px) + x_search_half_width_px + 1)
    else:
        x1, x2 = 0, w_img
    if x_max_px is not None:
        x2 = min(x2, int(x_max_px))
    band = gray[y1:y2, x1:x2]
    if band.size == 0:
        return None
    blurred = cv2.GaussianBlur(band, (5, 5), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    if float(max_val) < 10.0:
        return None
    bx, by = max_loc
    # Sub-pixel centroid in a small patch around the maximum.
    wx = min(10, band.shape[1] // 2)
    wy = min(10, band.shape[0] // 2)
    cx1 = max(0, bx - wx)
    cx2 = min(band.shape[1], bx + wx + 1)
    cy1 = max(0, by - wy)
    cy2 = min(band.shape[0], by + wy + 1)
    patch = band[cy1:cy2, cx1:cx2].astype(np.float64)
    total = float(patch.sum())
    if total < 1.0:
        return float(x1 + bx), float(y1 + by)
    ys_local, xs_local = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    sub_x = float(np.sum(xs_local * patch) / total) + cx1
    sub_y = float(np.sum(ys_local * patch) / total) + cy1
    return float(x1 + sub_x), float(y1 + sub_y)


def detect_seed_from_timer(
    video: VideoSource,
    config: "TrackingConfig",
    timer_roi: tuple[int, int, int, int],
    zero_line_y_px: float,
    scan_frames: int = 300,
    search_band_px: int = 30,
    x_hint_px: float | None = None,
    x_max_px: int | None = None,
) -> tuple[int, tuple[float, float] | None]:
    """Detect the tracking seed from the in-frame timer and zero graduation line.

    1. Scan the first *scan_frames* frames to find where the timer ROI starts
       changing — this is the frame at which the target drop crosses the zero
       graduation line (``onset_frame``).
    2. At *onset_frame*, locate the brightest feature within *search_band_px*
       pixels of *zero_line_y_px*; that is the target drop.

    If no timer onset is detected (onset == -1, e.g. the Brownian video where
    the timer is already frozen), a fallback is attempted: if *x_hint_px* is
    provided the full-height frame 0 is searched near that x-coordinate.

    Returns ``(onset_frame, (x_px, y_px))``.  If the drop cannot be located,
    the second element is ``None`` and a warning is emitted.
    """
    onset_frame = detect_timer_start(video, timer_roi, scan_frames=scan_frames)

    if onset_frame < 0:
        # Timer never started in this video (e.g. Brownian video already frozen).
        # Fall back: search near x_hint at any height in frame 0.
        if x_hint_px is not None:
            frame0 = video.read_frame(0)
            h_fallback = video.info.height_px
            drop_pos = find_drop_at_zero_line(
                frame0, h_fallback / 2, config, None,
                search_band_px=h_fallback // 2,
                x_hint_px=x_hint_px,
                x_search_half_width_px=80,
                x_max_px=x_max_px,
            )
            if drop_pos is not None:
                print(
                    f"  Timer-based seed (fallback, no onset): frame 0, "
                    f"drop at ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}) px"
                )
                return 0, drop_pos
        warnings.warn(
            "Timer onset not found and no x_hint available; "
            "falling back to manual/GUI drop selection."
        )
        return 0, None

    frame = video.read_frame(onset_frame)
    background: Any = None
    if config.bg_frames > 0:
        bg_start = max(0, onset_frame - config.bg_frames)
        background = compute_video_background(video, config.bg_frames, bg_start)
    drop_pos = find_drop_at_zero_line(
        frame, zero_line_y_px, config, background,
        search_band_px=search_band_px,
        x_hint_px=x_hint_px,
        x_max_px=x_max_px,
    )
    if drop_pos is None:
        warnings.warn(
            f"Timer onset at frame {onset_frame} (t={onset_frame / video.info.fps:.2f}s) but "
            f"no drop found within ±{search_band_px}px of zero line (y={zero_line_y_px:.0f}px). "
            "Check --zero-line-y-px and --timer-roi."
        )
    else:
        print(
            f"  Timer-based seed: onset frame {onset_frame} "
            f"(t={onset_frame / video.info.fps:.2f}s), "
            f"drop at ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}) px"
        )
    return onset_frame, drop_pos


def polarity_adjusted(gray_frame: Any, particle_polarity: str) -> Any:
    if particle_polarity == "dark":
        return cv2.bitwise_not(gray_frame)
    return gray_frame


def clamp_roi_from_center(
    center_x_px: float,
    center_y_px: float,
    roi_size_px: int,
    width_px: int,
    height_px: int,
) -> tuple[int, int, int, int]:
    half_size = max(2, int(round(roi_size_px / 2)))
    left_px = max(0, int(round(center_x_px)) - half_size)
    top_px = max(0, int(round(center_y_px)) - half_size)
    right_px = min(width_px, int(round(center_x_px)) + half_size)
    bottom_px = min(height_px, int(round(center_y_px)) + half_size)
    width = max(1, right_px - left_px)
    height = max(1, bottom_px - top_px)
    return left_px, top_px, width, height


def clamp_bbox(bbox: tuple[float, float, float, float], width_px: int, height_px: int) -> tuple[int, int, int, int]:
    left_px = max(0, int(round(bbox[0])))
    top_px = max(0, int(round(bbox[1])))
    right_px = min(width_px, int(round(bbox[0] + bbox[2])))
    bottom_px = min(height_px, int(round(bbox[1] + bbox[3])))
    return left_px, top_px, max(1, right_px - left_px), max(1, bottom_px - top_px)


def refine_centroid(frame: Any, bbox: tuple[float, float, float, float], config: TrackingConfig) -> tuple[float, float]:
    gray_frame = preprocess_gray(frame, config)
    height_px, width_px = gray_frame.shape[:2]
    left_px, top_px, roi_width_px, roi_height_px = clamp_bbox(bbox, width_px, height_px)
    roi = gray_frame[top_px : top_px + roi_height_px, left_px : left_px + roi_width_px]
    if roi.size == 0:
        return float(bbox[0] + bbox[2] / 2), float(bbox[1] + bbox[3] / 2)

    adjusted_roi = polarity_adjusted(roi, config.particle_polarity)
    adjusted_roi = adjusted_roi.astype("float64")
    adjusted_roi -= float(np.min(adjusted_roi))
    if float(np.max(adjusted_roi)) <= 0:
        return float(left_px + roi_width_px / 2), float(top_px + roi_height_px / 2)

    threshold_value = float(np.percentile(adjusted_roi, 70.0))
    weights = np.where(adjusted_roi >= threshold_value, adjusted_roi, 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        weights = adjusted_roi
        total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float(left_px + roi_width_px / 2), float(top_px + roi_height_px / 2)

    row_indices, column_indices = np.indices(weights.shape)
    centroid_x = left_px + float(np.sum(column_indices * weights) / total_weight)
    centroid_y = top_px + float(np.sum(row_indices * weights) / total_weight)
    return centroid_x, centroid_y


def interactive_click_points(
    frame: Any,
    *,
    window_name: str,
    prompt: str,
    min_points: int,
    max_points: int | None,
) -> list[tuple[float, float]]:
    selected_points: list[tuple[float, float]] = []

    def mouse_callback(event: int, x_coord: int, y_coord: int, flags: int, userdata: Any) -> None:
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            if max_points is None or len(selected_points) < max_points:
                selected_points.append((float(x_coord), float(y_coord)))
        elif event == cv2.EVENT_RBUTTONDOWN and selected_points:
            selected_points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    print(prompt)
    print("Left click: add point. Right click or 'u': undo. Enter: finish. Esc: cancel.")

    while True:
        display_frame = frame.copy()
        for point_index, (point_x, point_y) in enumerate(selected_points, start=1):
            cv2.circle(display_frame, (int(round(point_x)), int(round(point_y))), 6, (0, 255, 255), 2)
            cv2.putText(
                display_frame,
                str(point_index),
                (int(round(point_x)) + 8, int(round(point_y)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            display_frame,
            f"{len(selected_points)} selected; Enter to finish",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display_frame)
        key_code = cv2.waitKey(30) & 0xFF
        if key_code in (13, 10):
            if len(selected_points) >= min_points:
                break
            print(f"Select at least {min_points} point(s).")
        elif key_code == 27:
            cv2.destroyWindow(window_name)
            raise AnalysisError("Interactive selection cancelled.")
        elif key_code == ord("u") and selected_points:
            selected_points.pop()
    cv2.destroyWindow(window_name)
    return selected_points


def interactive_scale_calibration(frame: Any, grid_spacing_um: float, grid_intervals: float) -> float:
    points = interactive_click_points(
        frame,
        window_name="Scale calibration",
        prompt="Click two grid marks separated by the configured grid interval count.",
        min_points=2,
        max_points=2,
    )
    first_point, second_point = points
    pixel_distance = math.hypot(second_point[0] - first_point[0], second_point[1] - first_point[1])
    if pixel_distance <= 0:
        raise AnalysisError("Calibration points have zero pixel distance.")
    return float(grid_spacing_um * grid_intervals / pixel_distance)


def choose_tracking_method(requested_tracker: str) -> str:
    if requested_tracker != "auto":
        return requested_tracker
    if trackpy_module is not None:
        return "trackpy"
    return "csrt"


def create_opencv_tracker(method: str) -> Any:
    method_key = method.lower()
    method_map = {
        "csrt": "CSRT",
        "kcf": "KCF",
        "mil": "MIL",
        "mosse": "MOSSE",
    }
    if method_key not in method_map:
        raise AnalysisError(f"Unknown OpenCV tracker: {method}")
    factory_suffix = method_map[method_key]
    factory_names = [f"Tracker{factory_suffix}_create"]
    legacy_module = getattr(cv2, "legacy", None)
    for factory_name in factory_names:
        factory = getattr(cv2, factory_name, None)
        if factory is not None:
            return factory()
        if legacy_module is not None:
            legacy_factory = getattr(legacy_module, factory_name, None)
            if legacy_factory is not None:
                return legacy_factory()
    raise AnalysisError(f"OpenCV tracker {method} is not available in this cv2 build.")


def build_trajectory_dataframe(rows: list[dict[str, Any]], um_per_px: float, source_video: str) -> Any:
    if not rows:
        return pd.DataFrame(columns=["drop_id", "frame", "time_s", "x_px", "y_px", "x_um", "y_um", "source_video"])
    trajectory_df = pd.DataFrame(rows)
    trajectory_df["x_um"] = trajectory_df["x_px"] * um_per_px
    trajectory_df["y_um"] = trajectory_df["y_px"] * um_per_px
    trajectory_df["source_video"] = source_video
    return trajectory_df.sort_values(["drop_id", "frame"]).reset_index(drop=True)


def track_with_opencv(
    video: VideoSource,
    points: list[tuple[float, float]],
    drop_ids: list[str],
    um_per_px: float,
    config: TrackingConfig,
    source_video: str,
    start_frame: int,
    method: str,
) -> Any:
    first_frame = video.read_frame(start_frame)
    frame_height_px, frame_width_px = first_frame.shape[:2]
    trackers: list[dict[str, Any]] = []
    for drop_id, (point_x, point_y) in zip(drop_ids, points):
        bbox = clamp_roi_from_center(point_x, point_y, config.roi_size_px, frame_width_px, frame_height_px)
        tracker = create_opencv_tracker(method)
        tracker.init(first_frame, tuple(float(value) for value in bbox))
        trackers.append({"drop_id": drop_id, "tracker": tracker, "bbox": bbox, "active": True})

    rows: list[dict[str, Any]] = []
    for frame_index, frame in video.iter_frames(start_frame=start_frame, max_frames=config.max_frames):
        for tracker_state in trackers:
            if not tracker_state["active"]:
                continue
            if frame_index == start_frame:
                bbox = tracker_state["bbox"]
                success = True
            else:
                success, bbox = tracker_state["tracker"].update(frame)
                tracker_state["bbox"] = bbox
            if not success:
                tracker_state["active"] = False
                warnings.warn(f"Tracker lost drop {tracker_state['drop_id']} at frame {frame_index}.")
                continue
            centroid_x, centroid_y = refine_centroid(frame, bbox, config)
            rows.append(
                {
                    "drop_id": tracker_state["drop_id"],
                    "frame": int(frame_index),
                    "time_s": float(frame_index / video.info.fps),
                    "x_px": float(centroid_x),
                    "y_px": float(centroid_y),
                }
            )
    return build_trajectory_dataframe(rows, um_per_px, source_video)


def track_with_template(
    video: VideoSource,
    points: list[tuple[float, float]],
    drop_ids: list[str],
    um_per_px: float,
    config: TrackingConfig,
    source_video: str,
    start_frame: int,
) -> Any:
    background: Any = None
    if config.bg_frames > 0:
        print(f"Computing background for {source_video} ({config.bg_frames} frames)...")
        background = compute_video_background(video, config.bg_frames, start_frame)
    first_frame = video.read_frame(start_frame)
    first_gray = polarity_adjusted(preprocess_gray(first_frame, config, background), config.particle_polarity)
    frame_height_px, frame_width_px = first_gray.shape[:2]
    template_states: list[dict[str, Any]] = []
    for drop_id, (point_x, point_y) in zip(drop_ids, points):
        bbox = clamp_roi_from_center(point_x, point_y, config.roi_size_px, frame_width_px, frame_height_px)
        left_px, top_px, roi_width_px, roi_height_px = bbox
        template = first_gray[top_px : top_px + roi_height_px, left_px : left_px + roi_width_px].copy()
        template_states.append(
            {
                "drop_id": drop_id,
                "template": template,
                "center_x_px": float(point_x),
                "center_y_px": float(point_y),
                "bbox": bbox,
            }
        )

    rows: list[dict[str, Any]] = []
    for frame_index, frame in video.iter_frames(start_frame=start_frame, max_frames=config.max_frames):
        adjusted_gray = polarity_adjusted(preprocess_gray(frame, config, background), config.particle_polarity)
        frame_height_px, frame_width_px = adjusted_gray.shape[:2]
        for state in template_states:
            template = state["template"]
            template_height_px, template_width_px = template.shape[:2]
            if frame_index == start_frame:
                bbox = state["bbox"]
            else:
                search_half_width = int(round(config.search_range_px + template_width_px / 2))
                search_half_height = int(round(config.search_range_px + template_height_px / 2))
                left_px = max(0, int(round(state["center_x_px"])) - search_half_width)
                top_px = max(0, int(round(state["center_y_px"])) - search_half_height)
                right_px = min(frame_width_px, int(round(state["center_x_px"])) + search_half_width)
                bottom_px = min(frame_height_px, int(round(state["center_y_px"])) + search_half_height)
                search_roi = adjusted_gray[top_px:bottom_px, left_px:right_px]
                if search_roi.shape[0] < template_height_px or search_roi.shape[1] < template_width_px:
                    continue
                response = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_location = cv2.minMaxLoc(response)
                bbox = (left_px + max_location[0], top_px + max_location[1], template_width_px, template_height_px)
            centroid_x, centroid_y = refine_centroid(frame, bbox, config)
            state["center_x_px"] = centroid_x
            state["center_y_px"] = centroid_y
            state["bbox"] = bbox
            rows.append(
                {
                    "drop_id": state["drop_id"],
                    "frame": int(frame_index),
                    "time_s": float(frame_index / video.info.fps),
                    "x_px": float(centroid_x),
                    "y_px": float(centroid_y),
                }
            )
    return build_trajectory_dataframe(rows, um_per_px, source_video)


def track_with_trackpy(
    video: VideoSource,
    points: list[tuple[float, float]],
    drop_ids: list[str],
    um_per_px: float,
    config: TrackingConfig,
    source_video: str,
    start_frame: int,
) -> Any:
    if trackpy_module is None:
        raise AnalysisError("trackpy is not installed.")
    diameter_px = validate_odd_positive("--trackpy-diameter-px", int(config.trackpy_diameter_px))
    background: Any = None
    if config.bg_frames > 0:
        print(f"Computing background for {source_video} ({config.bg_frames} frames)...")
        background = compute_video_background(video, config.bg_frames, start_frame)
    feature_frames: list[Any] = []

    for frame_index, frame in video.iter_frames(start_frame=start_frame, max_frames=config.max_frames):
        gray_frame = preprocess_gray(frame, config, background)
        locate_image = polarity_adjusted(gray_frame, config.particle_polarity)
        locate_kwargs: dict[str, Any] = {"diameter": diameter_px}
        if config.trackpy_minmass is not None:
            locate_kwargs["minmass"] = config.trackpy_minmass
        features = trackpy_module.locate(locate_image, **locate_kwargs)
        if features is None or len(features) == 0:
            continue
        features = features.copy()
        features["frame"] = int(frame_index)
        feature_frames.append(features)

    if not feature_frames:
        raise AnalysisError("trackpy did not locate any particles.")

    feature_df = pd.concat(feature_frames, ignore_index=True)
    linked_df = trackpy_module.link_df(
        feature_df,
        search_range=float(config.search_range_px),
        memory=int(config.trackpy_memory_frames),
    )
    if "particle" not in linked_df.columns:
        raise AnalysisError("trackpy linking did not produce particle ids.")

    init_window_end = start_frame + max(5, int(config.trackpy_memory_frames) + 2)
    initial_candidates = linked_df[linked_df["frame"] <= init_window_end].copy()
    if initial_candidates.empty:
        first_linked_frame = int(linked_df["frame"].min())
        initial_candidates = linked_df[linked_df["frame"] == first_linked_frame].copy()

    selected_particles: dict[str, int] = {}
    used_particles: set[int] = set()
    for drop_id, (point_x, point_y) in zip(drop_ids, points):
        candidate_df = initial_candidates[~initial_candidates["particle"].isin(used_particles)].copy()
        if candidate_df.empty:
            warnings.warn(f"No unused trackpy candidate remains for drop {drop_id}.")
            continue
        candidate_df["distance_px"] = np.hypot(candidate_df["x"] - point_x, candidate_df["y"] - point_y)
        nearest_row = candidate_df.sort_values("distance_px").iloc[0]
        if float(nearest_row["distance_px"]) > config.selection_radius_px:
            warnings.warn(
                f"Nearest trackpy particle for drop {drop_id} is "
                f"{nearest_row['distance_px']:.1f} px from the clicked point."
            )
        particle_id = int(nearest_row["particle"])
        selected_particles[drop_id] = particle_id
        used_particles.add(particle_id)

    rows: list[dict[str, Any]] = []
    for drop_id, particle_id in selected_particles.items():
        particle_df = linked_df[linked_df["particle"] == particle_id].copy()
        for _, trajectory_row in particle_df.iterrows():
            frame_index = int(trajectory_row["frame"])
            rows.append(
                {
                    "drop_id": drop_id,
                    "frame": frame_index,
                    "time_s": float(frame_index / video.info.fps),
                    "x_px": float(trajectory_row["x"]),
                    "y_px": float(trajectory_row["y"]),
                }
            )
    return build_trajectory_dataframe(rows, um_per_px, source_video)


def track_video(
    video: VideoSource,
    points: list[tuple[float, float]],
    drop_ids: list[str],
    um_per_px: float,
    config: TrackingConfig,
    source_video: str,
    start_frame: int,
) -> Any:
    if not points:
        raise AnalysisError(f"No points were selected for {source_video} tracking.")
    if len(points) != len(drop_ids):
        raise AnalysisError("Point count and drop id count do not match.")

    method = choose_tracking_method(config.tracker)
    if method == "trackpy":
        try:
            print(f"Tracking {source_video} with trackpy...")
            return track_with_trackpy(video, points, drop_ids, um_per_px, config, source_video, start_frame)
        except Exception as exc:
            if config.tracker == "trackpy":
                raise
            warnings.warn(f"trackpy tracking failed ({exc}); falling back to OpenCV CSRT/template tracking.")
            method = "csrt"

    if method == "template":
        print(f"Tracking {source_video} with template matching...")
        return track_with_template(video, points, drop_ids, um_per_px, config, source_video, start_frame)

    try:
        print(f"Tracking {source_video} with OpenCV {method.upper()}...")
        return track_with_opencv(video, points, drop_ids, um_per_px, config, source_video, start_frame, method)
    except Exception as exc:
        if method == "template":
            raise
        warnings.warn(f"OpenCV {method.upper()} tracking failed ({exc}); falling back to template matching.")
        return track_with_template(video, points, drop_ids, um_per_px, config, source_video, start_frame)


def air_viscosity_pa_s(params: PhysicsParams) -> float:
    validate_positive("eta0-pa-s", params.eta0_pa_s)
    if params.viscosity_model == "constant":
        return float(params.eta0_pa_s)
    validate_positive("eta0-reference-k", params.eta0_reference_k)
    validate_positive("sutherland-c-k", params.sutherland_c_k)
    return float(
        params.eta0_pa_s
        * (params.temperature_k / params.eta0_reference_k) ** 1.5
        * (params.eta0_reference_k + params.sutherland_c_k)
        / (params.temperature_k + params.sutherland_c_k)
    )


def cunningham_factor(radius_m: float, params: PhysicsParams) -> float:
    return float(1.0 + params.cunningham_b_pa_m / (params.pressure_pa * radius_m))


def radius_from_terminal_velocity(terminal_velocity_m_s: float, params: PhysicsParams) -> tuple[float, float, float]:
    validate_positive("terminal velocity", terminal_velocity_m_s)
    validate_positive("pressure-pa", params.pressure_pa)
    validate_positive("oil-density-kgm3", params.oil_density_kgm3)
    if params.oil_density_kgm3 <= params.air_density_kgm3:
        raise AnalysisError("Oil density must be greater than air density.")
    eta_air = air_viscosity_pa_s(params)
    density_difference = params.oil_density_kgm3 - params.air_density_kgm3
    slip_length_m = params.cunningham_b_pa_m / (2.0 * params.pressure_pa)
    radius_m = math.sqrt(
        slip_length_m * slip_length_m
        + 9.0 * eta_air * terminal_velocity_m_s / (2.0 * density_difference * STANDARD_GRAVITY_M_S2)
    ) - slip_length_m
    if radius_m <= 0 or not math.isfinite(radius_m):
        raise AnalysisError("Computed non-positive droplet radius.")
    correction_factor = cunningham_factor(radius_m, params)
    eta_slip = eta_air / correction_factor
    return float(radius_m), float(correction_factor), float(eta_slip)


def fit_line(x_values: Any, y_values: Any) -> dict[str, float]:
    x_array = np.asarray(x_values, dtype="float64")
    y_array = np.asarray(y_values, dtype="float64")
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite_mask]
    y_array = y_array[finite_mask]
    if len(x_array) < 2:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "stderr": math.nan}
    result = scipy_stats.linregress(x_array, y_array)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r2": float(result.rvalue * result.rvalue),
        "stderr": float(result.stderr) if result.stderr is not None else math.nan,
    }


def robust_line_fit(x_values: Any, y_values: Any) -> dict[str, float]:
    first_fit = fit_line(x_values, y_values)
    if not math.isfinite(first_fit["slope"]):
        return first_fit
    x_array = np.asarray(x_values, dtype="float64")
    y_array = np.asarray(y_values, dtype="float64")
    predicted = first_fit["slope"] * x_array + first_fit["intercept"]
    residuals = y_array - predicted
    median_residual = float(np.nanmedian(residuals))
    mad = float(np.nanmedian(np.abs(residuals - median_residual)))
    if mad <= 0 or not math.isfinite(mad):
        return first_fit
    inlier_mask = np.abs(residuals - median_residual) <= 4.0 * 1.4826 * mad
    if int(np.sum(inlier_mask)) < 3:
        return first_fit
    return fit_line(x_array[inlier_mask], y_array[inlier_mask])


def detect_freefall_onset(times: Any, y_um: Any, smooth_window: int = 7) -> int:
    """Return the row index where free fall begins.

    Estimates terminal velocity from the latter half of the trajectory, then
    finds the first frame at which the smoothed vertical velocity exceeds 20 %
    of that estimate.  Returns 0 when onset cannot be determined (all frames
    assumed to be in free fall).
    """
    t = np.asarray(times, dtype="float64")
    y = np.asarray(y_um, dtype="float64")
    n = len(t)
    if n < smooth_window * 2:
        return 0
    v_raw = np.gradient(y, t)  # um/s, instantaneous
    kernel = np.ones(smooth_window) / smooth_window
    v_smooth = np.convolve(v_raw, kernel, mode="same")
    # Estimate terminal velocity from the latter 50 % of frames.
    v_terminal = float(np.median(v_smooth[n // 2:]))
    if abs(v_terminal) < 1.0:  # < 1 um/s — stationary; onset not detectable
        return 0
    threshold = 0.20 * abs(v_terminal)
    sign = 1.0 if v_terminal > 0.0 else -1.0
    candidates = np.where(sign * v_smooth > threshold)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def analyze_freefall(
    freefall_df: Any,
    params: PhysicsParams,
    fall_time_override_s: float | None = None,
    graduation_distance_um: float = 320.0,
) -> tuple[Any, dict[str, RadiusResult]]:
    """Estimate radii from free-fall trajectories.

    When *fall_time_override_s* is provided, the terminal velocity is derived
    directly from the measured graduation-crossing time instead of a noisy
    linear regression on the full trajectory:

        v_g = graduation_distance_um / fall_time_override_s
    """
    eta_air = air_viscosity_pa_s(params)
    radius_results: dict[str, RadiusResult] = {}
    rows: list[dict[str, Any]] = []
    for drop_id, drop_df in freefall_df.groupby("drop_id"):
        drop_df = drop_df.dropna(subset=["time_s", "y_um"]).sort_values("time_s")
        onset_idx = detect_freefall_onset(drop_df["time_s"].to_numpy(), drop_df["y_um"].to_numpy())
        if onset_idx > 0:
            onset_time = float(drop_df["time_s"].iloc[onset_idx])
            warnings.warn(
                f"Drop {drop_id}: free-fall onset detected at frame index {onset_idx} "
                f"(t ≈ {onset_time:.3f} s); {onset_idx} leading stationary frame(s) trimmed."
            )
            drop_df = drop_df.iloc[onset_idx:].copy()
        if len(drop_df) < 2:
            warnings.warn(f"Not enough free-fall points for drop {drop_id}.")
            continue
        if fall_time_override_s is not None and fall_time_override_s > 0:
            validate_positive("fall-time-s", fall_time_override_s)
            vertical_speed_um_s = graduation_distance_um / fall_time_override_s
            fit = {"slope": vertical_speed_um_s, "r2": float("nan")}
        else:
            fit = robust_line_fit(drop_df["time_s"].to_numpy(), drop_df["y_um"].to_numpy())
            vertical_speed_um_s = abs(float(fit["slope"]))
        terminal_velocity_m_s = vertical_speed_um_s * 1e-6
        try:
            radius_m, correction_factor, eta_slip = radius_from_terminal_velocity(terminal_velocity_m_s, params)
        except AnalysisError as exc:
            warnings.warn(f"Radius calculation failed for drop {drop_id}: {exc}")
            continue
        fall_time_s = float(drop_df["time_s"].max() - drop_df["time_s"].min())
        result = RadiusResult(
            drop_id=str(drop_id),
            fall_time_s=fall_time_s,
            frame_count=int(len(drop_df)),
            vertical_speed_um_s=vertical_speed_um_s,
            terminal_velocity_m_s=terminal_velocity_m_s,
            radius_m=radius_m,
            radius_um=radius_m * 1e6,
            cunningham_factor=correction_factor,
            eta_air_pa_s=eta_air,
            eta_slip_corrected_pa_s=eta_slip,
            fit_r2=float(fit["r2"]),
        )
        radius_results[str(drop_id)] = result
        rows.append(asdict(result))
    return pd.DataFrame(rows), radius_results


def dedrift_brownian(brownian_df: Any) -> tuple[Any, Any]:
    rows: list[Any] = []
    drift_rows: list[dict[str, Any]] = []
    for drop_id, drop_df in brownian_df.groupby("drop_id"):
        drop_df = drop_df.dropna(subset=["time_s", "x_um"]).sort_values("time_s").copy()
        if len(drop_df) < 2:
            warnings.warn(f"Not enough Brownian points to de-drift drop {drop_id}.")
            continue
        fit = fit_line(drop_df["time_s"].to_numpy(), drop_df["x_um"].to_numpy())
        first_time_s = float(drop_df["time_s"].iloc[0])
        drop_df["x_drift_um"] = fit["slope"] * (drop_df["time_s"] - first_time_s)
        drop_df["x_dedrift_um"] = drop_df["x_um"] - drop_df["x_drift_um"]
        rows.append(drop_df)
        drift_rows.append(
            {
                "drop_id": str(drop_id),
                "drift_velocity_um_s": float(fit["slope"]),
                "drift_fit_r2": float(fit["r2"]),
            }
        )
    if not rows:
        raise AnalysisError("No Brownian trajectories survived de-drifting.")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(drift_rows)


def displacement_for_lag(drop_df: Any, lag_frames: int) -> Any:
    series = drop_df.dropna(subset=["frame", "x_dedrift_um"]).set_index("frame")["x_dedrift_um"].sort_index()
    if series.empty:
        return np.asarray([], dtype="float64")
    frames = series.index.to_numpy(dtype="int64")
    target_frames = frames + int(lag_frames)
    valid_mask = np.isin(target_frames, series.index.to_numpy(dtype="int64"))
    if not np.any(valid_mask):
        return np.asarray([], dtype="float64")
    start_frames = frames[valid_mask]
    end_frames = target_frames[valid_mask]
    return series.loc[end_frames].to_numpy(dtype="float64") - series.loc[start_frames].to_numpy(dtype="float64")


def compute_brownian_statistics(
    dedrift_df: Any,
    lag_times_s: list[float],
    fps: float,
) -> tuple[Any, dict[tuple[str, float], Any]]:
    rows: list[dict[str, Any]] = []
    displacement_map: dict[tuple[str, float], Any] = {}
    overall_by_lag: dict[float, list[Any]] = defaultdict(list)

    for lag_time_s in lag_times_s:
        lag_frames = max(1, int(round(lag_time_s * fps)))
        actual_lag_s = float(lag_frames / fps)
        for drop_id, drop_df in dedrift_df.groupby("drop_id"):
            displacements_um = displacement_for_lag(drop_df, lag_frames)
            displacement_map[(str(drop_id), actual_lag_s)] = displacements_um
            if len(displacements_um) > 0:
                overall_by_lag[actual_lag_s].append(displacements_um)
            mean_dx_um = float(np.mean(displacements_um)) if len(displacements_um) else math.nan
            msd_um2 = float(np.mean(displacements_um * displacements_um)) if len(displacements_um) else math.nan
            rows.append(
                {
                    "drop_id": str(drop_id),
                    "lag_s": actual_lag_s,
                    "lag_frames": int(lag_frames),
                    "sample_count": int(len(displacements_um)),
                    "mean_dx_um": mean_dx_um,
                    "msd_um2": msd_um2,
                    "mean_dx_m": mean_dx_um * 1e-6 if math.isfinite(mean_dx_um) else math.nan,
                    "msd_m2": msd_um2 * 1e-12 if math.isfinite(msd_um2) else math.nan,
                }
            )

    for lag_time_s, displacement_arrays in sorted(overall_by_lag.items()):
        combined_displacements_um = np.concatenate(displacement_arrays) if displacement_arrays else np.asarray([], dtype="float64")
        displacement_map[("all", lag_time_s)] = combined_displacements_um
        mean_dx_um = float(np.mean(combined_displacements_um)) if len(combined_displacements_um) else math.nan
        msd_um2 = float(np.mean(combined_displacements_um * combined_displacements_um)) if len(combined_displacements_um) else math.nan
        rows.append(
            {
                "drop_id": "all",
                "lag_s": float(lag_time_s),
                "lag_frames": int(round(lag_time_s * fps)),
                "sample_count": int(len(combined_displacements_um)),
                "mean_dx_um": mean_dx_um,
                "msd_um2": msd_um2,
                "mean_dx_m": mean_dx_um * 1e-6 if math.isfinite(mean_dx_um) else math.nan,
                "msd_m2": msd_um2 * 1e-12 if math.isfinite(msd_um2) else math.nan,
            }
        )
    stats_df = pd.DataFrame(rows).drop_duplicates(subset=["drop_id", "lag_s"], keep="last")
    return stats_df.sort_values(["drop_id", "lag_s"]).reset_index(drop=True), displacement_map


def fit_msd_for_drop(stats_df: Any, drop_id: str, fit_min_s: float | None, fit_max_s: float | None) -> dict[str, float | str]:
    fit_df = stats_df[(stats_df["drop_id"] == drop_id) & (stats_df["sample_count"] > 0)].copy()
    fit_df = fit_df[np.isfinite(fit_df["msd_m2"])]
    if fit_min_s is not None:
        fit_df = fit_df[fit_df["lag_s"] >= fit_min_s]
    if fit_max_s is not None:
        fit_df = fit_df[fit_df["lag_s"] <= fit_max_s]
    if len(fit_df) < 2:
        return {
            "drop_id": drop_id,
            "msd_slope_m2_s": math.nan,
            "msd_intercept_m2": math.nan,
            "diffusion_m2_s": math.nan,
            "diffusion_stderr_m2_s": math.nan,
            "msd_fit_r2": math.nan,
            "fit_point_count": int(len(fit_df)),
        }
    fit = fit_line(fit_df["lag_s"].to_numpy(), fit_df["msd_m2"].to_numpy())
    diffusion_m2_s = float(fit["slope"] / 2.0)
    diffusion_stderr = float(fit["stderr"] / 2.0) if math.isfinite(fit["stderr"]) else math.nan
    return {
        "drop_id": drop_id,
        "msd_slope_m2_s": float(fit["slope"]),
        "msd_intercept_m2": float(fit["intercept"]),
        "diffusion_m2_s": diffusion_m2_s,
        "diffusion_stderr_m2_s": diffusion_stderr,
        "msd_fit_r2": float(fit["r2"]),
        "fit_point_count": int(len(fit_df)),
    }


def fit_all_msd(stats_df: Any, fit_min_s: float | None, fit_max_s: float | None) -> Any:
    drop_ids = sorted(str(drop_id) for drop_id in stats_df["drop_id"].dropna().unique())
    rows = [fit_msd_for_drop(stats_df, drop_id, fit_min_s, fit_max_s) for drop_id in drop_ids]
    return pd.DataFrame(rows)


def choose_radius_for_drop(drop_id: str, radius_results: dict[str, RadiusResult]) -> RadiusResult | None:
    if drop_id in radius_results:
        return radius_results[drop_id]
    if len(radius_results) == 1:
        return next(iter(radius_results.values()))
    return None


def build_basic_parameter_table(
    brownian_drop_ids: list[str],
    radius_results: dict[str, RadiusResult],
    msd_fit_df: Any,
    drift_df: Any,
    params: PhysicsParams,
) -> Any:
    rows: list[dict[str, Any]] = []
    drift_by_drop = {str(row["drop_id"]): row for _, row in drift_df.iterrows()}
    fit_by_drop = {str(row["drop_id"]): row for _, row in msd_fit_df.iterrows()}
    ensemble_fit = fit_by_drop.get("all")
    for drop_id in brownian_drop_ids:
        radius_result = choose_radius_for_drop(drop_id, radius_results)
        fit_row = fit_by_drop.get(drop_id, ensemble_fit)
        drift_row = drift_by_drop.get(drop_id)
        if radius_result is None or fit_row is None:
            warnings.warn(f"Skipping k_B calculation for drop {drop_id}; missing radius or diffusion fit.")
            continue
        diffusion_m2_s = float(fit_row["diffusion_m2_s"])
        if not math.isfinite(diffusion_m2_s) or diffusion_m2_s <= 0:
            boltzmann_j_k = math.nan
        else:
            boltzmann_j_k = (
                6.0
                * math.pi
                * radius_result.eta_slip_corrected_pa_s
                * radius_result.radius_m
                * diffusion_m2_s
                / params.temperature_k
            )
        rows.append(
            {
                "drop_id": drop_id,
                "radius_um": radius_result.radius_um,
                "diffusion_m2_s": diffusion_m2_s,
                "k_B_J_K": boltzmann_j_k,
                "k_B_ratio_to_SI": boltzmann_j_k / TRUE_BOLTZMANN_J_K if math.isfinite(boltzmann_j_k) else math.nan,
                "cunningham_factor": radius_result.cunningham_factor,
                "eta_slip_corrected_pa_s": radius_result.eta_slip_corrected_pa_s,
                "drift_velocity_um_s": float(drift_row["drift_velocity_um_s"]) if drift_row is not None else math.nan,
                "msd_fit_r2": float(fit_row["msd_fit_r2"]),
                "fit_point_count": int(fit_row["fit_point_count"]),
            }
        )
    return pd.DataFrame(rows)


def fit_gaussian_to_histogram(displacements_um: Any, bins: int = 30) -> tuple[Any, Any, Any]:
    if len(displacements_um) < 5:
        return None, None, None
    counts, bin_edges = np.histogram(displacements_um, bins=bins)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sigma_guess = float(np.std(displacements_um, ddof=1))
    if sigma_guess <= 0 or not math.isfinite(sigma_guess):
        return counts, centers, None

    def gaussian(center_values: Any, amplitude: float, mean_um: float, sigma_um: float) -> Any:
        return amplitude * np.exp(-0.5 * ((center_values - mean_um) / sigma_um) ** 2)

    initial_guess = [float(np.max(counts)), float(np.mean(displacements_um)), sigma_guess]
    try:
        fit_params, covariance = optimize.curve_fit(gaussian, centers, counts, p0=initial_guess, maxfev=10000)
        del covariance
        return counts, centers, fit_params
    except Exception:
        return counts, centers, None


def plot_trajectories(dedrift_df: Any, output_dir: Path, show_plots: bool) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for drop_id, drop_df in dedrift_df.groupby("drop_id"):
        axis.plot(drop_df["x_um"], drop_df["y_um"], marker=".", linewidth=1.0, markersize=2, label=f"Drop {drop_id}")
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_title("2D oil-drop trajectory")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_2d.png", dpi=200)
    if show_plots:
        plt.show()
    plt.close(figure)


def plot_displacement_histogram(
    displacement_map: dict[tuple[str, float], Any],
    hist_lag_s: float,
    output_dir: Path,
    show_plots: bool,
) -> None:
    available_lags = sorted(lag_time_s for drop_id, lag_time_s in displacement_map if drop_id == "all")
    if not available_lags:
        warnings.warn("No overall displacement data available for histogram.")
        return
    selected_lag_s = min(available_lags, key=lambda lag_time_s: abs(lag_time_s - hist_lag_s))
    displacements_um = displacement_map.get(("all", selected_lag_s), np.asarray([], dtype="float64"))
    if len(displacements_um) < 5:
        warnings.warn("Too few displacements for a histogram.")
        return
    counts, centers, fit_params = fit_gaussian_to_histogram(displacements_um)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(displacements_um, bins=30, alpha=0.65, color="tab:blue", edgecolor="white", label="Data")
    if fit_params is not None:
        center_grid = np.linspace(float(np.min(displacements_um)), float(np.max(displacements_um)), 300)
        amplitude, mean_um, sigma_um = fit_params
        gaussian_counts = amplitude * np.exp(-0.5 * ((center_grid - mean_um) / sigma_um) ** 2)
        axis.plot(center_grid, gaussian_counts, color="tab:red", linewidth=2, label=f"Gaussian fit, mean={mean_um:.3g} um")
    axis.set_xlabel("Horizontal displacement dx (um)")
    axis.set_ylabel("Count")
    axis.set_title(f"Displacement distribution at lag {selected_lag_s:.3g} s")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "displacement_histogram.png", dpi=200)
    if show_plots:
        plt.show()
    plt.close(figure)
    del counts, centers


def plot_msd_fit(stats_df: Any, msd_fit_df: Any, output_dir: Path, show_plots: bool) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    per_drop_df = stats_df[(stats_df["drop_id"] != "all") & (stats_df["sample_count"] > 0)]
    for drop_id, drop_df in per_drop_df.groupby("drop_id"):
        axis.plot(drop_df["lag_s"], drop_df["msd_um2"], marker="o", linestyle="--", alpha=0.45, label=f"Drop {drop_id}")
    overall_df = stats_df[(stats_df["drop_id"] == "all") & (stats_df["sample_count"] > 0)].copy()
    if not overall_df.empty:
        axis.plot(overall_df["lag_s"], overall_df["msd_um2"], marker="o", color="black", linewidth=2, label="All drops")
        fit_rows = msd_fit_df[msd_fit_df["drop_id"] == "all"]
        if not fit_rows.empty:
            fit_row = fit_rows.iloc[0]
            if math.isfinite(float(fit_row["msd_slope_m2_s"])):
                lag_grid = np.linspace(float(overall_df["lag_s"].min()), float(overall_df["lag_s"].max()), 200)
                fit_msd_m2 = float(fit_row["msd_slope_m2_s"]) * lag_grid + float(fit_row["msd_intercept_m2"])
                axis.plot(lag_grid, fit_msd_m2 * 1e12, color="tab:red", linewidth=2, label="Linear fit")
    axis.set_xlabel("Lag time (s)")
    axis.set_ylabel("Mean squared displacement (um^2)")
    axis.set_title("MSD linear fit")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "msd_fit.png", dpi=200)
    if show_plots:
        plt.show()
    plt.close(figure)


def warn_on_fit_quality(stats_df: Any, msd_fit_df: Any) -> None:
    overall_stats = stats_df[stats_df["drop_id"] == "all"].copy()
    if not overall_stats.empty:
        mean_abs_dx = float(np.nanmean(np.abs(overall_stats["mean_dx_um"])))
        rms_dx = float(math.sqrt(np.nanmean(overall_stats["msd_um2"]))) if np.any(np.isfinite(overall_stats["msd_um2"])) else math.nan
        if math.isfinite(mean_abs_dx) and math.isfinite(rms_dx) and rms_dx > 0 and mean_abs_dx > 0.25 * rms_dx:
            warnings.warn("The de-drifted mean displacement is not very close to zero; inspect drift and airflow.")
    for _, fit_row in msd_fit_df.iterrows():
        drop_id = str(fit_row["drop_id"])
        diffusion_m2_s = float(fit_row["diffusion_m2_s"])
        fit_r2 = float(fit_row["msd_fit_r2"])
        if math.isfinite(diffusion_m2_s) and diffusion_m2_s <= 0:
            warnings.warn(f"MSD fit for drop {drop_id} produced non-positive diffusion.")
        if math.isfinite(fit_r2) and fit_r2 < 0.8:
            warnings.warn(f"MSD fit R^2 for drop {drop_id} is low ({fit_r2:.3f}).")


def run_analysis(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = PhysicsParams(
        temperature_k=float(args.temperature_k),
        pressure_pa=float(args.pressure_pa),
        oil_density_kgm3=float(args.oil_density_kgm3),
        air_density_kgm3=float(args.air_density_kgm3),
        eta0_pa_s=float(args.eta0_pa_s),
        eta0_reference_k=float(args.eta0_reference_k),
        sutherland_c_k=float(args.sutherland_c_k),
        viscosity_model=str(args.viscosity_model),
        cunningham_b_pa_m=float(args.cunningham_b_pa_m),
    )
    for parameter_name, parameter_value in asdict(params).items():
        if parameter_name != "viscosity_model":
            validate_positive(parameter_name, float(parameter_value))
    if params.oil_density_kgm3 <= params.air_density_kgm3:
        raise AnalysisError("Oil density must be greater than air density.")

    freefall_video = VideoSource(Path(args.freefall_video), args.fps)
    brownian_video = VideoSource(Path(args.brownian_video), args.fps)
    try:
        brownian_frame = brownian_video.read_frame(int(args.brownian_selection_frame))
        freefall_frame = freefall_video.read_frame(int(args.freefall_selection_frame))

        if args.scale_um_per_px is not None:
            um_per_px = float(args.scale_um_per_px)
        else:
            if args.no_gui:
                raise AnalysisError("--scale-um-per-px is required when --no-gui is set.")
            validate_positive("grid-spacing-um", float(args.grid_spacing_um))
            validate_positive("grid-intervals", float(args.grid_intervals))
            um_per_px = interactive_scale_calibration(brownian_frame, float(args.grid_spacing_um), float(args.grid_intervals))
        validate_positive("scale-um-per-px", um_per_px)
        print(f"Scale factor: {um_per_px:.6g} um/px")

        config = TrackingConfig(
            tracker=str(args.tracker),
            roi_size_px=int(args.roi_size_px),
            search_range_px=int(args.search_range_px),
            trackpy_diameter_px=validate_odd_positive("--trackpy-diameter-px", int(args.trackpy_diameter_px)),
            trackpy_minmass=args.trackpy_minmass,
            trackpy_memory_frames=int(args.trackpy_memory_frames),
            selection_radius_px=float(args.selection_radius_px),
            particle_polarity=str(args.particle_polarity),
            clahe=not bool(args.no_clahe),
            blur_kernel_px=int(args.blur_kernel_px),
            max_frames=args.max_frames,
            bg_frames=int(args.bg_frames),
        )

        freefall_selection_frame = int(args.freefall_selection_frame)
        brownian_selection_frame = int(args.brownian_selection_frame)
        freefall_points = parse_point_list(args.freefall_points)
        brownian_points = parse_point_list(args.brownian_points)

        if args.use_timer_seed:
            timer_roi = parse_timer_roi(args.timer_roi)
            zero_y = float(args.zero_line_y_px)
            scan_n = int(args.timer_scan_frames)
            band_px = int(args.timer_search_band_px)
            x_max_drop = timer_roi[0]  # exclude display overlay (x >= display left edge)
            if not freefall_points:
                print("Detecting free-fall seed from timer...")
                onset_ff, pos_ff = detect_seed_from_timer(
                    freefall_video, config, timer_roi, zero_y,
                    scan_frames=scan_n, search_band_px=band_px,
                    x_max_px=x_max_drop,
                )
                if pos_ff is not None:
                    freefall_points = [pos_ff]
                    freefall_selection_frame = onset_ff
            if not brownian_points:
                print("Detecting Brownian seed from timer...")
                x_hint = freefall_points[0][0] if freefall_points else None
                onset_br, pos_br = detect_seed_from_timer(
                    brownian_video, config, timer_roi, zero_y,
                    scan_frames=scan_n, search_band_px=band_px,
                    x_hint_px=x_hint,
                    x_max_px=x_max_drop,
                )
                if pos_br is not None:
                    brownian_points = [pos_br]
                    brownian_selection_frame = onset_br

        if not brownian_points:
            if args.no_gui:
                raise AnalysisError("--brownian-points is required when --no-gui is set.")
            brownian_points = interactive_click_points(
                brownian_frame,
                window_name="Select Brownian drops",
                prompt="Click one or more balanced Brownian oil drops in Video B.",
                min_points=1,
                max_points=None,
            )
        brownian_drop_ids = parse_drop_ids(args.drop_ids, len(brownian_points))

        if not freefall_points:
            if args.no_gui:
                raise AnalysisError("--freefall-points is required when --no-gui is set.")
            freefall_points = interactive_click_points(
                freefall_frame,
                window_name="Select free-fall drops",
                prompt="Click the corresponding freely falling oil drop(s) in Video A.",
                min_points=1,
                max_points=None,
            )

        if len(freefall_points) == len(brownian_drop_ids):
            freefall_drop_ids = brownian_drop_ids
        elif len(freefall_points) == 1:
            freefall_drop_ids = [brownian_drop_ids[0]]
            warnings.warn("Only one free-fall drop selected; its radius will be reused if needed.")
        else:
            freefall_drop_ids = [f"freefall_{drop_number}" for drop_number in range(1, len(freefall_points) + 1)]
            warnings.warn("Free-fall and Brownian drop counts differ; unmatched radii will not be used per drop.")

        freefall_df = track_video(
            freefall_video,
            freefall_points,
            freefall_drop_ids,
            um_per_px,
            config,
            "freefall",
            freefall_selection_frame,
        )
        brownian_df = track_video(
            brownian_video,
            brownian_points,
            brownian_drop_ids,
            um_per_px,
            config,
            "brownian",
            brownian_selection_frame,
        )

        freefall_df.to_csv(output_dir / "freefall_trajectory.csv", index=False)
        brownian_df.to_csv(output_dir / "brownian_trajectory_raw.csv", index=False)

        fall_time_override = getattr(args, "fall_time_s", None)
        graduation_dist_um = float(getattr(args, "graduation_distance_um", 320.0))
        freefall_table, radius_results = analyze_freefall(
            freefall_df, params,
            fall_time_override_s=fall_time_override,
            graduation_distance_um=graduation_dist_um,
        )
        if freefall_table.empty:
            raise AnalysisError("No droplet radii could be calculated from Video A.")
        dedrift_df, drift_df = dedrift_brownian(brownian_df)
        dedrift_df.to_csv(output_dir / "brownian_trajectory_dedrifted.csv", index=False)
        drift_df.to_csv(output_dir / "drift_table.csv", index=False)

        lag_times_s = parse_lag_times(args.dt_list, float(args.max_lag_s), float(args.dt_step_s))
        stats_df, displacement_map = compute_brownian_statistics(dedrift_df, lag_times_s, brownian_video.info.fps)
        msd_fit_df = fit_all_msd(stats_df, args.fit_min_s, args.fit_max_s)
        basic_table = build_basic_parameter_table(brownian_drop_ids, radius_results, msd_fit_df, drift_df, params)
        warn_on_fit_quality(stats_df, msd_fit_df)


        save_table(freefall_table, output_dir, "freefall_radius_table")
        save_table(stats_df, output_dir, "statistics_table")
        save_table(msd_fit_df, output_dir, "msd_fit_table")
        save_table(basic_table, output_dir, "basic_parameter_table")

        plot_trajectories(dedrift_df, output_dir, bool(args.show_plots))
        plot_displacement_histogram(displacement_map, float(args.hist_lag_s), output_dir, bool(args.show_plots))
        plot_msd_fit(stats_df, msd_fit_df, output_dir, bool(args.show_plots))

        metadata = {
            "scale_um_per_px": um_per_px,
            "physics_params": asdict(params),
            "tracking_config": asdict(config),
            "freefall_video": asdict(freefall_video.info),
            "brownian_video": asdict(brownian_video.info),
            "brownian_points_px": brownian_points,
            "freefall_points_px": freefall_points,
        }
        (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("\nFree-fall / radius table")
        print(dataframe_to_markdown(freefall_table))
        print("\nBasic parameter table")
        print(dataframe_to_markdown(basic_table))
        print("\nStatistics table")
        print(dataframe_to_markdown(stats_df))
        print(f"\nSaved results to: {output_dir}")
        return 0
    finally:
        freefall_video.release()
        brownian_video.release()


def make_synthetic_brownian_data(params: PhysicsParams, output_dir: Path) -> tuple[Any, Any, dict[str, RadiusResult]]:
    rng = np.random.default_rng(20260515)
    fps = 30.0
    frame_count = 6000
    time_s = np.arange(frame_count, dtype="float64") / fps
    true_radius_m = 0.75e-6
    eta_air = air_viscosity_pa_s(params)
    correction_factor = cunningham_factor(true_radius_m, params)
    eta_slip = eta_air / correction_factor
    true_diffusion_m2_s = TRUE_BOLTZMANN_J_K * params.temperature_k / (6.0 * math.pi * eta_slip * true_radius_m)
    step_sigma_m = math.sqrt(2.0 * true_diffusion_m2_s / fps)
    x_steps_um = rng.normal(0.0, step_sigma_m * 1e6, size=frame_count)
    y_steps_um = rng.normal(0.0, step_sigma_m * 1e6, size=frame_count)
    drift_velocity_um_s = 0.035
    brownian_df = pd.DataFrame(
        {
            "drop_id": "1",
            "frame": np.arange(frame_count),
            "time_s": time_s,
            "x_px": 100.0 + np.cumsum(x_steps_um) / 0.5 + drift_velocity_um_s * time_s / 0.5,
            "y_px": 120.0 + np.cumsum(y_steps_um) / 0.5,
            "x_um": 50.0 + np.cumsum(x_steps_um) + drift_velocity_um_s * time_s,
            "y_um": 60.0 + np.cumsum(y_steps_um),
            "source_video": "synthetic_brownian",
        }
    )

    density_difference = params.oil_density_kgm3 - params.air_density_kgm3
    slip_length_m = params.cunningham_b_pa_m / (2.0 * params.pressure_pa)
    terminal_velocity_m_s = ((true_radius_m + slip_length_m) ** 2 - slip_length_m**2) * (
        2.0 * density_difference * STANDARD_GRAVITY_M_S2
    ) / (9.0 * eta_air)
    fall_time_s = 12.0
    freefall_time_s = np.arange(int(fall_time_s * fps), dtype="float64") / fps
    freefall_df = pd.DataFrame(
        {
            "drop_id": "1",
            "frame": np.arange(len(freefall_time_s)),
            "time_s": freefall_time_s,
            "x_px": 80.0,
            "y_px": 50.0 + (terminal_velocity_m_s * 1e6 * freefall_time_s) / 0.5,
            "x_um": 40.0,
            "y_um": 25.0 + terminal_velocity_m_s * 1e6 * freefall_time_s,
            "source_video": "synthetic_freefall",
        }
    )
    freefall_table, radius_results = analyze_freefall(freefall_df, params)
    freefall_table.to_csv(output_dir / "selftest_freefall_radius_table.csv", index=False)
    return brownian_df, freefall_table, radius_results


def run_self_test(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) / "self_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    params = PhysicsParams(
        temperature_k=float(args.temperature_k),
        pressure_pa=float(args.pressure_pa),
        oil_density_kgm3=float(args.oil_density_kgm3),
        air_density_kgm3=float(args.air_density_kgm3),
        eta0_pa_s=float(args.eta0_pa_s),
        eta0_reference_k=float(args.eta0_reference_k),
        sutherland_c_k=float(args.sutherland_c_k),
        viscosity_model=str(args.viscosity_model),
        cunningham_b_pa_m=float(args.cunningham_b_pa_m),
    )
    brownian_df, freefall_table, radius_results = make_synthetic_brownian_data(params, output_dir)
    dedrift_df, drift_df = dedrift_brownian(brownian_df)
    lag_times_s = parse_lag_times(args.dt_list, float(args.max_lag_s), float(args.dt_step_s))
    stats_df, displacement_map = compute_brownian_statistics(dedrift_df, lag_times_s, fps=30.0)
    msd_fit_df = fit_all_msd(stats_df, args.fit_min_s, args.fit_max_s)
    basic_table = build_basic_parameter_table(["1"], radius_results, msd_fit_df, drift_df, params)
    warn_on_fit_quality(stats_df, msd_fit_df)

    save_table(freefall_table, output_dir, "freefall_radius_table")
    save_table(stats_df, output_dir, "statistics_table")
    save_table(msd_fit_df, output_dir, "msd_fit_table")
    save_table(basic_table, output_dir, "basic_parameter_table")
    dedrift_df.to_csv(output_dir / "brownian_trajectory_dedrifted.csv", index=False)
    plot_trajectories(dedrift_df, output_dir, False)
    plot_displacement_histogram(displacement_map, float(args.hist_lag_s), output_dir, False)
    plot_msd_fit(stats_df, msd_fit_df, output_dir, False)

    recovered_ratio = float(basic_table["k_B_ratio_to_SI"].iloc[0]) if not basic_table.empty else math.nan
    if not (math.isfinite(recovered_ratio) and 0.55 <= recovered_ratio <= 1.45):
        raise AnalysisError(f"Self-test recovered k_B ratio {recovered_ratio:.3g}, outside tolerance.")

    print("Self-test passed.")
    print("\nBasic parameter table")
    print(dataframe_to_markdown(basic_table))
    print(f"\nSaved self-test results to: {output_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Millikan oil-drop videos to estimate Boltzmann's constant from Brownian motion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--freefall-video", type=Path, help="Video A: oil drops falling without electric field.")
    parser.add_argument("--brownian-video", type=Path, help="Video B: balanced oil drops showing Brownian motion.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for tables, plots, and metadata.")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic physics/statistics self-test without videos.")

    parser.add_argument("--grid-spacing-um", type=float, default=600.0, help="Physical distance per graticule interval.")
    parser.add_argument("--grid-intervals", type=float, default=1.0, help="Number of grid intervals between calibration clicks.")
    parser.add_argument("--scale-um-per-px", type=float, help="Known image scale; skips interactive scale calibration.")
    parser.add_argument("--fps", type=float, help="FPS override for both videos.")

    parser.add_argument("--temperature-k", type=float, default=293.15, help="Air temperature in kelvin.")
    parser.add_argument("--pressure-pa", type=float, default=101325.0, help="Air pressure in pascal.")
    parser.add_argument("--oil-density-kgm3", type=float, default=886.0, help="Oil density in kg/m^3.")
    parser.add_argument("--air-density-kgm3", type=float, default=1.204, help="Air density in kg/m^3.")
    parser.add_argument("--eta0-pa-s", type=float, default=1.81e-5, help="Reference air viscosity eta0 in Pa*s.")
    parser.add_argument("--eta0-reference-k", type=float, default=293.15, help="Temperature associated with eta0.")
    parser.add_argument("--sutherland-c-k", type=float, default=110.4, help="Sutherland constant for air.")
    parser.add_argument("--viscosity-model", choices=["sutherland", "constant"], default="sutherland", help="Temperature correction model for air viscosity.")
    parser.add_argument("--cunningham-b-pa-m", type=float, default=8.20e-3, help="Millikan/PASCO Cunningham b constant in Pa*m.")

    parser.add_argument("--tracker", choices=["auto", "trackpy", "csrt", "kcf", "mil", "mosse", "template"], default="auto", help="Tracking backend.")
    parser.add_argument("--particle-polarity", choices=["dark", "bright"], default="dark", help="Whether droplets are darker or brighter than the background.")
    parser.add_argument("--roi-size-px", type=int, default=60, help="Initial ROI size around clicked droplets.")
    parser.add_argument("--search-range-px", type=int, default=24, help="Tracking/linking search range in pixels.")
    parser.add_argument("--selection-radius-px", type=float, default=40.0, help="Maximum selection radius for mapping trackpy particles to clicks.")
    parser.add_argument("--trackpy-diameter-px", type=int, default=11, help="Odd feature diameter for trackpy locate.")
    parser.add_argument("--trackpy-minmass", type=float, help="Optional trackpy minmass threshold.")
    parser.add_argument("--trackpy-memory-frames", type=int, default=3, help="trackpy linking memory in frames.")
    parser.add_argument("--blur-kernel-px", type=int, default=3, help="Gaussian blur kernel before tracking/refinement.")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement.")
    parser.add_argument("--bg-frames", type=int, default=30, help="Frames averaged for static background subtraction (0 to disable).")
    parser.add_argument("--max-frames", type=int, help="Optional frame cap for quick tests.")

    parser.add_argument("--use-timer-seed", action="store_true",
                        help="Auto-detect drop seed from the in-frame timer (upper-right corner). "
                             "Finds the frame where the timer starts, then locates the drop at the zero "
                             "graduation line. Overrides --freefall-points / --brownian-points if omitted.")
    parser.add_argument("--timer-roi", default="950,115,330,35",
                        help="Timer ROI as 'x,y,w,h' pixels targeting the 't:' row. Only used with --use-timer-seed.")
    parser.add_argument("--zero-line-y-px", type=float, default=54.0,
                        help="Y-pixel coordinate of the zero graduation line. Only used with --use-timer-seed.")
    parser.add_argument("--timer-scan-frames", type=int, default=300,
                        help="Number of frames to scan when searching for the timer onset.")
    parser.add_argument("--timer-search-band-px", type=int, default=30,
                        help="Half-band (pixels) around the zero line to search for the drop at timer onset.")

    parser.add_argument("--brownian-points", help="Semicolon-separated Video B points, e.g. '120,80;220,90'.")
    parser.add_argument("--freefall-points", help="Semicolon-separated Video A points, e.g. '100,40'.")
    parser.add_argument("--drop-ids", help="Comma-separated IDs for Brownian drops, e.g. 'A,B,C'.")
    parser.add_argument("--brownian-selection-frame", type=int, default=0, help="Frame used for Video B droplet selection.")
    parser.add_argument("--freefall-selection-frame", type=int, default=0, help="Frame used for Video A droplet selection.")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV click windows; requires scale and point arguments.")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows in addition to saving plots.")

    parser.add_argument("--fall-time-s", type=float,
                        help="Measured time (seconds) for the drop to traverse from the 0 to the 1.6 "
                             "graduation mark.  When supplied, overrides the trajectory-regression "
                             "terminal velocity with v_g = graduation_distance_um / fall_time_s.")
    parser.add_argument("--graduation-distance-um", type=float, default=320.0,
                        help="Physical distance between the two graduation marks used for timing, in µm. "
                             "Default 320 µm (0 → 1.6 marks × 200 µm/mark).")

    parser.add_argument("--dt-list", help="Comma-separated lag times in seconds, e.g. '0.5,1.0,1.5'.")
    parser.add_argument("--max-lag-s", type=float, default=5.0, help="Maximum lag time when --dt-list is omitted.")
    parser.add_argument("--dt-step-s", type=float, default=0.5, help="Lag-time step when --dt-list is omitted.")
    parser.add_argument("--fit-min-s", type=float, help="Minimum lag included in MSD linear fit.")
    parser.add_argument("--fit-max-s", type=float, help="Maximum lag included in MSD linear fit.")
    parser.add_argument("--hist-lag-s", type=float, default=1.0, help="Lag time nearest to this value is used for the displacement histogram.")
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if args.freefall_video is None:
        raise AnalysisError("--freefall-video is required unless --self-test is used.")
    if args.brownian_video is None:
        raise AnalysisError("--brownian-video is required unless --self-test is used.")
    if not Path(args.freefall_video).exists():
        raise AnalysisError(f"Free-fall video does not exist: {args.freefall_video}")
    if not Path(args.brownian_video).exists():
        raise AnalysisError(f"Brownian video does not exist: {args.brownian_video}")
    if args.no_gui and args.scale_um_per_px is None:
        raise AnalysisError("--scale-um-per-px is required with --no-gui.")
    if args.no_gui and not args.brownian_points and not args.use_timer_seed:
        raise AnalysisError("--brownian-points or --use-timer-seed is required with --no-gui.")
    if args.no_gui and not args.freefall_points and not args.use_timer_seed:
        raise AnalysisError("--freefall-points or --use-timer-seed is required with --no-gui.")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    use_agg_backend = bool(args.no_gui or args.self_test or not args.show_plots)
    load_runtime_dependencies(need_video=not args.self_test, use_agg_backend=use_agg_backend)
    try:
        validate_cli_args(args)
        if args.self_test:
            return run_self_test(args)
        return run_analysis(args)
    except AnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())