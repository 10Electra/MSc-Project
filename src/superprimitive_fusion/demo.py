import os
import cv2
import math
import torch
import shutil
import numpy as np
import pyrealsense2 as rs
from natsort import natsort
from typing import Optional, Tuple, List
from vggt.utils.load_fn import load_and_preprocess_images

def detect_yellow(
    frame_rgb: np.ndarray,
    lower_hsv: Tuple[int, int, int] = (15, 127, 109),
    upper_hsv: Tuple[int, int, int] = (28, 255, 255),
    *,
    min_area_frac: float = 0.002,
    k_open: int = 5,
    k_close: int = 11,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB image (uint8).")
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8)

    h, w = frame_rgb.shape[:2]
    img_area = float(h * w)

    k_open = max(1, k_open) | 1
    k_close = max(1, k_close) | 1

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker_close, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * img_area
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return np.zeros_like(mask), None

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return clean, None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return clean, (cx, cy)

def segment_video(predictor, jpg_video_path:str, frame_1_seg_c:tuple, ann_frame_id:int=0, ann_obj_id:int=1):
    inference_state = predictor.init_state(video_path=jpg_video_path)
    predictor.reset_state(inference_state)

    points = np.array([frame_1_seg_c], dtype=np.float32)

    labels = np.array([1], np.int32) # 1 means positive click
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_id,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments

def predict_geometry(model, jpg_video_path:str, image_names:None|list[str], device=torch.device("cuda")):
    if image_names is None:
        image_names = [fname for fname in os.listdir(jpg_video_path) if fname.endswith('.jpg')]
        image_names = natsort.natsorted(image_names)
    images = load_and_preprocess_images([jpg_video_path+img_name for img_name in image_names]).to(device)
    with torch.no_grad():
        predictions = model(images)
    depth = predictions['depth'].cpu().numpy()[0]
    world_points = predictions['world_points'].cpu().numpy()[0]
    world_points_conf = predictions['world_points_conf'].cpu().numpy()[0]
    return world_points, world_points_conf, depth

def record_realsense_colour(pipeline: rs.pipeline,
                            max_frames: Optional[int],
                            window_name: str = "RealSense (r=start/stop, s=keyframe)"
                           ) -> Tuple[List[np.ndarray], List[int]]:
    """
    Show live colour frames from a started RealSense pipeline.
    Press 'r' to start/stop recording. While recording, press 's' to mark a keyframe.
    Returns (frames, keyframes), where:
      - frames is a list of RGB numpy arrays (len(frames) <= max_frames, if max_frames is not None)
      - keyframes is a list of indices into the *returned* frames list
    Display uses BGR; storage is RGB as delivered by rs.format.rgb8.

    max_frames:
      - If None: no cap, return all recorded frames.
      - If an int N: if more than N frames were recorded, uniformly subsample to N.
                     Keyframe indices are re-assigned to the nearest positions preserving relative order.
    """
    def _choose_uniform_indices(M: int, K: int) -> List[int]:
        """Select K indices from range(M) uniformly (centre-of-bin). Assumes 0 < K <= M."""
        return [int(math.floor(((j + 0.5) * M) / K)) for j in range(K)]

    def _remap_keyframes(orig_kf: List[int], M: int, K: int) -> List[int]:
        """Map original keyframe indices (0..M-1) to nearest relative positions in 0..K-1."""
        if K == 0:
            return []
        if M <= 1:
            return [0 for _ in orig_kf]
        mapped = []
        for k in orig_kf:
            r = k / (M - 1)                  # relative position in [0,1]
            j = int(math.floor(r * (K - 1) + 0.5))  # nearest index in new sequence
            j = max(0, min(K - 1, j))
            mapped.append(j)
        return mapped

    frames_saved: List[np.ndarray] = []   # RGB frames saved while recording
    keyframes: List[int] = []            # indices into frames_saved
    recording = False
    last_keyframe_idx = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            colour_frame = frames.get_color_frame()
            if not colour_frame:
                continue

            frame_rgb = np.asanyarray(colour_frame.get_data())

            if recording:
                frames_saved.append(frame_rgb.copy())

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Header
            overlay = frame_bgr.copy()
            h, w = frame_bgr.shape[:2]
            cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0, frame_bgr)
            cv2.putText(frame_bgr, "r: start/stop recording | s: mark keyframe",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Recording status + keyframe count
            if recording:
                status = f"REC \u25CF  saved: {len(frames_saved)}  keyframes: {len(keyframes)}"
                cv2.putText(frame_bgr, status, (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                if not recording:
                    recording = True
                    frames_saved.clear()
                    keyframes.clear()
                    last_keyframe_idx = None
                else:
                    # Stop recording and exit loop
                    break

            elif key == ord('s') and recording:
                idx = len(frames_saved) - 1
                if idx >= 0 and idx != last_keyframe_idx:
                    keyframes.append(idx)
                    last_keyframe_idx = idx

            elif key == 27:  # ESC: exit early (returns what you have; may still be subsampled)
                break

        # Post-process: enforce max_frames and remap keyframes
        M = len(frames_saved)
        if max_frames is None or M <= (max_frames or 0):
            # No subsampling required (or no cap provided)
            return frames_saved, keyframes

        K = max(1, int(max_frames))
        K = min(K, M)
        keep = _choose_uniform_indices(M, K)               # sorted, unique
        frames_out = [frames_saved[i] for i in keep]

        # Remap original keyframe indices (relative positions) into new 0..K-1 index space
        keyframes_out = _remap_keyframes(keyframes, M, K)

        return frames_out, keyframes_out

    finally:
        cv2.destroyWindow(window_name)

def get_frames(res:tuple=(640,480), fps:int=5, max_frames:int=30) -> tuple[list[np.ndarray], list[int]]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, res[0], res[1], rs.format.rgb8, fps)

    try:
        profile = pipeline.start(config)

        # Find sensors
        color_sensor = None
        for s in profile.get_device().query_sensors():
            name = s.get_info(rs.camera_info.name)
            if "RGB" in name:
                color_sensor = s

        if color_sensor:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)  # keep FPS stable

            # Set fixed values (tune for your scene)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 120)
            if color_sensor.supports(rs.option.white_balance):
                color_sensor.set_option(rs.option.white_balance, 3400)
            if color_sensor.supports(rs.option.power_line_frequency):
                color_sensor.set_option(rs.option.power_line_frequency, 1) # UK

        frames, keyframe_idx = record_realsense_colour(pipeline, max_frames=max_frames, window_name='Frame Capture')
    finally:
        pipeline.stop()
    
    return frames, keyframe_idx

def delete_contents(path_to_directory:str):
    filenames = os.listdir(path_to_directory)
    
    # Raise an error if not all images
    for filename in filenames:
        if not filename.endswith('.jpg'):
            raise FileExistsError('Folder contains more than just images')
    
    for filename in filenames:
        file_path = os.path.join(path_to_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def run_realsense_hsv_tuner():
    """
    Opens a window with sliders to tune HSV lower/upper bounds and morphology on a live RealSense colour stream.
    Keys: [s]=print values, [p]=pause/resume, [q]/Esc=quit.
    """
    # --- RealSense pipeline (RGB) ---
    pipeline = rs.pipeline()
    config = rs.config()
    # NOTE: This is RGB. If you prefer BGR frames, use rs.format.bgr8 instead.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # --- UI setup ---
    win = "Yellow HSV Tuner (RealSense)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def tb(name, val, maxv):
        cv2.createTrackbar(name, win, val, maxv, lambda _=None: None)

    defaults = {
        "H low": 15,   "S low": 134,   "V low": 109,
        "H high": 28,  "S high": 255, "V high": 255,
        "OpenK": 5,    "CloseK": 11,
        "MinArea_x100": 20,   # 0.20 %
    }
    tb("H low", defaults["H low"], 179)
    tb("S low", defaults["S low"], 255)
    tb("V low", defaults["V low"], 255)
    tb("H high", defaults["H high"], 179)
    tb("S high", defaults["S high"], 255)
    tb("V high", defaults["V high"], 255)
    tb("OpenK", defaults["OpenK"], 99)
    tb("CloseK", defaults["CloseK"], 151)
    tb("MinArea_x100", defaults["MinArea_x100"], 500)  # up to 5%

    paused = False
    last_lower = (defaults["H low"], defaults["S low"], defaults["V low"])
    last_upper = (defaults["H high"], defaults["S high"], defaults["V high"])

    try:
        profile = pipeline.start(config)
        color_sensor = None
        for s in profile.get_device().query_sensors():
            name = s.get_info(rs.camera_info.name)
            if "RGB" in name:
                color_sensor = s

        if color_sensor:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)  # keep FPS stable

            # Set fixed values (tune for your scene)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 120)
            if color_sensor.supports(rs.option.white_balance):
                color_sensor.set_option(rs.option.white_balance, 3400)
            if color_sensor.supports(rs.option.power_line_frequency):
                color_sensor.set_option(rs.option.power_line_frequency, 1) # UK

        while True:
            if not paused:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_rgb = np.asanyarray(color_frame.get_data())  # RGB

            # Read sliders
            hl = cv2.getTrackbarPos("H low", win)
            sl = cv2.getTrackbarPos("S low", win)
            vl = cv2.getTrackbarPos("V low", win)
            hh = cv2.getTrackbarPos("H high", win)
            sh = cv2.getTrackbarPos("S high", win)
            vh = cv2.getTrackbarPos("V high", win)
            ko = cv2.getTrackbarPos("OpenK", win)
            kc = cv2.getTrackbarPos("CloseK", win)
            min_area_x100 = cv2.getTrackbarPos("MinArea_x100", win)

            # Keep upper >= lower
            hh = max(hh, hl)
            sh = max(sh, sl)
            vh = max(vh, vl)

            lower = (hl, sl, vl)
            upper = (hh, sh, vh)
            min_area_frac = (min_area_x100 / 100.0) / 100.0

            # Detection
            mask, centroid = detect_yellow(
                frame_rgb, lower, upper,
                min_area_frac=min_area_frac,
                k_open=ko, k_close=kc
            )

            # Visuals: original (BGR for imshow), mask, overlay
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay = frame_bgr.copy()
            if centroid is not None:
                x, y = int(round(centroid[0])), int(round(centroid[1]))
                cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            masked_rgb = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
            masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)

            row = np.hstack([frame_bgr, color_mask, overlay if centroid is not None else masked_bgr])

            # Fit to a sensible width
            max_w = 1400
            scale = min(1.0, max_w / row.shape[1])
            if scale < 1.0:
                row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            status = (
                f"Lower {lower}  Upper {upper}  OpenK {ko|1}  CloseK {kc|1}  "
                f"MinArea {min_area_frac*100:.2f}%  "
                f"[s]=save  [p]=pause/resume  [q/Esc]=quit"
            )
            cv2.putText(row, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, row)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                last_lower, last_upper = lower, upper
                break
            elif key in (ord('s'), ord('S')):
                print("\n--- Current settings ---")
                print(f"lower_hsv = ({lower[0]}, {lower[1]}, {lower[2]})")
                print(f"upper_hsv = ({upper[0]}, {upper[1]}, {upper[2]})")
                print(f"k_open = {ko|1}   k_close = {kc|1}")
                print(f"min_area_frac = {min_area_frac:.6f}")
                last_lower, last_upper = lower, upper
            elif key in (ord('p'), ord('P')):
                paused = not paused

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return last_lower, last_upper

def crop_centre(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]