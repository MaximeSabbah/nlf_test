import torch
import numpy as np
import cv2
import time
from equilib import Equi2Pers, Equi2Equi
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import sys
import os
import torchvision.transforms as T
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

import sys

from equilib.torch_utils.grid import create_grid
import torch.nn.functional as F
import numpy as np




unNormalize = transforms.Normalize(
        mean=-np.array([0.485,0.456,0.406]) / np.array([0.229,0.224,0.225]),
        std=1 / np.array([0.229,0.224,0.225]))


def preprocess(img: np.ndarray, device: str = "cuda") -> torch.Tensor:
    #img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (H,W,C) → (C,H,W)
    img_tensor = torch.from_numpy(img).to(device, non_blocking=True).permute(2, 0, 1).float().div(255).unsqueeze(0)#.half()
    return img_tensor # Add batch dim & move to GPU





def postprocess_debug(img: torch.Tensor, to_cv2: bool = False) -> np.ndarray:
    """Post-traiter l'image en la ramenant à un format compatible pour OpenCV."""
    img = img.squeeze(0).cpu().detach().numpy()  # Retirer la dimension batch et revenir sur CPU
    img = np.transpose(img, (1, 2, 0))  # Changer l'ordre des axes pour OpenCV
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Normalisation
    if to_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Retour à BGR pour OpenCV
    return img

def inspect_image_format(image):

    # Check if the image is already a tensor
    if isinstance(image, torch.Tensor):
        print(f"Image is already a tensor with shape: {image.shape}")
    else:
        print(f"Image is not a tensor. Type: {type(image)}")
    
    # Check if it has the expected format (batch, channels, height, width)
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:  # (batch_size, channels, height, width)
            print(f"Image shape is valid for the pose estimator: {image.shape}")
        else:
            print(f"Unexpected tensor shape: {image.shape}")
    return image



def preprocess_for_nlf(frame):
    return (frame * 255.0).to(dtype=torch.float16, device='cuda')



'''def preprocess_for_nlf(img_tensor):
    """
    img_tensor: torch.Tensor [3,H,W] or [1,3,H,W], float32 or float16, in [0,1]
    Assumes input is already in RGB format and in [0,1] (e.g., pseudo_camera /255)
    Returns: [1, 3, H, W], float32, normalized for NLF
    """
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # [1,3,H,W]

    # Ensure float32 for stability
    img_tensor = img_tensor.to(dtype=torch.float32)

    # Normalize (ImageNet stats, like in ViT / NLF)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)

    img_tensor = (img_tensor - mean) / std
    return img_tensor'''



def plot_keypoints_3d(keypoints):
    """
    Plot 3D keypoints to debug and visualize them.

    Parameters:
    keypoints: List or array of shape (N, 3) where N is the number of keypoints and each keypoint is [X, Y, Z].
    """
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Unzip keypoints into separate X, Y, Z lists
    X = [kp[0] for kp in keypoints]
    Y = [kp[1] for kp in keypoints]
    Z = [kp[2] for kp in keypoints]

    # Plot the keypoints in 3D
    ax.scatter(X, Y, Z, c='r', marker='o')

    # Optionally, plot lines between keypoints to indicate connections (e.g., bones)
    #for i in range(len(keypoints) - 1):
        #ax.plot([X[i], X[i+1]], [Y[i], Y[i+1]], [Z[i], Z[i+1]], c='b')

    # Set labels for axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set the title
    ax.set_title('3D Keypoints Debug')

    # Show the plot
    plt.show()



def generate_trc_file(pred_verts, cfg_bio, output_path, frame_rate, R):
    """
    Generate a TRC file for Mokka with transformed 3D coordinates.

    Args:
        pred_verts (list of np.array): List of (N, 3) numpy arrays, one per frame, in camera coordinates.
        cfg_bio (dict): Mapping of marker names to their corresponding indices in pred_verts.
        output_path (str): Path to save the TRC file.
        frame_rate (float): Frame rate of the motion capture data.
        R (numpy array): 3x3 Rotation matrix for transforming to the correct reference frame.
    """

    # Number of frames and markers
    num_frames = len(pred_verts)
    num_markers = len(cfg_bio)
    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
    # TRC Header
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{frame_rate:.2f}\t{frame_rate:.2f}\t{num_frames}\t{num_markers}\tmm\t{frame_rate:.2f}\t1\t{num_frames}",
        "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in cfg_bio.keys()]),  # Marker names
        "\t\t" + marker_headers  # X, Y, Z headers
    ]

    # Initialize TRC data
    trc_data = []

    # Iterate over frames and transform the points
    for frame_idx, frame_verts in enumerate(pred_verts):
        time = frame_idx / frame_rate  # Compute time in seconds
        frame_data = [str(frame_idx + 1), f"{time:.5f}"]  # Frame number and timestamp
        keypoint_list=[]
        # Transform each marker point
        for marker in cfg_bio.keys():
            idx = cfg_bio[marker]  # Get the marker index
            P_cam = frame_verts[idx]  # Get the 3D point in camera coordinates
            #P_world = np.array(P_cam) @ np.array(R[frame_idx])  
            #P_cam = [-P_cam[2], -P_cam[1] + 750, -P_cam[0]]
            #P_world = P_cam
            # Apply rotation matrix
            P_world = np.array(P_cam) @ np.array(R[frame_idx])             
            keypoint_list.append(P_world)
            
            #P_world = R[frame_idx] @ P_cam  # Matrix multiplication

            # Format and add to frame data
            frame_data.extend([f"{P_world[0]:.5f}", f"{P_world[1]:.5f}", f"{P_world[2]:.5f}"])
        #plot_keypoints_3d(pred_verts[0][0])
        trc_data.append("\t".join(frame_data))  # Add formatted data row

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")  # Write header
        f.write("\n".join(trc_data) + "\n")  # Write data rows

    print(f"TRC file successfully saved to {output_path}")



def generate_trc_file_subset(pred_verts, cfg_bio, output_path, frame_rate, R):
    """
    Generate a TRC file for Mokka with transformed 3D coordinates.

    Args:
        pred_verts (list of np.array): One (N, 3) array per frame, in correct marker order.
        cfg_bio (dict): Mapping of marker names to indices (used only for names, order assumed to match pred_verts).
        output_path (str): Path to save the TRC file.
        frame_rate (float): Frame rate of the motion capture data.
        R (list of np.array): One (3, 3) rotation matrix per frame.
    """

    num_frames = len(pred_verts)

    first_valid_frame = next((frame for frame in pred_verts if len(frame) > 0), None)
    if first_valid_frame is None:
        raise ValueError("No valid frames with markers found.")

    num_markers = len(first_valid_frame)
    marker_names = list(cfg_bio.keys())
    if len(marker_names) != num_markers:
        raise ValueError("cfg_bio and pred_verts length mismatch.")

    # Header construction
    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
    marker_names_line = "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{frame_rate:.2f}\t{frame_rate:.2f}\t{num_frames}\t{num_markers}\tmm\t{frame_rate:.2f}\t1\t{num_frames}",
        marker_names_line,
        "\t\t" + marker_headers
    ]

    trc_data = []

    for frame_idx, frame_verts in enumerate(pred_verts):
        time = frame_idx / frame_rate
        frame_data = [str(frame_idx + 1), f"{time:.5f}"]

        # Apply rotation
        transformed = np.array(frame_verts)@ np.array(R[frame_idx])
        for x, y, z in transformed:
            frame_data.extend([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])

        trc_data.append("\t".join(frame_data))

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(trc_data) + "\n")

    print(f"TRC file successfully saved to {output_path}")


def generate_trc_file_world_frame(
    pred_verts, cfg_bio, output_path, timestamps_s, R_list,
    units="mm", world_offset_y=0.75
):
    """
    TRC writer where each frame uses the SAME transformation as the RTOSIM socket:

        1. vertices3d from NLF are in mm
        2. convert to meters
        3. apply camera rotation R
        4. world transform: (-z, -y + offset, -x)
        5. convert back to mm for TRC (OpenSim prefers mm)

    Args:
        pred_verts : list of (M,3) in mm
        cfg_bio    : dict {marker_name: idx}
        timestamps_s : per-frame timestamps (s)
        R_list     : per-frame (3,3) rotation matrices
        units      : "mm" (default)
        world_offset_y : vertical shift, 0.75 m
    """

    import numpy as np

    marker_names = list(cfg_bio.keys())
    M = len(marker_names)
    N = len(pred_verts)

    if not (len(R_list) == len(timestamps_s) == N):
        raise ValueError("pred_verts, timestamps_s, and R_list must match in length.")

    # ---- normalize timestamps ----
    ts = np.asarray(timestamps_s, dtype=np.float64)
    ts -= ts[0]

    # ---- prepare output matrix ----
    fixed_V = []
    for v_mm, R in zip(pred_verts, R_list):
        v_mm = np.asarray(v_mm, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        if v_mm.shape != (M, 3):
            v_mm = np.full((M, 3), np.nan)

        # mm → m
        v_m = v_mm / 1000.0

        # apply camera rotation
        v_cam = v_m @ R.T

        # apply world transform (same as RTOSIM)
        v_world = np.zeros_like(v_cam)
        for i, (x, y, z) in enumerate(v_cam):
            v_world[i] = np.array([
                -z,
                -y + world_offset_y,
                -x
            ])

        # convert back to mm (OpenSim)
        v_world_mm = v_world * 1000.0

        fixed_V.append(v_world_mm)

    # ---- build TRC header ----
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    data_rate = (1.0 / np.median(diffs)) if diffs.size else 0.0

    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(M)])
    marker_names_line = "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.2f}\t{data_rate:.2f}\t{N}\t{M}\t{units}\t{data_rate:.2f}\t1\t{N}",
        marker_names_line,
        "\t\t" + marker_headers
    ]

    # ---- write frames ----
    lines = []
    for i in range(N):
        verts = fixed_V[i]
        t = float(ts[i])
        row = [str(i + 1), f"{t:.5f}"]

        for x, y, z in verts:
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                row += [f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
            else:
                row += ["", "", ""]

        lines.append("\t".join(row))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")

    print(f"[TRC] Saved world-frame TRC → {output_path}")
    print(f"[TRC] Frames={N}, Markers={M}, DataRate~{data_rate:.2f} Hz")



def generate_trc_file_subset_native_time(pred_verts, cfg_bio, output_path, timestamps_s, R_list, units="mm", scale=None):
    """
    TRC writer using native timestamps (seconds). Robust to missing frames.

    Args:
        pred_verts : list of (M,3)-like arrays/lists per frame (marker order must match cfg_bio keys order).
        cfg_bio    : dict mapping marker_name -> index (order of keys is used for header).
        output_path: str
        timestamps_s: list/array of length N (seconds; irregular OK)
        R_list     : list of (3,3) rotation matrices per frame (len N). If an R is bad, uses I.
        units      : "mm" or "m" for TRC header
        scale      : optional float to multiply coordinates (e.g., 1000.0 if verts in meters and you want mm)
    """
    import numpy as np

    # ---------- basic checks ----------
    marker_names = list(cfg_bio.keys())
    M = len(marker_names)
    N = len(pred_verts)
    if not (len(R_list) == len(timestamps_s) == N):
        raise ValueError("Lengths of pred_verts, R_list, and timestamps_s must match.")

    # ---------- timestamps: normalize & ensure non-decreasing ----------
    ts = np.asarray(timestamps_s, dtype=np.float64)
    if ts.size == 0:
        raise ValueError("Empty timestamps_s.")
    ts = ts - ts[0]
    if not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts)
        ts = ts[order]
        pred_verts = [pred_verts[i] for i in order]
        R_list     = [R_list[i]     for i in order]

    # ---------- sanitize verts & rotations per frame ----------
    fixed_V = []
    fixed_R = []
    I3 = np.eye(3, dtype=np.float64)
    for v, R in zip(pred_verts, R_list):
        v_arr = np.asarray(v, dtype=np.float64)
        if v_arr.shape != (M, 3):
            # coerce any bad/missing frame to NaNs of correct shape
            v_arr = np.full((M, 3), np.nan, dtype=np.float64)
        if scale is not None:
            v_arr = v_arr * float(scale)

        R_arr = np.asarray(R, dtype=np.float64)
        if R_arr.shape != (3, 3) or not np.isfinite(R_arr).all():
            R_arr = I3.copy()

        fixed_V.append(v_arr)
        fixed_R.append(R_arr)

    # ---------- header fields ----------
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    data_rate = (1.0 / np.median(diffs)) if diffs.size else 0.0

    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(M)])
    marker_names_line = "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.2f}\t{data_rate:.2f}\t{N}\t{M}\t{units}\t{data_rate:.2f}\t1\t{N}",
        marker_names_line,
        "\t\t" + marker_headers
    ]

    # ---------- rows ----------
    lines = []
    for i in range(N):
        verts = fixed_V[i] @ fixed_R[i]  # (M,3) * (3,3)
        t = float(ts[i])
        row = [str(i + 1), f"{t:.5f}"]
        for x, y, z in verts:
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                row += [f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
            else:
                row += ["", "", ""]  # TRC blank component for missing
        lines.append("\t".join(row))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")

    print(f"TRC (native time) saved → {output_path}  |  ~DataRate {data_rate:.2f} Hz  |  Frames {N}  Markers {M}")




def compute_camera_parameters(out_width, out_height, fov, yaw, pitch):
    """
    Compute the intrinsic and extrinsic parameters of the pseudo-camera.
    
    Args:
        out_width (int): Output perspective image width.
        out_height (int): Output perspective image height.
        fov (float): Field of view in degrees.
        yaw (float): Yaw angle in degrees.
        pitch (float): Pitch angle in degrees.
    
    Returns:
        K (numpy array): Intrinsic matrix (3x3).
        R (numpy array): Rotation matrix (3x3).
        t (numpy array): Translation vector (3x1).
    """
    # Convert angles to radians
    #print('yaw', yaw)
    fov_rad = np.radians(fov)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    #print('pitch', pitch, pitch_rad)
    # Compute focal length
    f_x = f_y = out_width / (2 * np.tan(fov_rad / 2))
    c_x, c_y = out_width / 2, out_height / 2

    # Intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    # Rotation matrix
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    '''R_pitch = np.array([
    [np.cos(pitch_rad), -np.sin(pitch_rad), 0],  # X and Y affected
    [np.sin(pitch_rad), np.cos(pitch_rad), 0],   # X and Y affected
    [0, 0, 1]                                    # Z unaffected
])'''
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ], dtype=np.float32)
    
    #R = R_pitch @ R_yaw
   #R = R_yaw @ R_pitch
    R = R_pitch @ R_yaw

    # Translation vector
    t = np.zeros((3, 1))  # No translation in this case

    return K, R, t
    







# Function to rotate the image by 180 degrees
def rotate_image(image, yaw):
    width = int(image.shape[1])
    pixel = int((yaw + 180)*width/360)
    return np.roll(image, int((width/2)-pixel), axis=1)

# Function to check for T-pose : T-Pose detected if angles between arms and trunk are below 110 (180 is when they are aligned to trunk)

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cosine_angle = dot_product / magnitude if magnitude != 0 else 0
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def is_t_pose(keypoints):
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_elbow, right_elbow = keypoints[7], keypoints[8]
    left_hip, right_hip = keypoints[11], keypoints[12]

    left_vec = np.array(left_elbow) - np.array(left_shoulder)
    right_vec = np.array(right_elbow) - np.array(right_shoulder)
    left_vertical = np.array(left_shoulder) - np.array(left_hip)
    right_vertical = np.array(right_shoulder) - np.array(right_hip)

    left_angle = calculate_angle(left_vec, left_vertical)
    right_angle = calculate_angle(right_vec, right_vertical)
    #print('ANGLES', left_angle, right_angle)
    return (left_angle <= 130 and right_angle <= 130), left_angle, right_angle


# Define a margin : for the bounding box not being too close to the edge  ; or to detect if the bbox is in the middle
MARGIN_PERCENTAGE = 0.01  # 5% margin from the edge, adjust as needed
MARGIN_PERCENTAGE_middle = 0.05
# Function to check if the bounding box is too close to the edge
def is_bbox_too_close_to_edge(x, y, w, h, frame_width, frame_height, margin_percentage=MARGIN_PERCENTAGE):
    margin_x = frame_width * margin_percentage
    margin_y = frame_height * margin_percentage
    
    # Check if any part of the bounding box is too close to the edge
    if x - w / 2 < margin_x or x + w / 2 > frame_width - margin_x or \
       y - h / 2 < margin_y or y + h / 2 > frame_height - margin_y:
        return True
    return False

# Put near your other constants if you want to tweak defaults
CENTER_BAND_RATIO = 0.10  # 10% of frame width around center is “preferred”

def is_bbox_middle(x, y, w, h, frame_width, frame_height, margin_percentage=CENTER_BAND_RATIO):
    """
    Ultralytics boxes.xywh -> (cx, cy, bw, bh)
    Return True if bbox center is within a central vertical band around frame center.
    """
    cx = float(x)
    band_half = frame_width * float(margin_percentage)  # e.g., 0.10 * 3840 = 384 px
    center_x = frame_width * 0.5
    return (center_x - band_half) <= cx <= (center_x + band_half)

def start_tracking(result, id_a_suivre, frame_width=3840, frame_height=1920, center_band_ratio=CENTER_BAND_RATIO):
    """
    Pick a track to follow.
    1) Prefer someone whose bbox center is already in the central band;
       if multiple, choose the one closest to exact center.
    2) If none in band, choose the person closest to center anyway.
    Returns the chosen track id (int) or None if no valid track ids yet.
    """
    if not (result and result.boxes is not None) or id_a_suivre is not None:
        return id_a_suivre

    boxes_xywh = result.boxes.xywh.cpu().numpy()                       # (N,4) -> (cx,cy,w,h)
    track_ids = (result.boxes.id.int().cpu().tolist()
                 if result.boxes.id is not None else [-1] * len(boxes_xywh))

    # Gather candidates with valid ids
    candidates = []
    center_x = frame_width * 0.5
    for (cx, cy, bw, bh), tid in zip(boxes_xywh, track_ids):
        if tid == -1:
            continue  # no stable tracker id yet
        dist = abs(float(cx) - center_x)
        in_band = is_bbox_middle(cx, cy, bw, bh, frame_width, frame_height, center_band_ratio)
        candidates.append((in_band, dist, tid))

    if not candidates:
        # No valid tracked IDs yet — let the caller try again next frame
        return None

    # 1) Prefer in-band, pick the closest to center; else 2) pick the closest overall
    in_band_candidates = [c for c in candidates if c[0]]
    chosen = min(in_band_candidates, key=lambda x: x[1]) if in_band_candidates else min(candidates, key=lambda x: x[1])

    _, _, chosen_id = chosen
    return chosen_id


# For a given result,     
def detect_t_pose(result, t_pose_person, t_pose_threshold, t_pose_duration, width=3840, height=1920):
    if result and result.boxes is not None and t_pose_person is None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes)
        keypoints = result.keypoints.xy.cpu().numpy()

        for idx, keypoint in enumerate(keypoints):
            t_pose_detected, _, _ = is_t_pose(keypoint)
            
            x, y, w, h = boxes[idx]

            if is_bbox_too_close_to_edge(x, y, w, h, width, height):
                
                continue  

            if t_pose_detected:
                #print('hehehe')
                if track_ids[idx] not in t_pose_duration:
                    t_pose_duration[track_ids[idx]] = 0
                t_pose_duration[track_ids[idx]] = 1 + t_pose_duration[track_ids[idx]]
                

                if t_pose_duration[track_ids[idx]] >= t_pose_threshold:
                    return True, track_ids[idx], t_pose_threshold, t_pose_duration

    return False, t_pose_person, t_pose_threshold, t_pose_duration


# For a given result, and index of a person, returns a boolean True and the yaw angle of the center of its bounding box         
def align_equi_to_tposed(result, t_pose_person):
    if not result or result.boxes is None:
        return False, 0.0, 0.0

    w_img, h_img = 3840, 1920

    boxes = result.boxes.xywh.cpu().numpy()

    # Track IDs
    track_ids = (
        result.boxes.id.int().cpu().tolist()
        if result.boxes.id is not None
        else [-1] * len(boxes)
    )

    keypoints = result.keypoints.xy.cpu().numpy()

    for idx in range(len(keypoints)):
        if track_ids[idx] != t_pose_person:
            continue

        kp = keypoints[idx]

        # Shoulder indices (YOLOv8/YOLO11 standard)
        L = kp[5]   # left shoulder
        R = kp[6]   # right shoulder

        cx = R[0]
        cy = R[1]

        yaw_deg   = (cx / w_img) * 360.0 - 180.0
        pitch_deg = -(cy / h_img) * 180.0 + 90.0

        print(f"[ALIGN] shoulders cx={cx:.1f}, cy={cy:.1f}, yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}")

        return True, yaw_deg, pitch_deg

    return False, 0.0, 0.0




            
# If no one is detected, returns the previous camera extrinsic parameters not to skip frames.
# Would be great to have a controlable 'safe zone', as by default bitetrack keeps the track of an id for a maximum of 30 consecutive frames if it disappears.
def main_tracking(result, id_a_suivre, rotate_frame, pitch): 
    if result and result.boxes is not None and id_a_suivre is not None:
        boxes = result.boxes.xywh.cpu().numpy()

        # Check if track IDs exist before accessing them
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)  # Assign -1 if no track ID exists

        keypoints = result.keypoints.xy.cpu().numpy()

        for idx, keypoint in enumerate(keypoints):
            if track_ids[idx] == id_a_suivre:
                # Get bounding box coordinates

                L = keypoints[idx][5]   # [x,y] left shoulder
                R = keypoints[idx][6]   # [x,y] right shoulder

                cx = R[0]
                cy = R[1]

                yaw = (cx / 3840) * 360 - 180
                pitch = -(cy / 1920) * 180 + 90

                return rotate_frame, pitch

    # If no return was reached, return 0
    return rotate_frame, pitch
            




def compute_yaw_pitch_from_vertices(vertices3d):
    """
    Estimate yaw/pitch offsets (in degrees) from NLF 3D vertices.

    vertices3d: np.ndarray of shape (M, 3), camera-frame coordinates.
    We use the centroid of all markers as a proxy for the body center.

    Returns:
        delta_yaw_deg, delta_pitch_deg
        (angles telling us how far the subject is from the camera center)
    """
    import numpy as np

    v = np.asarray(vertices3d, dtype=np.float32)

    if v.ndim != 2 or v.shape[1] != 3:
        return 0.0, 0.0

    # Use centroid of all markers as tracking anchor
    center = np.nanmean(v, axis=0)  # (3,)

    if not np.all(np.isfinite(center)):
        return 0.0, 0.0

    x, y, z = center
    norm = float(np.linalg.norm(center))
    if norm < 1e-6 or not np.isfinite(norm):
        return 0.0, 0.0

    # Angles in camera frame:
    # - yaw: horizontal angle (x vs z)
    # - pitch: vertical angle (y vs radius)
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(np.clip(y / norm, -1.0, 1.0))

    yaw_deg = float(np.degrees(yaw))
    pitch_deg = float(np.degrees(pitch))

    return yaw_deg, pitch_deg


def compute_yaw_pitch_from_2d(pose2d, img_w=256, img_h=256):
    """
    pose2d: numpy array of shape (N,2) with [x,y] in pseudo-camera coordinates.
    Returns:
        delta_yaw_deg, delta_pitch_deg
    """

    import numpy as np

    # Basic validity check
    if pose2d is None or len(pose2d.shape) != 2 or pose2d.shape[1] != 2:
        print("[TRACK-2D] pose2d invalid shape:", None if pose2d is None else pose2d.shape)
        return 0.0, 0.0

    # Try torso-based tracking: shoulders + hips
    torso_indices = [5, 6, 11, 12]  # Works with YOLO indexing & NLF output structure
    
    try:
        # Filter indices that are in range
        valid_indices = [i for i in torso_indices if i < pose2d.shape[0]]
        torso_pts = pose2d[valid_indices]  # shape (K,2)

        # Remove invalid values (NaN)
        torso_pts = torso_pts[np.all(np.isfinite(torso_pts), axis=1)]

        if len(torso_pts) == 0:
            raise ValueError("No valid torso keypoints")

        C = torso_pts.mean(axis=0)  # shape (2,)
    except Exception:
        # Fallback to centroid of all points
        pts = pose2d[np.all(np.isfinite(pose2d), axis=1)]
        if len(pts) == 0:
            print("[TRACK-2D] No valid 2D pts, returning 0.")
            return 0.0, 0.0
        C = pts.mean(axis=0)

    # At this point, C **must** be a 1D array of shape (2,)
    C = np.asarray(C).reshape(2,)   # <- FIX: FORCE shape (2,)

    cx, cy = float(C[0]), float(C[1])

    # Normalized offsets: [-1,1]
    dx = (cx - img_w/2) / (img_w/2)
    dy = (cy - img_h/2) / (img_h/2)

    # Convert to yaw/pitch deltas.
    # NOTE: signs will be tuned after test
    delta_yaw_deg   =  dx * 10.0        # horizontal
    delta_pitch_deg = -dy * 10.0        # vertical (image y is downward)

    # Debug: print small info occasionally
    # print(f"[TRACK-2D] cx={cx:.1f}, cy={cy:.1f}, dx={dx:.3f}, dy={dy:.3f} → yaw={delta_yaw_deg:.2f}, pitch={delta_pitch_deg:.2f}")

    return delta_yaw_deg, delta_pitch_deg

def compute_yaw_pitch_from_shoulder_2d(pose2d, idx_shoulder,
                                       img_w=256, img_h=256,
                                       gain_yaw=10.0, gain_pitch=10.0):
    """
    Compute yaw and pitch corrections so that the chosen shoulder keypoint
    stays centered in the pseudo-camera.
    """

    import numpy as np

    if pose2d is None or pose2d.ndim != 2 or pose2d.shape[1] != 2:
        return 0.0, 0.0

    try:
        x, y = pose2d[idx_shoulder]
        if not np.isfinite(x) or not np.isfinite(y):
            return 0.0, 0.0
    except:
        return 0.0, 0.0

    # Normalized offsets relative to image center
    dx = (x - img_w/2) / (img_w/2)
    dy = (y - img_h/2) / (img_h/2)

    # Convert to angular offsets
    delta_yaw   =  dx * gain_yaw
    delta_pitch = -dy * gain_pitch   # image y grows downward → pitch flips

    return delta_yaw, delta_pitch
