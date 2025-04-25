import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
# Calibration constants
x_min, x_max = -10, 10

y_min, y_max = -0.2, 3
width_m = x_max - x_min
height_m = y_max - y_min

scale_factor = 150.0
margin_x = 200
margin_y = 300
FPS = 15
half_window_sec = 2.5
half_window = int(half_window_sec * FPS)
min_appearances = 10
max_interpolation_gap = half_window  # only interpolate within window
# max_frame_jump = 1 # meters/frame
max_speed_scale = 4
max_avg_frame_delta = 3
max_frame_delta = 15

topdown_width = int(width_m * scale_factor + margin_x)
topdown_height = int(height_m * scale_factor + margin_y)

# Convert real-world coordinates to top-down canvas coordinates
def real_to_canvas(rx, ry):
    shifted_x = rx - x_min
    shifted_y = ry - y_min
    cx = int(shifted_x * scale_factor) + margin_x // 2
    cy = int(shifted_y * scale_factor) + margin_y // 2
    cy = topdown_height - cy
    return cx, cy

# Draw background grid for reference
def draw_grid(canvas):
    grid_spacing_m = 1.0
    for x in np.arange(x_min, x_max + grid_spacing_m, grid_spacing_m):
        cx, _ = real_to_canvas(x, y_min)
        cv2.line(canvas, (cx, 0), (cx, topdown_height), (200, 200, 200), 1)
        cv2.putText(canvas, f"{x:.1f}", (cx + 5, topdown_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    for y in np.arange(y_min, y_max + grid_spacing_m, grid_spacing_m):
        _, cy = real_to_canvas(x_min, y)
        cv2.line(canvas, (0, cy), (topdown_width, cy), (200, 200, 200), 1)
        cv2.putText(canvas, f"{y:.1f}", (5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

class Aggregator:
    def __init__(self, smooth_output_path="output_topdown_smoothed.mp4", raw_output_path="output_topdown_raw.mp4"):
        # store raw detections per frame
        self.frame_clusters = []  # list of clusters per frame
        self.smooth_output_path = smooth_output_path
        self.raw_output_path = raw_output_path

    def update(self, clusters):
        # Store clusters for offline processing
        # clusters: list of dicts with 'pos' and 'labels'
        self.frame_clusters.append([{'pos': c['pos'], 'labels': set(c['labels'])} for c in clusters])

    def finalize(self):
        num_frames = len(self.frame_clusters)
        # Step 1: assign persistent IDs
        tracks = {}  # pid -> {'frames':[], 'positions':[], 'labels': set()}
        next_pid = 0
        for f_idx, clusters in tqdm(enumerate(self.frame_clusters)):
            
            taken_tracks = set()
            cluster2pid = {i: None for i in range(len(clusters))}
            while True:
                best_iou = 0
                best_cluster_id = None
                pid = None
                for id, c in enumerate(clusters):
                    labels = c['labels']       
                    for existing_pid, data in tracks.items():
                        if existing_pid in taken_tracks:
                            continue
                        if f_idx - data['frames'][-1] > max_frame_delta:
                            continue
                        curr_iou = len(labels & data['labels']) / len(labels | data['labels'])
                        # curr_iou = len(labels & data['labels'])
                        if curr_iou > best_iou:
                            best_iou = curr_iou
                            pid = existing_pid
                            best_cluster_id = id
                if pid is None:
                    break
                taken_tracks.add(pid)
                cluster2pid[best_cluster_id] = pid
            
            seen_labels = set()
            for existing_pid, data in tracks.items():
                for label in data['labels']:
                    seen_labels.add(label)
            
            for id, c in enumerate(clusters):
                labels = c['labels']
                pid = cluster2pid[id]
                # if "C2:3" in c['labels'] and "C3:1" in c['labels'] and "C4:1" in c["labels"]:
                #     print(pid)   
                # if "C2:4" in c['labels'] and len(c['labels']) == 1:
                #     print("new ", pid)
                is_new = False
                if pid is None:
                    is_new = True
                    pid = next_pid
                    next_pid += 1
                    tracks[pid] = {'frames': [], 'positions': [], 'labels': set()}
                    
                tracks[pid]['frames'].append(f_idx)
                tracks[pid]['positions'].append(c['pos'])
                tracks[pid]['labels'] |= set(label for label in labels if label not in seen_labels)
                if is_new and len(tracks[pid]['labels']) == 0:
                    tracks[pid]['labels'] |= labels | set([str(i) for i in range(10)]) # some trash
                    
        # Step 2: filter out short-lived tracks
        valid_pids = set()
        for pid, d in tracks.items():
            # if pid == 71:
            #     print(d)
            # 1) must appear in at least min_appearances frames
            if len(d['frames']) < min_appearances:
                continue

            # 2) compute per‐step rate = distance / frame_delta,
            #    drop if any rate > max_frame_jump
            drop = False
            avg_frame_delta = 0
            for i in range(1, len(d['positions'])):
                f0, f1 = d['frames'][i-1],    d['frames'][i]
                frame_delta = f1 - f0
                avg_frame_delta += frame_delta
            avg_frame_delta /= len(d['positions']) - 1
            
            # if avg_frame_delta < max_avg_frame_delta:
            valid_pids.add(pid)
        # Step 3: build per-frame raw positions map
        pid_positions = {pid: [None]*num_frames for pid in valid_pids}
        for pid in valid_pids:
            for f, pos in zip(tracks[pid]['frames'], tracks[pid]['positions']):
                pid_positions[pid][f] = pos
        # Step 4: outlier removal & interpolation
        for pid, positions in pid_positions.items():
            for f in range(num_frames):
                pos = positions[f]
                # detect outlier
                if pos is not None:
                    # collect up to 5 valid frames before and after f
                    prev_idxs = []
                    i = f
                    while i >= 0 and len(prev_idxs) < 5:
                        if positions[i] is not None:
                            prev_idxs.append(i)
                        i -= 1

                    next_idxs = []
                    i = f
                    while i < num_frames and len(next_idxs) < 5:
                        if positions[i] is not None:
                            next_idxs.append(i)
                        i += 1

                    # only if we have any data to compare
                    if prev_idxs or next_idxs:
                        # build speed lists
                        speeds_prev = [
                            euclidean(positions[prev_idxs[idx]], positions[prev_idxs[idx + 1]]) / abs(prev_idxs[idx + 1] - prev_idxs[idx])
                            for idx in range(len(prev_idxs) - 1)
                        ]
                        speeds_next = [
                            euclidean(positions[next_idxs[idx]], positions[next_idxs[idx + 1]]) / abs(next_idxs[idx + 1] - next_idxs[idx])
                            for idx in range(len(next_idxs) - 1)
                        ]

                        # merge to get a single median
                        all_speeds = speeds_prev + speeds_next
                        med_speed = np.median(all_speeds)

                        # outlier if left‐side has full window and its max > median×scale
                        if len(speeds_prev) == 5 and max(speeds_prev) > med_speed * max_speed_scale:
                            positions[f] = None
                            pos = None

                        # same for right side
                        elif len(speeds_next) == 5 and max(speeds_next) > med_speed * max_speed_scale:
                            positions[f] = None
                            pos = None
            for f in range(num_frames):
                pos = positions[f]
                # interpolate gap
                if pos is None:
                    prev_idx = next((i for i in range(f-1, max(-1, f-half_window-1), -1) if positions[i] is not None), None)
                    next_idx = next((i for i in range(f+1, min(num_frames, f+half_window+1)) if positions[i] is not None), None)
                    if prev_idx is not None and next_idx is not None:
                        t = (f - prev_idx) / (next_idx - prev_idx)
                        p0, p1 = positions[prev_idx], positions[next_idx]
                        positions[f] = ((p0[0] + t*(p1[0]-p0[0])), (p0[1] + t*(p1[1]-p0[1])))
        # Step 5: smoothing
        smoothed = {pid: [None]*num_frames for pid in valid_pids}
        for pid, positions in pid_positions.items():
            
            for f in range(num_frames):
                xs = ys = ws = 0.0
                for w in range(-half_window, half_window+1):
                    idx = f + w
                    if 0 <= idx < num_frames and positions[idx] is not None:
                        weight = 1.0/(abs(w)+1)
                        xs += positions[idx][0]*weight
                        ys += positions[idx][1]*weight
                        ws += weight
                prev_exists = any(
                    positions[i] is not None for i in range(0, f+1)
                )
                next_exists = any(
                    positions[i] is not None
                    for i in range(f, num_frames)
                )
                if ws > 0 and prev_exists and next_exists:
                    smoothed[pid][f] = (xs/ws, ys/ws)
        # Step 6: initialize writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_raw = cv2.VideoWriter(self.raw_output_path, fourcc, FPS, (topdown_width, topdown_height))
        writer_smooth = cv2.VideoWriter(self.smooth_output_path, fourcc, FPS, (topdown_width, topdown_height))
        # Step 7: render frames
        for f in range(num_frames):
            # Raw canvas
            raw_canvas = np.ones((topdown_height, topdown_width, 3), dtype=np.uint8)*255
            draw_grid(raw_canvas)
            for c in self.frame_clusters[f]:
                pos = c['pos']
                cx, cy = real_to_canvas(*pos)
                # color per camera origin
                cam_id = int(next(iter(c['labels'])).split(':')[0][1:])
                color = (255, 0, 0) if cam_id == 1 else (0, 128, 0)
                if len(c['labels']) > 1:
                    color = (0, 0, 255)
                cv2.circle(raw_canvas, (cx, cy), 8, color, -1)
                label_text = " | ".join(sorted(c['labels']))
                cv2.putText(raw_canvas, label_text, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writer_raw.write(raw_canvas)
            # Smoothed canvas
            smooth_canvas = np.ones((topdown_height, topdown_width, 3), dtype=np.uint8)*255
            draw_grid(smooth_canvas)
            for pid in valid_pids:
                s = smoothed[pid][f]
                if s:
                    cx, cy = real_to_canvas(*s)
                    color = (0,0,255) if len(tracks[pid]['labels'])>1 else (255,0,0)
                    cv2.circle(smooth_canvas, (cx, cy), 8, color, -1)
                    cv2.putText(smooth_canvas, f"P{pid}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writer_smooth.write(smooth_canvas)
        writer_raw.release()
        writer_smooth.release()
    def __del__(self):
        self.finalize()
