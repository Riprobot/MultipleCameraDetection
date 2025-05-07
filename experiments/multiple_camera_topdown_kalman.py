import cv2
import numpy as np
import supervision as sv  # If needed, else remove
from ultralytics import YOLO
from collections import defaultdict
from reid import get_features
from camera_calib import Camera
from scipy.spatial.distance import cosine
from itertools import combinations

from MultipleCameraDetection.experiments.kalman_filtering import KalmanTrackerManager

def distance_2d(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def main():
    calibration_files = [
        "data/calibration1.py",
        "data/calibration2.py",
        "data/calibration3.py"
    ]
    video_sources = [
        "data/cam1.mp4",
        "data/cam2.mp4",
        "data/cam3.mp4"
    ]
    
    num_cameras = len(calibration_files)
    
    cams = [Camera(calibration_files[i]) for i in range(num_cameras)]
    models = [YOLO("yolo11x.pt") for _ in range(num_cameras)]
    
    tracker_config = "botsort.yaml"  
    
    tracking_generators = []
    for i in range(num_cameras):
        gen = models[i].track(
            source=video_sources[i],
            stream=True,
            tracker=tracker_config,
            conf=0.4,
            iou=0.75,
            imgsz=(1280, 720),
            classes=[0],  
            batch=8,
            augment=True,
        )
        tracking_generators.append(gen)
    
    captures = [cv2.VideoCapture(video_sources[i]) for i in range(num_cameras)]
    for idx, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Could not open {video_sources[idx]}")
            return
    
    fps_list = [cap.get(cv2.CAP_PROP_FPS) or 30 for cap in captures]
    frame_widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in captures]
    frame_heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in captures]
    
    max_frames = int(fps_list[0] * 20)
    
    x_min, x_max = -10, 10
    y_min, y_max = -0.2, 2
    width_m = x_max - x_min
    height_m = y_max - y_min

    scale_factor = 150.0
    margin_x = 200
    margin_y = 300
    topdown_width = int(width_m * scale_factor + margin_x)
    topdown_height = int(height_m * scale_factor + margin_y)

    def real_to_canvas(rx, ry):
        shifted_x = rx - x_min
        shifted_y = ry - y_min
        cx = int(shifted_x * scale_factor) + margin_x // 2
        cy = int(shifted_y * scale_factor) + margin_y // 2
        cy = topdown_height - cy
        return cx, cy

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
    
    out_cams = []
    for i in range(num_cameras):
        writer = cv2.VideoWriter(
            f"output_cam{i+1}_annotated.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_list[i],
            (frame_widths[i], frame_heights[i])
        )
        out_cams.append(writer)
    
    out_topdown = cv2.VideoWriter(
        "output_topdown_combined.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps_list),
        (topdown_width, topdown_height)
    )
    
    kalman_manager = KalmanTrackerManager(dt=1.0)
    
    frame_count = 0

    match_history = [] 

    while True:
        frames = []
        ret_flags = []
        for cap in captures:
            ret, frame = cap.read()
            frames.append(frame)
            ret_flags.append(ret)
        if not all(ret_flags):
            print("One of the streams ended or cannot read frames anymore.")
            break
        if frame_count >= max_frames:
            break
        
        results = []
        for gen in tracking_generators:
            try:
                result = next(gen)
            except StopIteration:
                result = None
            results.append(result)
        
        real_positions_all = [dict() for _ in range(num_cameras)]
        weights_all = [dict() for _ in range(num_cameras)]
        embeddings_all = [dict() for _ in range(num_cameras)]
        
        for i in range(num_cameras):
            if results[i] is None:
                continue
            bboxes = []
            track_ids = []
            weight_dict = {}
            xyxys = []
            if hasattr(results[i], 'boxes') and results[i].boxes is not None:
                for box in results[i].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    xyxys.append([x1, y1, x2, y2])
                    tid = int(box.id[0]) if box.id is not None else -1
                    track_ids.append(tid)
                    weight_dict[tid] = box.conf[0].item() * np.log((x2 - x1) * (y2 - y1))
                    bboxes.append([x1, y1, x2, y2])
            if len(xyxys) > 0:
                emb = get_features(np.array(xyxys), results[i].orig_img)
            else:
                emb = np.array([])
            for idx, (bb, tid) in enumerate(zip(bboxes, track_ids)):
                x1, y1, x2, y2 = map(int, bb)
                if not cams[i].is_footpoint_visable(x1, y1, x2, y2):
                    continue
                color = (0, 255, 0)
                if cams[i].is_mannequin(x1, y1, x2, y2):
                    color = (255, 0, 0)
                cv2.rectangle(frames[i], (x1, y1), (x2, y2), color, 2)
                cv2.putText(frames[i], f"ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if not cams[i].is_mannequin(x1, y1, x2, y2):
                    foot_x, foot_y = cams[i].get_footpoint(x1, y1, x2, y2)
                    rx, ry = cams[i].transform_image_to_real(foot_x, foot_y)
                    cv2.rectangle(frames[i], (foot_x-1, foot_y-1), (foot_x+1, foot_y+1), (255, 0, 0), 2)
                    real_positions_all[i][tid] = (rx, ry)
                    embeddings_all[i][tid] = emb[idx, :] if emb.size else None
            weights_all[i] = weight_dict
        
        all_detections = []
        for i in range(num_cameras):
            for tid, pos in real_positions_all[i].items():
                if embeddings_all[i].get(tid) is None:
                    continue
                all_detections.append({
                    "cam": i,
                    "tid": tid,
                    "rx": pos[0],
                    "ry": pos[1],
                    "weight": weights_all[i].get(tid, 1),
                    "embedding": embeddings_all[i][tid]
                })
        
        clusters = []
        for det in all_detections:
            clusters.append({
                "pos": (det["rx"], det["ry"]),
                "weight": det["weight"],
                "embedding": det["embedding"],
                "labels": [f"C{det['cam']+1}:{det['tid']}"],
                "cam_set": {det["cam"]}
            })
        
        dist_match_threshold = 0.4 
        same_camera_dist_match_threshold = 0.1
        cosine_threshold = 0.8 
        
        merged = True
        while merged:
            merged = False
            best_pair = None
            best_score = float('inf')
            history_pairs = set()
            for frame_matches in match_history:
                history_pairs.update(frame_matches)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    d = distance_2d(clusters[i]["pos"], clusters[j]["pos"])
                    if clusters[i]["cam_set"].intersection(clusters[j]["cam_set"]) and d > same_camera_dist_match_threshold:
                        continue
                    if d > dist_match_threshold:
                        continue
                    emb_distance = cosine(clusters[i]["embedding"], clusters[j]["embedding"])
                    if emb_distance > cosine_threshold:
                        continue
                    score = emb_distance + 0.2 * d
                    stable = False
                    for lab1 in clusters[i]["labels"]:
                        for lab2 in clusters[j]["labels"]:
                            pair = tuple(sorted([lab1, lab2]))
                            if pair in history_pairs:
                                stable = True
                                break
                        if stable:
                            break
                    if stable:
                        score /= 1.5
                    if score < best_score:
                        best_score = score
                        best_pair = (i, j)
            if best_pair is not None:
                i, j = best_pair
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                print(cluster1['labels'], cluster2['labels'],
                      cosine(cluster1["embedding"], cluster2["embedding"]),
                      distance_2d(cluster1["pos"], cluster2["pos"]))
                total_weight = cluster1["weight"] + cluster2["weight"]
                new_rx = (cluster1["pos"][0] * cluster1["weight"] + cluster2["pos"][0] * cluster2["weight"]) / total_weight
                new_ry = (cluster1["pos"][1] * cluster1["weight"] + cluster2["pos"][1] * cluster2["weight"]) / total_weight
                new_emb = (cluster1["embedding"] * cluster1["weight"] + cluster2["embedding"] * cluster2["weight"]) / total_weight
                new_labels = cluster1["labels"] + cluster2["labels"]
                new_cam_set = cluster1["cam_set"].union(cluster2["cam_set"])
                new_cluster = {
                    "pos": (new_rx, new_ry),
                    "weight": total_weight,
                    "embedding": new_emb,
                    "labels": new_labels,
                    "cam_set": new_cam_set
                }
                clusters.pop(j)
                clusters.pop(i)
                clusters.append(new_cluster)
                merged = True
        
        current_matches = set()
        for cluster in clusters:
            if len(cluster["labels"]) > 1:
                for pair in combinations(sorted(cluster["labels"]), 2):
                    current_matches.add(pair)
        match_history.append(current_matches)
        if len(match_history) > 5:
            match_history.pop(0)
        
        for track in kalman_manager.tracks:
            track.updated = False
        
        kalman_manager.update_tracks(clusters)
        topdown_canvas = np.ones((topdown_height, topdown_width, 3), dtype=np.uint8) * 255
        draw_grid(topdown_canvas)
        
        for cluster in clusters:
            cx, cy = real_to_canvas(cluster["pos"][0], cluster["pos"][1])
            if len(cluster["labels"]) > 1:
                color = (0, 0, 255)
            else:
                cam_id = int(cluster["labels"][0].split(":")[0][1:])
                color = (255, 0, 0) if cam_id == 1 else (0, 128, 0)
            cv2.circle(topdown_canvas, (cx, cy), 8, color, -1)
            label_text = " | ".join(cluster["labels"])
            cv2.putText(topdown_canvas, label_text, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Frame {frame_count} | Cluster at ({cluster['pos'][0]:.2f}, {cluster['pos'][1]:.2f}) with labels: {label_text}")
        
        for track in kalman_manager.tracks:
            if hasattr(track, "updated") and track.updated:
                pos = track.get_state()
                cx, cy = real_to_canvas(pos[0], pos[1])
                cv2.circle(topdown_canvas, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(topdown_canvas, f"T{track.track_id}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for i in range(num_cameras):
            out_cams[i].write(frames[i])
        out_topdown.write(topdown_canvas)
        
        frame_count += 1

    for cap in captures:
        cap.release()
    for writer in out_cams:
        writer.release()
    out_topdown.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
