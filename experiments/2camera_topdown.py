import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
from reid import get_features
from camera_calib import Camera
from scipy.spatial.distance import cosine
def distance_2d(p1, p2):
    """Euclidean distance in 2D plane."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def main():
    cam1 = Camera("data/calibration1.py")
    cam3 = Camera("data/calibration3.py")
    x_min, x_max = -10, 10
    y_min, y_max = -0.2, 3
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
            _, cy2 = real_to_canvas(x, y_max)
            cv2.line(canvas, (cx, 0), (cx, topdown_height), (200, 200, 200), 1)
            cv2.putText(canvas, f"{x:.1f}", (cx + 5, topdown_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        for y in np.arange(y_min, y_max + grid_spacing_m, grid_spacing_m):
            _, cy = real_to_canvas(x_min, y)
            cv2.line(canvas, (0, cy), (topdown_width, cy), (200, 200, 200), 1)
            cv2.putText(canvas, f"{y:.1f}", (5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    model1 = YOLO("yolo11x.pt")
    model2 = YOLO("yolo11x.pt")

    tracker_config = "botsort.yaml"

    results_gen_cam1 = model1.track(
        source="data/cam1.mp4",
        stream=True,
        tracker=tracker_config,
        conf=0.4,
        iou=0.75,
        imgsz=(1280, 720),
        classes=[0],
        batch=8,
        augment=True,
    )

    results_gen_cam3 = model2.track(
        source="data/cam3.mp4",
        stream=True,
        tracker=tracker_config,
        conf=0.4,
        iou=0.75,
        imgsz=(1280, 720),
        classes=[0],
        batch=8,
        augment=True,
    )
    cap1 = cv2.VideoCapture("data/cam1.mp4")
    cap3 = cv2.VideoCapture("data/cam3.mp4")

    if not cap1.isOpened():
        print("Error: Could not open cam1.mp4")
        return
    if not cap3.isOpened():
        print("Error: Could not open cam3.mp4")
        return

    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
    fps3 = cap3.get(cv2.CAP_PROP_FPS) or 30

    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width3 = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height3 = int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = int(fps1 * 20)

    out_cam1 = cv2.VideoWriter(
        "output_cam1_annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps1,
        (frame_width1, frame_height1)
    )
    out_cam3 = cv2.VideoWriter(
        "output_cam3_annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps3,
        (frame_width3, frame_height3)
    )
    out_topdown = cv2.VideoWriter(
        "output_topdown_combined.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps1, fps3),
        (topdown_width, topdown_height)
    )

    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret3, frame3 = cap3.read()
        if not ret1 or not ret3:
            print("One of the streams ended or cannot read frames anymore.")
            break
        if frame_count >= max_frames:
            break

        try:
            result1 = next(results_gen_cam1)
            result3 = next(results_gen_cam3)
        except StopIteration:
            print("No more tracking results from one of the generators.")
            break

        bboxes_cam1 = []
        track_ids_cam1 = []
        weight_cam1 = {}
        xyxys = []
        if hasattr(result1, 'boxes') and result1.boxes is not None:
            for box in result1.boxes:
                # if box.conf < 0.3:
                #     continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xyxys.append([x1, y1, x2, y2])
                track_id = int(box.id[0]) if box.id is not None else -1
                weight_cam1[track_id] = box.conf[0].item() * np.log((x2 - x1) * (y2 - y1))
                bboxes_cam1.append([x1, y1, x2, y2])
                track_ids_cam1.append(track_id)
        emb1 = get_features(np.array(xyxys), result1.orig_img)
        
        bboxes_cam3 = []
        track_ids_cam3 = []
        weight_cam3 = {}
        xyxys = []
        if hasattr(result3, 'boxes') and result3.boxes is not None:
            for box in result3.boxes:
                # if box.conf < 0.3:
                #     continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xyxys.append([x1, y1, x2, y2])
                track_id = int(box.id[0]) if box.id is not None else -1
                weight_cam3[track_id] = box.conf[0].item() * np.log((x2 - x1) * (y2 - y1))
                bboxes_cam3.append([x1, y1, x2, y2])
                track_ids_cam3.append(track_id)
        emb3 = get_features(np.array(xyxys), result3.orig_img)
        real_positions_cam1 = {} 
        
        for bb, tid in zip(bboxes_cam1, track_ids_cam1):
            x1, y1, x2, y2 = map(int, bb)
            if not cam1.is_footpoint_visable(x1, y1, x2, y2):
                continue
            color = (0, 255, 0)
            if cam1.is_mannequin(x1, y1, x2, y2):
                color = (255, 0, 0)
            cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame1, f"ID:{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            foot_x = (x1 + x2) // 2
            foot_y = y2
            if not cam1.is_mannequin(x1, y1, x2, y2):
                rx, ry = cam1.transform_image_to_real(foot_x, foot_y)
                real_positions_cam1[tid] = (rx, ry)

        real_positions_cam3 = {}
        for bb, tid in zip(bboxes_cam3, track_ids_cam3):
            x1, y1, x2, y2 = map(int, bb)
            if not cam3.is_footpoint_visable(x1, y1, x2, y2):
                continue
            color = (0, 255, 0)
            if cam3.is_mannequin(x1, y1, x2, y2):
                color = (255, 0, 0)
            cv2.rectangle(frame3, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame3, f"ID:{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            foot_x = (x1 + x2) // 2
            foot_y = y2
            if not cam3.is_mannequin(x1, y1, x2, y2):
                rx, ry = cam3.transform_image_to_real(foot_x, foot_y)
                real_positions_cam3[tid] = (rx, ry)

        topdown_canvas = np.ones((topdown_height, topdown_width, 3), dtype=np.uint8) * 255
        draw_grid(topdown_canvas)

        detected_cam1 = [(tid, pos[0], pos[1]) for tid, pos in real_positions_cam1.items()]
        detected_cam3 = [(tid, pos[0], pos[1]) for tid, pos in real_positions_cam3.items()]

        dist_match_threshold = 0.6  
        used_cam3 = set() 
        for (tid1, rx1, ry1) in detected_cam1:
            best_match3 = None
            best_dist = float('inf')
            for (tid3, rx3, ry3) in detected_cam3:
                if tid3 in used_cam3:
                    continue
                dist = distance_2d((rx1, ry1), (rx3, ry3))
                if dist > dist_match_threshold:
                    continue
                emb_dist = 0
                index1 = track_ids_cam1.index(tid1)
                index3 = track_ids_cam3.index(tid3)
                emb_dist = cosine(emb1[index1, :], emb3[index3, :])
                if emb_dist < best_dist:
                    best_dist = emb_dist
                    best_match3 = (tid3, rx3, ry3)
            if best_match3:
                used_cam3.add(best_match3[0])
                w1 = weight_cam1[tid1]
                w3 = weight_cam3[tid3]
                mx = (rx1 * w1 + best_match3[1] * w3) / (w1 + w3)
                my = (ry1 * w1 + best_match3[2] * w3) / (w1 + w3)
                # print(w1, w3)
                cx, cy = real_to_canvas(mx, my)
                color = (0, 0, 255)  # red
                cv2.circle(topdown_canvas, (cx, cy), 8, color, -1)
                cv2.putText(topdown_canvas, f"C1:{tid1}=C3:{best_match3[0]}",
                            (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"Frame {frame_count} | MATCH -> Cam1 ID {tid1} and Cam3 ID {best_match3[0]}, dist={best_dist:.2f}m")
            else:
                cx, cy = real_to_canvas(rx1, ry1)
                color = (255, 0, 0)
                cv2.circle(topdown_canvas, (cx, cy), 6, color, -1)
                cv2.putText(topdown_canvas, f"C1:{tid1}", (cx+10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # print(f"Frame {frame_count} | Cam1 ID {tid1} has no match")

        for (tid3, rx3, ry3) in detected_cam3:
            if tid3 in used_cam3:
                continue
            cx, cy = real_to_canvas(rx3, ry3)
            color = (0, 128, 0)
            cv2.circle(topdown_canvas, (cx, cy), 6, color, -1)
            cv2.putText(topdown_canvas, f"C3:{tid3}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # print(f"Frame {frame_count} | Cam3 ID {tid3} has no match")

        out_cam1.write(frame1)
        out_cam3.write(frame3)
        out_topdown.write(topdown_canvas)

        frame_count += 1

    cap1.release()
    cap3.release()
    out_cam1.release()
    out_cam3.release()
    out_topdown.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
