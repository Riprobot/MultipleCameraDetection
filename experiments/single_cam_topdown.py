import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

from camera_calib import Camera

def main():
    cam = Camera("data/calibration1.py")

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
            _, _ = real_to_canvas(x, y_max)
            cv2.line(canvas, (cx, 0), (cx, topdown_height), (200, 200, 200), 1)
            cv2.putText(canvas, f"{x:.1f}", (cx + 5, topdown_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

        for y in np.arange(y_min, y_max + grid_spacing_m, grid_spacing_m):
            _, cy = real_to_canvas(x_min, y)
            cv2.line(canvas, (0, cy), (topdown_width, cy), (200, 200, 200), 1)
            cv2.putText(canvas, f"{y:.1f}", (5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    model = YOLO("yolo11x.pt") 


    video_path = "data/cam1.mp4"
    tracker_config = "my_botsort.yaml" 

    results_generator = model.track(
        source=video_path,
        stream=True,          
        tracker=tracker_config,
        conf=0.5,   
        iou=0.75,
        imgsz = (1280, 720),
        classes=[0],          
        batch = 8,
        augment = True,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open", video_path)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * 120) 
    frame_count = 0

    out_vid_original = cv2.VideoWriter(
        "output_cam3_annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    out_vid_topdown = cv2.VideoWriter(
        "output_topdown.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (topdown_width, topdown_height)
    )

    static_counters = defaultdict(int)
    MOTION_MEAN_THRESHOLD = 20
    STATIC_FRAMES_THRESHOLD = 10
    not_moving_track_ids = set()
    prev_frame_gray = None

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            result = next(results_generator)
        except StopIteration:
            break
        
        if not hasattr(result, 'boxes') or result.boxes is None:
            topdown_canvas = np.ones((topdown_height, topdown_width, 3), dtype=np.uint8) * 255
            draw_grid(topdown_canvas)
            out_vid_original.write(frame)
            out_vid_topdown.write(topdown_canvas)
            frame_count += 1
            continue

        
        bboxes = []
        track_ids = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id[0]) if box.id is not None else -1  # -1 if no ID
            bboxes.append([x1, y1, x2, y2])
            track_ids.append(track_id)

        if frame_count % 10 == 0:
            if prev_frame_gray is None:
                prev_frame_gray = frame_gray.copy()
            else:
                for bb, tid in zip(bboxes, track_ids):
                    x1, y1, x2, y2 = map(int, bb)
                    
                    x1 = max(0, min(x1, frame_width-1))
                    x2 = max(0, min(x2, frame_width-1))
                    y1 = max(0, min(y1, frame_height-1))
                    y2 = max(0, min(y2, frame_height-1))

                    if x2 - x1 <= 1 or y2 - y1 <= 1:
                        continue

                    roi_height = y2 - y1
                    head_end = y1 + roi_height // 3
                    head_cur = frame_gray[y1:head_end, x1:x2]
                    head_prev = prev_frame_gray[y1:head_end, x1:x2]

                    head_diff_mean = 0
                    if head_cur.size > 0 and head_prev.size > 0:
                        head_diff = cv2.absdiff(head_cur, head_prev)
                        head_diff_mean = np.quantile(head_diff, 0.99)

                    if head_diff_mean < MOTION_MEAN_THRESHOLD:
                        static_counters[tid] += 1
                    else:
                        static_counters[tid] = 0
                        if tid in not_moving_track_ids:
                            not_moving_track_ids.remove(tid)

                    if static_counters[tid] >= STATIC_FRAMES_THRESHOLD:
                        not_moving_track_ids.add(tid)

                prev_frame_gray = frame_gray.copy()

        topdown_canvas = np.ones((topdown_height, topdown_width, 3), dtype=np.uint8) * 255
        draw_grid(topdown_canvas)

        for bb, tid in zip(bboxes, track_ids):
            x1, y1, x2, y2 = map(int, bb)

            is_mannequin_cam = cam.is_mannequin(x1, y1, x2, y2)
            color = (255, 0, 0) if is_mannequin_cam else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            foot_x = (x1 + x2) // 2
            foot_y = y2
            real_x, real_y = cam.transform_image_to_real(foot_x, foot_y)
            cx, cy = real_to_canvas(real_x, real_y)

            if 0 <= cx < topdown_width and 0 <= cy < topdown_height:
                if not is_mannequin_cam:
                    cv2.circle(topdown_canvas, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(topdown_canvas, str(tid), (cx+10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_vid_original.write(frame)
        out_vid_topdown.write(topdown_canvas)
        frame_count += 1

    cap.release()
    out_vid_original.release()
    out_vid_topdown.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
