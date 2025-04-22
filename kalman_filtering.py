import numpy as np

class KalmanTracker:
    def __init__(self, initial_position, initial_ids, dt=1/30.0):
        self.dt = dt
        self.x = np.array([initial_position[0], initial_position[1], 0, 0], dtype=float)
        self.P = np.eye(4) * 100.0  
        self.Q = np.eye(4) * 0.1  
        self.R = np.eye(2) * 1e-6  
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.kalman_ids = set(initial_ids)
        self.track_id = None
        self.updated = True

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2]

    def update(self, measurement, new_ids=None):
        z = np.array(measurement, dtype=float)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        if new_ids is not None:
            self.kalman_ids.update(new_ids)
        self.updated = True

    def get_state(self):
        return self.x[:2]


class KalmanTrackerManager:
    def __init__(self, dt=1/30.0):
        self.tracks = []
        self.next_track_id = 0
        self.dt = dt

    def add_track(self, measurement, initial_ids):
        tracker = KalmanTracker(measurement, initial_ids, self.dt)
        tracker.track_id = self.next_track_id
        self.next_track_id += 1
        tracker.updated = True
        self.tracks.append(tracker)
        return tracker

    def predict_all(self):
        predictions = {}
        for tracker in self.tracks:
            predictions[tracker.track_id] = tracker.predict()
        return predictions

    def update_tracks(self, detections):
        for detection in detections:
            measurement = detection["pos"]
            detection_ids = set(detection["labels"])
            matched_track = None
            for tracker in self.tracks:
                if tracker.kalman_ids.intersection(detection_ids):
                    matched_track = tracker
                    break
            if matched_track:
                matched_track.update(measurement, detection_ids)
            else:
                self.add_track(measurement, detection_ids)

    def get_tracks(self):
        result = []
        for tracker in self.tracks:
            result.append({
                "track_id": tracker.track_id,
                "position": tracker.get_state(),
                "kalman_ids": tracker.kalman_ids,
                "updated": tracker.updated
            })
        return result
