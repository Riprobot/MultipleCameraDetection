import cv2
import numpy as np

def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y2 - y1)**2)**0.5

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

class Camera:
    def __init__(self, calibration_file):
        scope = {}
        with open(calibration_file, 'r') as f:
            code = compile(f.read(), calibration_file, 'exec')
            exec(code, scope)

        if 'imgpoints' not in scope or 'objpoints' not in scope:
            raise ValueError(f"Need imgpoints and objpoints in {calibration_file}")
        self.mannequin_footpoints = scope.get('mannequin_footpoints', [])

        imgp = np.array(scope['imgpoints'], dtype=np.float32).reshape(-1,1,2)
        objp = np.array(scope['objpoints'], dtype=np.float32).reshape(-1,1,3)

        img_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            [objp], [imgp], img_size, None, None
        )

        self.rvec = rvecs[0]
        self.tvec = tvecs[0]
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.C = -self.R.T.dot(self.tvec.reshape(3,1))

        self.newcam_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, img_size, 1, img_size
        )

        undist = cv2.undistortPoints(imgp, self.mtx, self.dist, None, self.newcam_mtx)
        src = undist.reshape(-1,2)
        # print(src)
        dst = np.array([p[:2] for p in scope['objpoints']], dtype=np.float32)
        self.H_ground, _ = cv2.findHomography(src, dst, cv2.RANSAC)

    def get_footpoint(self, x1, y1, x2, y2):
        mid_x = (x1 + x2) // 2
        p = mid_x / CAMERA_WIDTH
        foot_x = int(x1 * p + x2 * (1 - p))
        coef = (1/6) * (y2 / CAMERA_HEIGHT)**2
        foot_y = int(y2 - (y2 - y1) * coef)
        return (foot_x, foot_y)

    def transform_via_homography(self, img_x, img_y):
        pt = np.array([[[img_x, img_y]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, self.mtx, self.dist, None, self.newcam_mtx)
        real = cv2.perspectiveTransform(und, self.H_ground)
        return float(real[0,0,0]), float(real[0,0,1])

    def transform_image_to_world(self, img_x, img_y, z=0.0):
        pt = np.array([[[img_x, img_y]]], dtype=np.float32)
        und = cv2.undistortPoints(pt, self.mtx, self.dist, None, None)
        x_norm, y_norm = und[0,0]
        v_cam = np.array([[x_norm], [y_norm], [1.0]])
        # print(v_cam)
        dir_world = self.R.T.dot(v_cam)
        # print(dir_world)
        C = self.C
        # print(C)
        lam = (z - float(C[2])) / float(dir_world[2])
        P = C + lam * dir_world
        return float(P[0]), float(P[1])

    transform_image_to_real = transform_via_homography

    def is_mannequin(self, x1, y1, x2, y2):
        fx, fy = (x1 + x2) / 2.0, y2
        p1 = self.transform_image_to_real(fx, fy)
        d_real = np.inf
        d_img  = np.inf
        for ip in self.mannequin_footpoints:
            p2 = self.transform_image_to_real(*ip)
            d_real = min(d_real, distance(p1[0], p1[1], p2[0], p2[1]))
            d_img  = min(d_img, distance(fx, fy, ip[0], ip[1]))
        return (d_real <= 0.1) or (d_img <= 10)

    def is_footpoint_visible(self, x1, y1, x2, y2):
        return (CAMERA_HEIGHT - y2) > 11
