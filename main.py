"""
[x] membaca video
[x] frame
[x] face object detection
[x] face landmark detection 5 titik
[x] face landmark detection 68 titik
[x] topeng -> landmark 5 titik dan png topeng
[x] menampilkan processing
[x] menulis video
"""
import argparse
import os

import cv2
import dlib
import numpy as np
from tqdm import tqdm

from retinaface import RetinaFace

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# topeng
topeng = cv2.imread("topeng.png", cv2.IMREAD_UNCHANGED)
m_re = (128, 200)
m_le = (313, 200)
m_mr = (141, 380)
m_ml = (300, 380)
pts1 = np.float32([m_re, m_le, m_mr, m_ml])


def main(
        video_path: str,
        with_mask: bool,
        with_bbox: bool,
        with_landmark_5: bool,
        with_landmark_68: bool) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("unable to read video")
    _dir, _fname = os.path.split(video_path)
    out_dir = _dir + "_out"
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, os.path.splitext(_fname)[0] + ".mp4")
    sink = cv2.VideoWriter(
        out_p, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=25,
        frameSize=(1920, 1080))
    if not sink.isOpened():
        raise IOError("cannot create sink video file")
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    pbar = tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        unit="frame")
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        frame = process(
            frame,
            with_mask=with_mask,
            with_bbox=with_bbox,
            with_landmark_5=with_landmark_5,
            with_landmark_68=with_landmark_68)
        sink.write(frame)
        cv2.imshow("preview", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        pbar.update()
    cap.release()
    sink.release()
    pbar.close()


def process(
        frame: np.ndarray,
        with_mask: bool,
        with_bbox: bool,
        with_landmark_5: bool,
        with_landmark_68: bool) -> np.ndarray:
    result = RetinaFace.detect_faces(frame)
    k, v = list(result.items())[0]
    # topeng
    if with_mask:
        pts2 = np.float32([
            v["landmarks"]["right_eye"],
            v["landmarks"]["left_eye"],
            v["landmarks"]["mouth_right"],
            v["landmarks"]["mouth_left"]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        topeng_warped = cv2.warpPerspective(
            topeng, M, (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_CONSTANT)

        # compositing
        # Extract the RGB channels
        srcRGB = topeng_warped[..., :3]
        dstRGB = frame

        # Extract the alpha channels and normalise to range 0..1
        srcA = topeng_warped[..., 3]/255.0
        dstA = np.ones(frame.shape[:2], dtype=np.float)

        # Work out resultant alpha channel
        outA = srcA + dstA*(1-srcA)

        # Work out resultant RGB
        outRGB = (srcRGB*srcA[..., np.newaxis] + dstRGB*dstA[...,
                                                             np.newaxis]*(1-srcA[..., np.newaxis])) / outA[..., np.newaxis]

        # Merge RGB and alpha (scaled back up to 0..255) back into single image
        frame = np.dstack((outRGB, outA*255)).astype(np.uint8)
        # RGBA to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # show bbox
    if with_bbox:
        cv2.rectangle(
            frame, v["facial_area"][:2],
            v["facial_area"][2:], color=(0, 0, 255),
            thickness=5)
        cv2.putText(
            frame,
            k.upper() + ": {}%".format(int(v["score"] * 100)),
            (v["facial_area"][0], v["facial_area"][3] + 50),
            cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
    # show landmark 5
    if with_landmark_5:
        for lk, lv in v["landmarks"].items():
            cv2.circle(
                frame,
                (int(lv[0]), int(lv[1])),
                5, (0, 255, 0),
                thickness=-1)
    # dlib
    if with_landmark_68:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for k, v in result.items():
            rect = dlib.rectangle(*v["facial_area"])
            shape = predictor(frame_rgb, rect)
            parts = shape.parts()
            points_68 = list(map(lambda pt: (pt.x, pt.y), list(parts)))
            for pt in points_68:
                cv2.circle(
                    frame,
                    pt,
                    3, (0, 255, 0),
                    thickness=-1)
    return frame


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--mask",
        action="store_true",
        help="Show mask")
    argparser.add_argument(
        "--bbox",
        action="store_true",
        help="Show bbox")
    argparser.add_argument(
        "--landmark_5",
        action="store_true",
        help="Show 5 landmarks")
    argparser.add_argument(
        "--landmark_68",
        action="store_true",
        help="Show 68 landmarks")
    argparser.add_argument(
        "video_path", help="Path to the video file")
    args = argparser.parse_args()
    main(
        args.video_path,
        args.mask,
        args.bbox,
        args.landmark_5,
        args.landmark_68)
