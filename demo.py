# -*- coding:utf-8 -*
"""
@version: ??
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: demo.py
@time: 2019-11-15 上午9:48
"""
from detector import FaceDetector
from keypoints import FaceKps
import cv2
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    detector = FaceDetector("./model/det.pb")
    keypoints = FaceKps("./model/kps.pb")
    img = cv2.imread("demo.jpg")
    bnds = detector.predict(img, 1024)
    for bnd in bnds:
        bnd = [int(x) for x in bnd]
        x0, y0, x1, y1 = bnd
        pts = keypoints.predict(img, bnd)
        img = keypoints.vis(img, pts)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
    cv2.imwrite("res.jpg", img)

