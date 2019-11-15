# -*- coding:utf-8 -*
"""
@version: ??
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: detector.py
@time: 2019-11-15 上午9:48
"""
import os
import tensorflow as tf
import cv2
import numpy as np


class FaceDetector(object):
    def __init__(self, model_path):
        assert os.path.exists(model_path)
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with session.as_default():
            with graph.as_default():
                output_graph_def = tf.GraphDef()
                with open(model_path, "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(output_graph_def, name="")
                session.run(tf.global_variables_initializer())

                inputs = session.graph.get_tensor_by_name('inputs_image:0')
                shapes = session.graph.get_tensor_by_name('inputs_shape:0')
                outputs = session.graph.get_tensor_by_name('outputs:0')

        self.session = session
        self.inputs = inputs
        self.shape = shapes
        self.outputs = outputs
        self.pre_bnds = None

    @staticmethod
    def calculate_iou(a, b):
        """
        :param a: [N, 4]
        :param b: [M, 4]
        :return: [N, M]
        """
        a = np.array(a)
        b = np.array(b)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        return intersection / ua

    def smooth_boxes(self, pre_bnd, cur_bnd):
        num_cur = len(cur_bnd)
        num_pre = len(pre_bnd)
        if num_cur == 0 or num_pre == 0:
            return cur_bnd
        ious = self.calculate_iou(cur_bnd, pre_bnd)  # [C, P]
        loop_count = min(num_cur, num_pre)
        for _ in range(loop_count):
            idx = np.argmax(ious)
            idx_y = int(np.floor(idx * 1.0 / num_pre))
            idx_x = int(idx % num_pre)
            # print(idx_y, idx_x)
            best_iou = ious[idx_y, idx_x]
            cx0, cy0, cx1, cy1 = cur_bnd[idx_y]
            px0, py0, px1, py1 = pre_bnd[idx_x]
            dx0, dy0, dx1, dy1 = cx0 - px0, cy0 - py0, cx1 - px1, cy1 - py1
            scale = (1 - best_iou) ** 0.1
            x0 = px0 + scale * dx0
            y0 = py0 + scale * dy0
            x1 = px1 + scale * dx1
            y1 = py1 + scale * dy1
            cur_bnd[idx_y] = [x0, y0, x1, y1]
            ious[idx_y, :] = -1
            ious[:, idx_x] = -1
        return cur_bnd

    def predict(self, img, image_size=256, smooth=False):
        img_h, img_w, _ = img.shape
        pad = [0, 0]
        if img_h > img_w:
            scale = img_h * 1.0 / image_size
            img_h = image_size
            img_w = int(img_w * 1.0 / scale)
            pad[1] = (image_size - img_w) // 2
            pass
        else:
            scale = img_w * 1.0 / image_size
            img_w = image_size
            img_h = int(img_h * 1.0 / scale)
            pad[0] = (image_size - img_h) // 2
            pass
        tmp_image = cv2.resize(img, (img_w, img_h))
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
        bk_image = np.zeros([image_size, image_size, 3], dtype=np.uint8)
        bk_image[pad[0]:pad[0]+img_h, pad[1]:pad[1]+img_w] = tmp_image

        with self.session.as_default():
            with self.session.graph.as_default():
                detection = self.session.run(self.outputs, feed_dict={self.inputs: [bk_image, ],
                                                                      self.shape: [image_size, ]*2})
        res = []
        for det in detection:
            box = det[:4]
            box -= [pad[1], pad[0], pad[1], pad[0]]
            box *= scale
            x0, y0, x1, y1 = box.astype(np.int)
            cls_idx = int(det[4])
            score = det[5]
            if score < 0.5:
                continue
            res.append([x0, y0, x1, y1])
        if smooth:
            if self.pre_bnds is not None:
                pre_bnd = self.pre_bnds[:]
                cur_bnd = res[:]
                res = self.smooth_boxes(pre_bnd, cur_bnd)
            self.pre_bnds = res[:]
        return res
