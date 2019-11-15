# -*- coding:utf-8 -*
"""
@version: ??
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: keypoints.py
@time: 2019-11-15 上午9:48
"""
import os
import tensorflow as tf
import cv2

class FaceKps(object):
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
                outputs = session.graph.get_tensor_by_name('predict:0')

        self.session = session
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, img, bnd, image_size=112):
        img_h, img_w, _ = img.shape
        x0, y0, x1, y1 = bnd
        c_x = int((x0 + x1)/2)
        c_y = int((y0 + y1)/2)
        scale = 1.2
        w_ = int(x1 - x0) * scale
        h_ = int(y1 - y0) * scale
        s_ = max(w_, h_)
        w_, h_ = s_, s_
        w = w_ * 0.5
        h = h_ * 0.5

        x0 = int(min(max(c_x - w, 0), img_w-1))
        x1 = int(min(max(c_x + w, 0), img_w-1))
        y0 = int(min(max(c_y - h, 0), img_h-1))
        y1 = int(min(max(c_y + h, 0), img_h-1))

        sub_image = img[y0:y1+1, x0:x1+1]
        sub_image = cv2.resize(sub_image, (image_size, image_size))
        # sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
        diff_x = x0
        diff_y = y0
        scale_x = (x1 - x0)/image_size
        scale_y = (y1 - y0)/image_size

        # print(diff_x, scale_x)
        with self.session.as_default():
            with self.session.graph.as_default():
                heats = self.session.run(self.outputs, feed_dict={self.inputs: [sub_image, ]})
        heats = heats[0]
        pts = []
        for i in range(0, len(heats), 2):
            x = int((heats[i] * 0.01 + 0.5) * image_size)
            y = int((heats[i+1] * 0.01 + 0.5) * image_size)

            x = int(x * scale_x + diff_x)
            y = int(y * scale_y + diff_y)
            pts.append([x, y])
        return pts

    def vis(self, img, pts):
        connectionts = [[0, 32],
                        [33, 41], [41, 33],
                        [42, 50], [50, 42],
                        [51, 54],
                        [55, 59],
                        [60, 67], [67, 60],
                        [68, 75], [75, 68],
                        [76, 87], [87, 76],
                        [88, 95], [95, 88]]

        for conn in connectionts:
            s, e = conn
            if s <= e:
                for i in range(s+1, e + 1):
                    x0, y0 = pts[i-1]
                    x1, y1 = pts[i]
                    cv2.circle(img, (x0, y0), 2, (255, 0, 0), -1)
                    cv2.circle(img, (x1, y1), 2, (255, 0, 0), -1)
                    cv2.line(img, (x0,  y0), (x1, y1), (255, 0, 0), 1)
            else:
                x0, y0 = pts[s]
                x1, y1 = pts[e]
                cv2.circle(img, (x0, y0), 2, (255, 0, 0), -1)
                cv2.circle(img, (x1, y1), 2, (255, 0, 0), -1)
                cv2.line(img, (x0,  y0), (x1, y1), (255, 0, 0), 1)
        return img