import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
# from PIL import Image
import os

def tps_for_total(num):
    print('%d\n' %num)
    gt_path = '/root/autodl-fs/DB_CRNN/dataset/txt_format/Train/poly_gt_img1240.txt'
    img_path = '/root/autodl-fs/DB_CRNN/dataset/total_text/train/img1240.jpg'
    TPS = cv2.createThinPlateSplineShapeTransformer()
    #gt = open(gt_path, 'r')
    if os.path.exists(gt_path):
        gt = open(gt_path, 'r')
        image = cv2.imread(img_path)
        count = 1
        for ii, line in enumerate(gt):
            image_ = image.copy()
            items = line.split(':')
            # print(items)
            assert items.__len__() == 5
            # print(items)
            type_ = items[3][4:5]
            # print(type_)
            xs = items[1][3:-5].split()
            ys = items[2][3:-8].split()
            assert xs.__len__() == ys.__len__()
            n = xs.__len__()
            points = []
            for i in range(n):
                point = [int(xs[i]), int(ys[i])]
                points.append(point)
            rect = cv2.boundingRect(np.array(points))
            word_image = image_[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            w = image.shape[1]
            h = image.shape[0]
            point_number = int(points.__len__() / 2)
            length_unit = w / (point_number - 1)
            new_points = []
            for i in range(point_number):
                new_point = [length_unit * i, 0]
                new_points.append(new_point)
            for i in range(point_number):
                new_point = [w - (length_unit * i), h]
                new_points.append(new_point)
            matches = []
            for i in range(1, points.__len__()):
                matches.append(cv2.DMatch(i, i, 0))
            sourceShape = np.array(points, np.float32).reshape(1, -1, 2)
            targetShape = np.array(new_points, np.float32).reshape(1, -1, 2)

            if type_ != 'h' :
            #有m，c，h三类
                TPS.estimateTransformation(targetShape, sourceShape, matches)
                new_image = TPS.warpImage(image_)
            else:
                new_image=word_image
            # 数据集里标注为‘c’和'm'的需要tps变换，‘h'的不需要，若都变换，则注释掉
            new_image = cv2.resize(new_image, (100, 32))
            savepath = '/root/autodl-fs/DB_CRNN/test1.jpg'
            cv2.imwrite(savepath, new_image)
            count += 1
    else:
        print('\n')

def main():
    tps_for_total(1)
if __name__ == '__main__':
    main()

