import ast
import os
import gc
import glob
import time
import random
import torch.nn.functional as F
import imageio
import logging
from functools import wraps
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as torch_utils
from .postprocess import SegDetectorRepresenter
import torch.nn as nn
from torch.autograd import Variable
import collections
from torchvision.ops import roi_pool
device = 'cuda'
# TODO 给定图像和点，返回指定大小的图像
def crop_and_resize(image, points, target_size=(128, 32),type=0):
    TPS = cv2.createThinPlateSplineShapeTransformer()
    image = image.permute(1,2,0).cpu().numpy()
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
    # 数据集里标注为‘c’和'm'的需要tps变换，‘h'的不需要，若都变换，则注释掉
    TPS.estimateTransformation(targetShape, sourceShape, matches)
    new_image = TPS.warpImage(image)
    new_image = cv2.resize(new_image, target_size)
    if type:
        new_image = np.expand_dims(new_image, axis=-1)
    new_image = torch.from_numpy(new_image).permute(2,0,1).to('cuda')
    return new_image






def setup_determinism(seed=42):
    """
    https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(logger_name='dbtext', log_file_path=None):
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')

    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)

    return logger


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(">>> Function {}: {}'s".format(func.__name__, end - start))
        return result

    return wrapper


def to_device(batch, device='cpu'):
    new_batch = []

    for ele in batch:
        if isinstance(ele, torch.Tensor):
            new_batch.append(ele.to(device))
        else:
            new_batch.append(ele)
    return new_batch


def dict_to_device(batch, device='cpu'):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def to_list_tuples_coords(anns):
    new_anns = []
    data = []
    for i in range(len(anns[0])):
        temp = []
        for j in range(len(anns[0][i])):
            temp.append(tuple(anns[0][i][j]))
        data.append(temp)
        pass
    pass
    # points = anns[0][0]
    # new_array = [tuple(point) for point in points]
    # for ann in anns:
    #     points = []
    #     for x, y in ann:
    #         points.append((x[0].tolist(), y[0].tolist()))
    #     new_anns.append(points)
    return data



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def str_to_bool(value):
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{} is not a valid boolean value'.format(value))

#对图像进行最大最小归一化,并设置为numpy形式
def minmax_scaler_img(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        'uint8')  # noqa
    return img


def visualize_tfb(tfb_writer,
                  imgs,
                  preds,
                  global_steps,
                  thresh=0.5,
                  mode="TRAIN"):
    # origin img
    # imgs.shape = (batch_size, 3, image_size, image_size)
    imgs = torch.stack([
        torch.Tensor(
            minmax_scaler_img(img_.to('cpu').numpy().transpose((1, 2, 0))))
        for img_ in imgs
    ])
    imgs = torch.Tensor(imgs.numpy().transpose((0, 3, 1, 2)))
    imgs_grid = torch_utils.make_grid(imgs)
    imgs_grid = torch.unsqueeze(imgs_grid, 0)
    # imgs_grid.shape = (3, image_size, image_size * batch_size)
    tfb_writer.add_images('{}/origin_imgs'.format(mode), imgs_grid,
                          global_steps)

    # pred_prob_map / pred_thresh_map
    pred_prob_map = preds[:, 0, :, :]
    pred_thred_map = preds[:, 1, :, :]
    pred_prob_map[pred_prob_map <= thresh] = 0
    pred_prob_map[pred_prob_map > thresh] = 1

    # make grid
    pred_prob_map = pred_prob_map.unsqueeze(1)
    pred_thred_map = pred_thred_map.unsqueeze(1)

    probs_grid = torch_utils.make_grid(pred_prob_map, padding=0)
    probs_grid = torch.unsqueeze(probs_grid, 0)
    probs_grid = probs_grid.detach().to('cpu')

    thres_grid = torch_utils.make_grid(pred_thred_map, padding=0)
    thres_grid = torch.unsqueeze(thres_grid, 0)
    thres_grid = thres_grid.detach().to('cpu')

    tfb_writer.add_images('{}/prob_imgs'.format(mode), probs_grid,
                          global_steps)
    tfb_writer.add_images('{}/thres_imgs'.format(mode), thres_grid,
                          global_steps)


def test_resize(img, size=640, pad=False):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)

    new_img = None
    if pad:
        new_img = np.zeros((size, size, c), img.dtype)
        new_img[:h, :w] = cv2.resize(img, (w, h))
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img


def show_image(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    # 确保图像数据类型为 uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # 如果图像是 RGB 格式，转换为 BGR 格式
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def read_img(img_fp):
    img = cv2.imread(img_fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_origin, w_origin, _ = img.shape
    return img, h_origin, w_origin
def test_preprocess(img,
                    mean=[103.939, 116.779, 123.68],
                    to_tensor=True,
                    pad=False):
    img = test_resize(img, size=640, pad=pad)
    img = img.astype(np.float32)
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = np.expand_dims(img, axis=0)
    if to_tensor:
        img = torch.Tensor(img.transpose(0, 3, 1, 2))
    return img
def show(image, type=2):
    # 输入的是tensor类型的向量,C,H,W
    # 转换为numpy数组
    array1 = image.numpy()
    # 归一化，将图像数据扩展到[0, 255]
    rec_mean = rec_std = torch.tensor(0.5).numpy()
    array1 = (array1 * rec_std + rec_mean) * 255.0
    # 转换为uint8类型
    mat = np.uint8(array1)
    # 转置矩阵
    mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814, 3)
    # 使用matplotlib显示图像
    plt.imshow(mat,cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()
def draw_bbox(img, result, color=(255, 0, 0), thickness=3):
    """
    :input: RGB img
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img

# TODO 从预测图中截取图像
def intercept(args,origin_img,preds,batch):
    seg_obj = SegDetectorRepresenter(thresh=args.thresh,
                                     box_thresh=args.box_thresh,
                                     unclip_ratio=args.unclip_ratio)
    box_list, score_list = seg_obj(batch,
                                   preds,
                                   is_output_polygon=args.is_output_polygon)
def visualize_heatmap(args, img_fn, tmp_img, tmp_pred):
    pred_prob = tmp_pred[0]
    pred_prob[pred_prob <= args.prob_thred] = 0
    pred_prob[pred_prob > args.prob_thred] = 1

    np_img = minmax_scaler_img(tmp_img[0].to(device).numpy().transpose(
        (1, 2, 0)))
    plt.imshow(np_img)
    plt.imshow(pred_prob, cmap='jet', alpha=args.alpha)
    img_fn = "heatmap_result_{}".format(img_fn)
    plt.savefig(os.path.join(args.save_dir, img_fn),
                dpi=200,
                bbox_inches='tight')
    gc.collect()

def visualize_polygon(args, img_fn, origin_info, batch, preds, vis_char=False):
    img_origin, h_origin, w_origin = origin_info
    seg_obj = SegDetectorRepresenter(thresh=args.thresh,
                                     box_thresh=args.box_thresh,
                                     unclip_ratio=args.unclip_ratio)
    box_list, score_list = seg_obj(batch,
                                   preds,
                                   is_output_polygon=args.is_output_polygon)
    box_list, score_list = box_list[0], score_list[0]
    if len(box_list) > 0:
        if args.is_output_polygon:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
    else:
        box_list, score_list = [], []
    for i,_  in enumerate(box_list):
        data = crop_and_resize(img_origin,np.array(box_list[i]))
    # tmp_img = draw_bbox(img_origin, np.array(box_list))
    tmp_img = img_origin
    tmp_pred = cv2.resize(preds[0, 0, :, :].cpu().numpy(),
                          (w_origin, h_origin))
    # https://stackoverflow.com/questions/42262198
    h_, w_ = 32, 100
    if not args.is_output_polygon and vis_char:

        char_img_fps = glob.glob(os.path.join("./tmp/reconized", "*"))
        for char_img_fp in char_img_fps:
            os.remove(char_img_fp)

        for index, (box_list_,
                    score_list_) in enumerate(zip(box_list,
                                                  score_list)):  # noqa
            src_pts = np.array(box_list_.tolist(), dtype=np.float32)
            dst_pts = np.array([[0, 0], [w_, 0], [w_, h_], [0, h_]],
                               dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp = cv2.warpPerspective(img_origin, M, (w_, h_))
            imageio.imwrite("./tmp/reconized/word_{}.jpg".format(index), warp)

    plt.imshow(tmp_img)
    plt.imshow(tmp_pred, cmap='inferno', alpha=args.alpha)
    if args.is_output_polygon:
        img_fn = "poly_result_{}".format(img_fn)
    else:
        img_fn = "rect_result_{}".format(img_fn)
    plt.savefig(os.path.join(args.save_dir, img_fn),
                dpi=200,
                bbox_inches='tight')
    gc.collect()

#!/usr/bin/python
# encoding: utf-8

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def encode_batch(self, texts):
        # Find the maximum length of the sequences in the batch
        max_length = max(len(text) for text in texts)
        batch_size = len(texts)
        # Initialize the target tensor with zeros (0 is reserved for 'blank' required by CTC)
        target = torch.zeros(batch_size, max_length, dtype=torch.int32)
        _, length = self.encode(texts)
        # Encode each text and place it in the target tensor
        for i, text in enumerate(texts):
            encoded_text, _ = self.encode(text)
            target[i, :len(encoded_text)] = encoded_text
        return target, length

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

# TODO CRNN部分

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def encode_batch(self, texts):
        # Find the maximum length of the sequences in the batch
        max_length = 50
        batch_size = len(texts)
        # Initialize the target tensor with zeros (0 is reserved for 'blank' required by CTC)
        target = torch.zeros(batch_size, max_length, dtype=torch.int32)
        _, length = self.encode(texts)
        # Encode each text and place it in the target tensor
        for i, text in enumerate(texts):
            encoded_text, _ = self.encode(text)
            target[i, :len(encoded_text)] = encoded_text
        return target, length

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.resize_(data.size()).copy_(data)
    # with torch.no_grad():
        #v.data.resize_(data.size()).copy_(data)
def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))
def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img