# create data and label for RAF
# containing 12271 training samples and 3068 testing samples after aligned.


# 0: Surprise
# 1: Fear
# 2: Disgust
# 3: Happiness
# 4: Sadness
# 5: Anger
# 6: Neutral
# 7: Contempt

# 这里将FERPlus的标签与RAF标签统一
idx_to_emos = {
    0: 6,
    1: 3,
    2: 0,
    3: 4,
    4: 5,
    5: 2,
    6: 1,
    7: 7, # contempt
}

import csv
import os
import numpy as np
import pandas as pd
import h5py
import skimage.io
from skimage import transform
from PIL import Image
import sys
import cv2
import dlib
import transforms as transforms
import torch
import attention_seg_face_gt as gt
from models.ferattentionnet import AttentionResNet
import generate_mask_bylandmark as gmark

transform_test = transforms.Compose([
    # transforms.TenCrop(cut_size),
    # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Resize(128),
    transforms.ToTensor(),
])

def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    #print()
    return hull_mask


def get_landmarks(image):

    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    if len(rects) == 0:
        landmarks = np.matrix([[]])
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])
        # print(landmarks, type(landmarks))
        # for idx, point in enumerate(landmarks):
        #     # 68点的坐标
        #     pos = (point[0, 0], point[0, 1])
        #     print(idx + 1, pos)
    return landmarks


def get_mask(image):
    origin_shape = image.shape
    # if image.shape[0] < 80 or image.shape[1] < 80:
    #     image = cv2.resize(image, (80, 80))
    # print(image.shape)
    image = cv2.resize(image, (128, 128))
    landmarks = get_landmarks(image)
    if landmarks.shape[1] == 0:
        return image, False
    else:
        mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)

    # print(mask)
    return mask, True

def image255(image):

    w = image.shape[0]
    h = image.shape[1]

    for i in range(w):
        for j in range(h):
            if image[i, j] == 1:
                image[i, j] = 255
    return image

def tensorToStr(tensors):

    res = ''
    w = tensors.shape[0]
    h = tensors.shape[1]

    for i in range(w):
        for j in range(h):
            if i == w - 1 and j == h - 1:
                res = res + str(tensors[i, j] * 255)
            else:
                res = res + str(tensors[i, j] * 255) + ' '
    return res

def _process_data(emotion_raw, mode = 'majority'):
    '''
    Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:

    Majority: return the emotion that has the majority vote, or unknown if the count is too little.
    Probability or Crossentropty: convert the count into probability distribution.abs
    Multi-target: treat all emotion with 30% or more votes as equal.
    '''
    size = len(emotion_raw)
    emotion_unknown = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal)
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size

    if mode == 'majority':
        # find the peak value of the emo_raw list
        maxval = max(emotion_raw)
        if maxval > 0.5 * sum_list:
            emotion[np.argmax(emotion_raw)] = maxval
        else:
            emotion = emotion_unknown  # force setting as unknown
    elif (mode == 'probability') or (mode == 'crossentropy'):
        sum_part = 0
        count = 0
        valid_emotion = True
        while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
            maxval = max(emotion_raw)
            for i in range(size):
                if emotion_raw[i] == maxval:
                    emotion[i] = maxval
                    emotion_raw[i] = 0
                    sum_part += emotion[i]
                    count += 1
                    if i >= 8:  # unknown or non-face share same number of max votes
                        valid_emotion = False
                        if sum(emotion) > maxval:  # there have been other emotions ahead of unknown or non-face
                            emotion[i] = 0
                            count -= 1
                        break
        if sum(emotion) <= 0.5 * sum_list or count > 3:  # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
            emotion = emotion_unknown  # force setting as unknown
    elif mode == 'multi_target':
        threshold = 0.3
        for i in range(size):
            if emotion_raw[i] >= threshold * sum_list:
                emotion[i] = emotion_raw[i]
        if sum(emotion) <= 0.5 * sum_list:  # less than 50% of the votes are integrated, we discard this example
            emotion = emotion_unknown  # set as unknown

    return [float(i) / sum(emotion) for i in emotion]

def get_occlu_pos(img, landmarks):
    pos_lists = []

    # 双眼
    c1 = landmarks[36][0] - 5
    if c1 < 0:
        c1 = 0
    c2 = landmarks[45][0] + 5
    if c2 >= 128:
        c2 = c2 - 5
    r1 = np.max(np.array([landmarks[37][1], landmarks[38][1], landmarks[43][1], landmarks[44][1]])) - 5
    if r1 < 0:
        r1 = 0
    r2 = np.max(np.array([landmarks[40][1], landmarks[41][1], landmarks[46][1], landmarks[47][1]])) + 5
    if r2 >= 128:
        r2 = r2 - 5
    pos_lists.append([r1, r2, c1, c2])

    # 左眼
    c1 = landmarks[36][0] - 5
    if c1 < 0:
        c1 = 0
    c2 = landmarks[39][0] + 5
    if c2 >= 128:
        c2 = c2 - 5
    r1 = np.max(np.array([landmarks[37][1], landmarks[38][1]])) - 5
    if r1 < 0:
        r1 = 0
    r2 = np.max(np.array([landmarks[40][1], landmarks[41][1]])) + 5
    if r2 >= 128:
        r2 = r2 - 5
    pos_lists.append([r1, r2, c1, c2])

    # 右眼
    c1 = landmarks[42][0] - 5
    if c1 < 0:
        c1 = 0
    c2 = landmarks[45][0] + 5
    if c2 >= 128:
        c2 = c2 - 5
    r1 = np.max(np.array([landmarks[43][1], landmarks[44][1]])) - 5
    if r1 < 0:
        r1 = 0
    r2 = np.max(np.array([landmarks[46][1], landmarks[47][1]])) + 5
    if r2 >= 128:
        r2 = r2 - 5
    pos_lists.append([r1, r2, c1, c2])

    # 鼻子
    c1 = landmarks[31][0] - 5
    if c1 < 0:
        c1 = 0
    c2 = landmarks[35][0] + 5
    if c2 >= 128:
        c2 = c2 - 5
    r1 = np.max(np.array([landmarks[30][1]])) - 5
    if r1 < 0:
        r1 = 0
    r2 = np.max(np.array([landmarks[33][1]])) + 5
    if r2 >= 128:
        r2 = r2 - 5
    pos_lists.append([r1, r2, c1, c2])

    # 嘴巴
    c1 = landmarks[48][0] - 5
    if c1 < 0:
        c1 = 0
    c2 = landmarks[54][0] + 5
    if c2 >= 128:
        c2 = c2 - 5
    r1 = np.max(np.array([landmarks[50][1], landmarks[51][1], landmarks[52][1]])) - 5
    if r1 < 0:
        r1 = 0
    r2 = np.max(np.array([landmarks[56][1], landmarks[57][1], landmarks[58][1]])) + 5
    if r2 >= 128:
        r2 = r2 - 5
    pos_lists.append([r1, r2, c1, c2])

    # 5个随机区域
    pil_img = Image.fromarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    for i in range(5):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_img, scale=(0.1, 0.3), ratio=(1, 1))
        pos_lists.append([i, i + h, j, j + w])

    return pos_lists

base_path = 'source_data'
datapath = os.path.join(base_path,'RAF_data_gt.h5')
occlu_datapath = os.path.join(base_path,'RAF_occlu_data_gt.h5')


if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

privatetest_data_x = []
privatetest_data_y = []
privatetest_data_z = []


# 计数



data = h5py.File(datapath, 'r', driver='core')
PrivateTest_data = data['Test_pixel']
PrivateTest_labels = data['Test_label']
PrivateTest_gt = data['Test_gt']

privatetest_cnt = len(PrivateTest_data)
cnt = 0

for idx in range(privatetest_cnt):

    data_array = PrivateTest_data[idx]
    image = data_array.reshape(128, 128)
    # image = cv2.resize(image, (128, 128))
    landmarks = np.array(gmark.get_landmarks(image))
    print(landmarks.shape)
    if landmarks.shape[0] == 1:  # 利用dlib没有检测到关键点, 直接跳过
        continue
    occlu_pos = get_occlu_pos(image, landmarks)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for i in range(len(occlu_pos)):
        image_tmp = image.copy()
        image_tmp[occlu_pos[i][0]:occlu_pos[i][1], occlu_pos[i][2]:occlu_pos[i][3]] = 0
        imageName = 'tmp' + str(i) + '.jpg'
        cv2.imwrite(imageName, image_tmp)
        privatetest_data_x.append(image_tmp.tolist())
        privatetest_data_y.append(PrivateTest_labels[idx])
        privatetest_data_z.append(PrivateTest_gt[idx])
    cnt = cnt + 1
    print('cnt', cnt)


# # 读取csv文件
# fer2013_data = pd.read_csv(fer2013_path)
# ferplus_data = pd.read_csv(fer2013new_path)
#
# imageCount = 0
# # 遍历csv文件内容，并将图片数据按分类保存
# for index in range(len(fer2013_data)):
#     # 解析每一行csv文件内容
#     emotion_data = fer2013_data.loc[index][0]
#     image_data = fer2013_data.loc[index][1]
#     usage_data = fer2013_data.loc[index][2]
#
#     emotion_raw =[int(ferplus_data.loc[index][i]) for i in range(2, 12)]
#     emotion = _process_data(emotion_raw, mode = 'crossentropy')
#     idx = np.argmax(emotion)
#     if idx < 8:  # 7: not contempt, unknown or non-face ; 8: not unknown or non-face
#         emotion = emotion[:-2]
#         emotion = [float(i) / sum(emotion) for i in emotion]
#         emotion_data = emotion
#         # emotion_data = idx_to_emos[idx]
#         data_array = list(map(float, image_data.split()))
#         data_array = np.asarray(data_array)
#         image = data_array.reshape(48, 48)
#
#         image_rgb = Image.fromarray(image.astype('uint8')).convert('RGB')
#         img = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
#         res, flag = get_mask(img)
#         if flag:
#             res = np.resize(res, (128, 128))
#             res = image255(res)
#             # cv2.imwrite('atts.jpg', res)
#             seg_image_data = res.tolist()
#         else:
#             inputs = transform_test(image_rgb)
#             net = AttentionResNet()
#
#             checkpoint = torch.load(os.path.join('FER2013_Att', 'AttResNet_PrivateTest_model.t7'),
#                                     map_location=torch.device('cpu'))
#             net.load_state_dict(checkpoint['net'], False)
#             net.eval()
#             c, h, w = np.shape(inputs)
#
#             inputs = inputs.view(-1, c, h, w)
#             outputs = net(inputs)
#             outputs = outputs.view(1, h, w)
#
#             seg_image_data = transforms.ToPILImage()(outputs).convert('L')
#             seg_image_data = cv2.cvtColor(np.asarray(seg_image_data), cv2.COLOR_RGB2BGR)
#
#             seg_image_data = cv2.cvtColor(seg_image_data, cv2.COLOR_RGB2GRAY)
#             seg_image_data = cv2.resize(seg_image_data, (128, 128))
#             _, seg_image_data = cv2.threshold(seg_image_data, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#             # cv2.imwrite('atts.jpg', seg_image_data)
#             seg_image_data = seg_image_data.tolist()
#
#         if usage_data == 'Training':
#             image = np.uint8(image)
#             train_data_x.append(image.tolist())
#             train_data_y.append(emotion_data)
#             train_data_z.append(seg_image_data)
#         elif usage_data == 'PublicTest':
#             image = np.uint8(image)
#             publictest_data_x.append(image.tolist())
#             publictest_data_y.append(emotion_data)
#             publictest_data_z.append(seg_image_data)
#         elif usage_data == 'PrivateTest':
#             image = np.uint8(image)
#             privatetest_data_x.append(image.tolist())
#             privatetest_data_y.append(emotion_data)
#             privatetest_data_z.append(seg_image_data)
#         imageCount = imageCount + 1
#
#     else:
#         # vote is not majority, vaild data, pass
#         continue
#
#
#     print(imageCount)
#
#     # if imageCount == 20:
#     #      break
#
# print('总共有' + str(imageCount) + '张图片')
#
#
#
# print(np.shape(train_data_x))
# print(np.shape(train_data_y))
# print(np.shape(train_data_z))
# print(np.shape(publictest_data_x))
# print(np.shape(publictest_data_y))
# print(np.shape(publictest_data_z))
# print(np.shape(privatetest_data_x))
# print(np.shape(privatetest_data_y))
# print(np.shape(privatetest_data_z))
#
#
datafile = h5py.File(occlu_datapath, 'w')
datafile.create_dataset("Test_pixel", dtype = 'uint8', data=privatetest_data_x)
datafile.create_dataset("Test_label", dtype = 'int64', data=privatetest_data_y)
datafile.create_dataset("Test_gt", dtype = 'uint8', data=privatetest_data_z)
datafile.close()


print("Save data finish!!!")



