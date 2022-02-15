import os.path as osp
import os
import numpy as np
import cv2
import dlib
import math
from torchvision import transforms
from PIL import Image

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

# 加入alpha通道 控制透明度
def merge_add_alpha(img_1, mask):
    # merge rgb and mask into a rgba image
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def merge_add_mask(img_1, mask):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        channel_num = mask.shape[2]
        for row in range(height):
            for col in range(width):
                for c in range(channel_num):
                    if mask[row, col, c] == 0:
                        mask[row, col, c] = 0
                    else:
                        mask[row, col, c] = 255

        r_channel, g_channel, b_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, mask)
        g_channel = cv2.bitwise_and(g_channel, mask)
        b_channel = cv2.bitwise_and(b_channel, mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
    else:
        res_img = img_1
    return res_img

def get_landmarks(image):

    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # rects = detector(img_gray, 0)
    rects = detector(image, 0)
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

def single_face_alignment(face, landmarks):
    #print(landmarks, type(landmarks))
    order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
    for j in order:
        x = landmarks[j, 0]
        y = landmarks[j, 1]
        cv2.circle(face, (x, y), 2, (0, 0, 255), -1)
    eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,  # 计算两眼的中心坐标
                  (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
    dx = (landmarks[45, 0] - landmarks[36, 0])  # note: right - right
    dy = (landmarks[45, 1] - landmarks[36, 1])

    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
    return align_face


def get_seg_face(image):
    #image = cv2.imread("iuput_pic/854.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origin_shape = image.shape
    if image.shape[0] < 80 or image.shape[1] < 80:
        image = cv2.resize(image, (80, 80))
    #print(image.shape)
    landmarks = get_landmarks(image)
    if landmarks.shape[1] == 0:
        return image, False
    else:
        mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)

    #image_bgra = merge_add_alpha(image, mask)
    #image_bgra = cv2.resize(image_bgra, (origin_shape[0], origin_shape[1]))
    #cv2.imwrite("result_add_alpha.png", image_bgra)

    image_bgr = merge_add_mask(image, mask)

    # image_bgr = cv2.resize(image_bgr, (origin_shape[0], origin_shape[1]))
    #image_bgr = single_face_alignment(image_bgr, landmarks)
    return image_bgr, True
    #cv2.imwrite("result_add_mask.png", image_bgr)
    #print(np.shape(image_bgra))
    #print(np.shape(image_bgr))

def get_mask(image):
    # origin_shape = image.shape
    # if image.shape[0] < 80 or image.shape[1] < 80:
    #     image = cv2.resize(image, (80, 80))
    image = cv2.resize(image, (128, 128))
    # print(image.shape)
    landmarks = get_landmarks(image)
    if landmarks.shape[1] == 0:
        return image, False
    else:
        mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)

    # image_bgra = merge_add_alpha(image, mask)
    # image_bgra = cv2.resize(image_bgra, (origin_shape[0], origin_shape[1]))
    # cv2.imwrite("result_add_alpha.png", image_bgra)

    # image_bgr = merge_add_mask(image, mask)

    # image_bgr = cv2.resize(image_bgr, (origin_shape[0], origin_shape[1]))
    # image_bgr = single_face_alignment(image_bgr, landmarks)
    # print(mask)
    return mask, True

if __name__ == '__main__':
    img = cv2.imread('0.jpg')
    if img.shape[0] < 128 or img.shape[1] < 128:
        img = cv2.resize(img, (128, 128))
    img_bgr, flag = get_seg_face(img)
    # cv2.imwrite('result_add_mask.png', img_bgr)
    landmarks = np.array(get_landmarks(img))

    occlu_eyes_img = img
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
    # print(r1, r2, c1, c2)
    # pil_img = Image.fromarray(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    # i, j, h, w = transforms.RandomResizedCrop.get_params(
    #         pil_img, scale=(0.1, 0.4), ratio=(1, 1))
    # occlu_eyes_img[i:i+h, j:j+w, 0:3] = 0
    occlu_eyes_img[r1:r2, c1:c2, 0:3] = 0
    cv2.imwrite('result_occlu_random.png', occlu_eyes_img)
    # print(landmarks)