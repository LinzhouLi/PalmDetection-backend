import math
import os
import argparse

import numpy as np
import torch.utils.data
import albumentations
import cv2 as cv
from flask import Flask, request, jsonify

from core import model
from dataloader.LFW_loader import LFW

app = Flask(__name__)

net = None
device = 'cpu'

import imageio
img_name_list=[]
img_list=[]
imdir = os.listdir("database")
for dir in imdir:
    im_fold_list = os.listdir(os.path.join("database", dir))
    img_name_list.append(dir)
    im_in_fold_list = []
    for imdir in im_fold_list:
        im = imageio.imread(os.path.join("database", dir, imdir))
        im_in_fold_list.append(im)
    img_list.append(im_in_fold_list)


def onePoint(x, y, angle):
    X = x * math.cos(angle) + y * math.sin(angle)
    Y = y * math.cos(angle) - x * math.sin(angle)
    return [int(X), int(Y)]


def extractROI(img, dfg, pc):
    # 将普通图片转换成ROI图片返回，结构：dfg:[[x1,y1],[x2,y2]], pc:[[x3,y3]]，像素坐标，坐标均为yolo输出坐标的左上右下坐标相加除以二，即中点坐标
    (H, W) = img.shape[:2]
    if W > H:
        im = np.zeros((W, W, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = W
    else:
        im = np.zeros((H, H, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = H

    center = (edge / 2, edge / 2)

    x1 = float(dfg[0][0])
    y1 = float(dfg[0][1])
    x2 = float(dfg[1][0])
    y2 = float(dfg[1][1])
    x3 = float(pc[0][0])
    y3 = float(pc[0][1])

    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2

    unitLen = math.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    k1 = (y1 - y2) / (x1 - x2 + 1e-5)  # line AB
    b1 = y1 - k1 * x1

    k2 = (-1) / (k1 + 1e-5)
    b2 = y3 - k2 * x3

    tmpX = (b2 - b1) / (k1 - k2 + 1e-5)
    tmpY = k1 * tmpX + b1

    vec = [x3 - tmpX, y3 - tmpY]
    sidLen = math.sqrt(np.square(vec[0]) + np.square(vec[1]))
    vec = [vec[0] / (sidLen + 1e-5), vec[1] / (sidLen + 1e-5)]
    # print(vec)

    if vec[1] < 0 and vec[0] > 0:
        angle = math.pi / 2 - math.acos(vec[0])
    elif vec[1] < 0 and vec[0] < 0:
        angle = math.acos(-vec[0]) - math.pi / 2
    elif vec[1] >= 0 and vec[0] > 0:
        angle = math.acos(vec[0]) - math.pi / 2
    else:
        angle = math.pi / 2 - math.acos(-vec[0])
    # print(angle/math.pi*18)

    x0, y0 = onePoint(x0 - edge / 2, y0 - edge / 2, angle)

    x0 += edge / 2
    y0 += edge / 2

    M = cv.getRotationMatrix2D(center, angle / math.pi * 180, 1.0)
    tmp = cv.warpAffine(im, M, (edge, edge))
    ROI = tmp[int(y0 + unitLen / 2):int(y0 + unitLen * 3), int(x0 - unitLen * 5 / 4):int(x0 + unitLen * 5 / 4), :]
    ROI = cv.resize(ROI, (320, 320), interpolation=cv.INTER_CUBIC)
    cv.imwrite("img/temp_ROI.jpg", ROI)
    return ROI


def detect_two_image(dir1, dir2, net):  # 预测两个掌纹是否一致，输入为两个ROI图片路径
    dir1 = [dir1]
    dir2 = [dir2]
    train_tran = albumentations.Compose([
        albumentations.Resize(height=320, width=320),
    ])
    dataset = LFW(dir1, dir2, train_tran)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=0, drop_last=False)
    for data in loader:
        if device == 'cuda':
            for i in range(len(data)):
                data[i] = data[i].cuda()
        res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        featureL = featureL / (np.sqrt(np.sum(np.power(featureL, 2))))
        featureR = featureR / (np.sqrt(np.sum(np.power(featureR, 2))))
        score = np.sum(np.multiply(featureL, featureR))
        print("score:" + str(score))
    threshold = 0.1939
    if score >= threshold:
        print(True)
        return True
    else:
        print(False)
        return False


def save_ROI(img, dfg, pc, name, LorR):  # 输入一张图片和yolo预测，人名，左手/右手，将其转换为ROI并保存到数据库中
    ROI = extractROI(img, dfg, pc)
    if os.path.exists(os.path.join("database", name + "_" + LorR)):
        imdir = os.listdir(os.path.join("database", name + "_" + LorR))
        cv.imwrite(os.path.join("database", name + "_" + LorR, str(len(imdir) + 1) + "_ROI.jpg"), ROI)
        img_list[img_name_list.index(name + "_" + LorR)].append(imageio.imread(os.path.join("database", name + "_" + LorR, str(len(imdir) + 1) + "_ROI.jpg")))
    else:
        os.mkdir(os.path.join("database", name + "_" + LorR))
        cv.imwrite(os.path.join("database", name + "_" + LorR, "1_ROI.jpg"), ROI)
        img_name_list.insert(0, name)
        temp = [imageio.imread(os.path.join("database", name + "_" + LorR, "1_ROI.jpg"))]
        img_list.insert(0, temp)


def compare_to_database(img, dfg, pc, net):  # 输入一张图片和yolo预测和匹配网络, 检测是否和数据库中的数据匹配，返回False,或者匹配的人名和左右手
    extractROI(img, dfg, pc)
    img_ROI=imageio.imread("img/temp_ROI.jpg")
    for i in range(len(img_name_list)):
        imlist = img_list[i]
        num_im = len(imlist)
        correct = 0
        for im in imlist:
            if detect_two_image(im, img_ROI, net):
                correct = correct + 1
        if correct / num_im >= 0.75:
            print("match:" + img_name_list[i])
            return img_name_list[i]
    print("No match")
    return False




@app.route('/register', methods=["POST"])
def register():
    if not request.method == "POST":
        return

    im_file = request.files["image"]
    im_bytes = im_file.read()
    img = cv.imdecode(np.frombuffer(im_bytes, np.uint8), cv.IMREAD_COLOR)

    dfg = [
        [int(request.form["dfgx1"]), int(request.form["dfgy1"])],
        [int(request.form["dfgx2"]), int(request.form["dfgy2"])]
    ]
    pc = [
        [int(request.form["pcx"]), int(request.form["pcy"])]
    ]
    name = request.form["name"]
    LorR = request.form["LorR"]

    if dfg[0][1] > pc[0][1] and dfg[1][1] > pc[0][1]:
        img_center_x = int(img.shape[1] / 2)
        img_center_y = int(img.shape[0] / 2)
        dfg[0][0] = img_center_x + img_center_x - dfg[0][0]
        dfg[1][0] = img_center_x + img_center_x - dfg[1][0]
        dfg[0][1] = img_center_y + img_center_y - dfg[0][1]
        dfg[1][1] = img_center_y + img_center_y - dfg[1][1]
        pc[0][0] = img_center_x + img_center_x - pc[0][0]
        pc[0][1] = img_center_y + img_center_y - pc[0][1]
        img = cv.flip(img, -1)

    save_ROI(img, dfg, pc, name, LorR)
    result = {"success": True}
    return jsonify(result)


@app.route('/detect', methods=["POST"])
def detect():
    if not request.method == "POST":
        return

    print(request)
    im_file = request.files["image"]
    im_bytes = im_file.read()
    img = cv.imdecode(np.frombuffer(im_bytes, np.uint8), cv.IMREAD_COLOR)
    print(request.form)

    dfg = [
        [int(request.form["dfgx1"]), int(request.form["dfgy1"])],
        [int(request.form["dfgx2"]), int(request.form["dfgy2"])]
    ]
    pc = [
        [int(request.form["pcx"]), int(request.form["pcy"])]
    ]
    if dfg[0][1] > pc[0][1] and dfg[1][1] > pc[0][1]:
        img_center_x = int(img.shape[1] / 2)
        img_center_y = int(img.shape[0] / 2)
        dfg[0][0] = img_center_x + img_center_x - dfg[0][0]
        dfg[1][0] = img_center_x + img_center_x - dfg[1][0]
        dfg[0][1] = img_center_y + img_center_y - dfg[0][1]
        dfg[1][1] = img_center_y + img_center_y - dfg[1][1]
        pc[0][0] = img_center_x + img_center_x - pc[0][0]
        pc[0][1] = img_center_y + img_center_y - pc[0][1]
        img = cv.flip(img, -1)   
    res = compare_to_database(img, dfg, pc, net)

    result = {'match': None, 'person': None, 'LorR': None}
    if res == False:
        result['match'] = False
    else:
        result['match'] = True
        r = res.split('_')
        result['person'] = '_'.join(r[0:-1])
        result['LorR'] = r[-1]
    return jsonify(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--device', type=str, default='cpu', help='If use gpu')
    parser.add_argument('--resume', type=str, default='weight.ckpt',
                        help='The path pf save model')  # 权重路径
    args = parser.parse_args()

    # 模型权重预加载
    device = args.device
    net = model.MobileFacenet()
    if args.device == 'cuda':
        net = net.cuda()
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    app.run(host="0.0.0.0", port=args.port)

