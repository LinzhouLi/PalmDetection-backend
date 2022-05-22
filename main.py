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

    k1 = (y1 - y2) / (x1 - x2)  # line AB
    b1 = y1 - k1 * x1

    k2 = (-1) / k1
    b2 = y3 - k2 * x3

    tmpX = (b2 - b1) / (k1 - k2)
    tmpY = k1 * tmpX + b1

    vec = [x3 - tmpX, y3 - tmpY]
    sidLen = math.sqrt(np.square(vec[0]) + np.square(vec[1]))
    vec = [vec[0] / sidLen, vec[1] / sidLen]
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
    cv.imwrite("img/temp_ROI.jpg",ROI)
    return ROI


def detect_two_image(dir1, dir2, net):  # 预测两个掌纹是否一致，输入为两个ROI图片路径
    dir1=[dir1]
    dir2=[dir2]
    train_tran = albumentations.Compose([
        albumentations.Resize(height=320, width=320),
    ])
    dataset = LFW(dir1, dir2,train_tran)
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
    else:
        os.mkdir(os.path.join("database", name + "_" + LorR))
        cv.imwrite(os.path.join("database", name + "_" + LorR, "1_ROI.jpg"), ROI)


def compare_to_database(img, dfg, pc, net):  # 输入一张图片和yolo预测和匹配网络, 检测是否和数据库中的数据匹配，返回False,或者匹配的人名和左右手
    extractROI(img, dfg, pc)
    imdir = os.listdir("database")
    for dir in imdir:
        imlist = os.listdir(os.path.join("database", dir))
        num_im = len(imlist)
        correct = 0
        for imdir in imlist:
            if detect_two_image(os.path.join("database", dir, imdir), "img/temp_ROI.jpg", net):
                correct = correct + 1
        if correct / num_im >= 0.75:
            print("match:" + dir)
            return dir
    print("No match")
    return False

def compare_to_database2(ROI, net):  # 输入一张图片和yolo预测和匹配网络, 检测是否和数据库中的数据匹配，返回False,或者匹配的人名和左右手
    imdir = os.listdir("database")
    cv.imwrite("img/temp_ROI.jpg",ROI)
    for dir in imdir:
        imlist = os.listdir(os.path.join("database", dir))
        num_im = len(imlist)
        correct = 0
        for imdir in imlist:
            if detect_two_image(os.path.join("database", dir, imdir), "img/temp_ROI.jpg", net):
                correct = correct + 1
        if correct / num_im >= 0.75:
            print("match:" + dir)
            return dir
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
        [request.form["dfgx1"], request.form["dfgy1"]],
        [request.form["dfgx2"], request.form["dfgy2"]]
    ]
    pc = [
        [request.form["pcx"], request.form["pcy"]]
    ]
    name = request.form["name"]
    LorR = request.form["LorR"]

    save_ROI(img, dfg, pc, name, LorR)
    return("success!")


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

    res = compare_to_database(img, dfg, pc, net)

    result = { 'match': None, 'person': None, 'LorR': None }
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

    # 掌纹注册保存手掌ROI接口调用,可以多次调用，自动在数据库相应文件夹下增加图片，建议注册操作请求4次以上，可以保证识别错误率降为千分之一
    # img = cv.imread("img/LXY_right/4.jpg")  # 示例，实际为请求发送过来的原图
    # name = "LXY"  # 示例，实际为请求发送过来的人名
    # LorR = "right"  # or "right" 同样是请求内容
    # dfg = []
    # pc = []
    # with open("img/LXY_right/4.txt", "r", encoding="utf-8") as f:  # 示例，实际为请求发送过来的yolo预测结果
    #     lines = f.read().splitlines()
    # for line in lines:
    #     line = line.split(" ")
    #     if line[0] == "0":
    #         dfg.append([(int(line[1]) + int(line[3])) / 2, (int(line[2]) + int(line[4])) / 2])  # 求中点计算
    #     elif line[0] == "1":
    #         pc.append([(int(line[1]) + int(line[3])) / 2, (int(line[2]) + int(line[4])) / 2])  # 求中点计算
    # save_ROI(img, dfg, pc, name, LorR)

    # 掌纹识别接口
    # img = cv.imread("img/LXY_right/1.jpg")  # 示例，实际为请求发送过来的原图
    # dfg = []
    # pc = []
    # with open("img/LXY_right/1.txt", "r", encoding="utf-8") as f:  # 示例，实际为请求发送过来的yolo预测结果
    #     lines = f.read().splitlines()
    # for line in lines:
    #     line = line.split(" ")
    #     if line[0] == "0":
    #         dfg.append([(int(line[1]) + int(line[3])) / 2, (int(line[2]) + int(line[4])) / 2])  # 求中点计算
    #     elif line[0] == "1":
    #         pc.append([(int(line[1]) + int(line[3])) / 2, (int(line[2]) + int(line[4])) / 2])  # 求中点计算
    # res = compare_to_database(img, dfg, pc, net)
    # if res == False:
    #     print("识别失败")
    # else:
    #     print("识别成功，识别者：", res)

    # 简单ROI测试，比较两个ROI图片
    # ROI1 = "img/LXY_left/1_ROI.jpg"  # 我的左手的一个ROI
    # ROI2 = "img/LXY_left/2_ROI.jpg"  # 我的左手的一个ROI
    # ROI3 = "img/LXY_right/1_ROI.jpg" # 我的右手的一个ROI
    # detect_two_image(ROI1, ROI2, net, gpu=True)
    # detect_two_image(ROI2, ROI3, net, gpu=True)

    app.run(host="0.0.0.0", port=args.port)

