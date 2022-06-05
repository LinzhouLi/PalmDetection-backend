import numpy as np
import torch

class LFW(object):
    def __init__(self, imgl, imgr,transform=None):
        self.imgl_list = imgl
        self.imgr_list = imgr
        self.transform = transform

    def __getitem__(self, index):
        imgl = self.imgl_list[index]
        if self.transform is not None:
            aug = self.transform(image=imgl)
            imgl = aug["image"]
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        imgr = self.imgr_list[index]
        if self.transform is not None:
            aug = self.transform(image=imgr)
            imgr = aug["image"]

        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs

    def __len__(self):
        return 1

