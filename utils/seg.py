# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import torch
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
import math
from networks.segmentation.model import U2NETP

model = None
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
interpolatemode = 'bicubic'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def init():
    global model
    model = U2NETP(mode='eval').to(device)
    checkpoint = torch.load('./results/seg_checkpoint/seg.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


def generate_mask(image):
    """
    return mask
    """
    if model is None:
        init()
    if isinstance(image, list):
        image = [F.interpolate(im, (512,512), mode=interpolatemode) for im in image]
        image = torch.vstack(image)
    else:
        image = F.interpolate(image, (512,512), mode=interpolatemode)
    with torch.no_grad():
        image = image.cuda()
        d0, d1, d2, d3, d4, d5, d6 = model(image)
    return d0


def rectify(image, mask, threshold=128):
    """
    rectify geometric transform
    """
    if isinstance(image,torch.Tensor):
        image = image.clone().detach_()
        image = image.cpu().numpy().transpose(0,2,3,1)[0]
        image = (image+1.)*127.5
    else:
        image = np.array(image)

    # binarize mask
    mask = mask[:, :, 0]
    mask = np.uint8(mask * 255)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    # find top20 minimum bounding rectangles
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    # estimate rotation angle
    angles = [0]
    for cnt in sorted_contours:
        rec = cv2.minAreaRect(cnt)
        width, height = rec[1]
        angle = rec[2]
        if width >= 55 and height >= 55 and width <= 4 * height and height <= 4 * width and height<=265 and width <= 265:
            if abs(angle) > 45:
                if angle < 0:
                    angle = -(angle + 90)
                else:
                    angle = 90 - angle
            else:
                angle = -angle
            angle = round(angle)
            angles.append(angle)
    angles = search_near(angles, [8,4,2])
    angle = np.mean(angles)

    # inverse rotation
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), -angle, 1)
    rotate_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    rotate_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    # find top20 minimum bounding rectangles
    rotate_mask[rotate_mask < threshold] = 0
    rotate_mask[rotate_mask >= threshold] = 255
    _, thresh = cv2.threshold(rotate_mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    # obtain all blocks with defalut size (128x128)
    images_list = []
    height_list = []
    width_list = []
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h>= 55 and w >= 55 and w <= 4 * h and h <= 4 * w and h<=265 and w<=265:
            tmp = rotate_img[y:y + h, x:x + w]
            tmp = cv2.resize(tmp, (128, 128))
            images_list.append(tmp)
            height_list.append(h)
            width_list.append(w)


    if len(height_list) == 0:
        height_list.append(128)
    if len(width_list) == 0:
        width_list.append(128)
    height_list = search_near(height_list, [64,32,16,8])
    width_list = search_near(width_list, [64,32,16,8])

    if len(images_list) == 0:
        images_list.append(cv2.resize(rotate_img, (128,128)))

    images_list = [torch.tensor(image) for image in images_list]
    images_tensor = torch.stack(images_list)
    images_tensor = images_tensor.permute((0,3,1,2))
    images_tensor = (images_tensor/255-0.5)/0.5

    return images_tensor, [np.mean(height_list), np.mean(width_list)], angle


def search_near(num_list, qs_list=[]):
    """
    search near results within given quantization step
    """
    for qs in qs_list:
        num_list_bak = []
        for i in num_list:
            num_list_bak.append(i // qs * qs)
        maxlabel = max(set(num_list_bak), key=num_list_bak.count)
        result = []
        for i in range(len(num_list_bak)):
            if num_list_bak[i] == maxlabel:
                result.append(num_list[i])
        num_list = result
    return num_list


def pad_split_seg_rectify(image, targetH=512, targetW=512):
    """
    segment and rectify
    """
    height, width = image.shape[2], image.shape[3]
    target_height = math.ceil(height/targetH)*targetH
    target_width = math.ceil(width/targetW)*targetW

    # pad, split and segment
    image_pad = F.pad(image, [0, target_width-width, 0, target_height-height], mode='constant', value=0)
    imagepatchs = image_pad.view(image_pad.shape[0],image_pad.shape[1],image_pad.shape[2]//targetH,targetH,image_pad.shape[3]//targetW,targetW)
    imagepatchs = imagepatchs.permute(2,4,0,1,3,5)
    imagestensor = imagepatchs.reshape(-1, imagepatchs.shape[3], imagepatchs.shape[4], imagepatchs.shape[5])
    mask_list = []
    for i in range(0, imagestensor.shape[0], 8):
        mask_tensor = generate_mask(imagestensor[i:i+8])
        mask_list.append(mask_tensor)
    mask_tensor = torch.vstack(mask_list)
    mask_tensor = mask_tensor.reshape(imagepatchs.shape[0],imagepatchs.shape[1],imagepatchs.shape[2],1,imagepatchs.shape[4],imagepatchs.shape[5])
    mask_tensor = mask_tensor.permute(2,3,0,4,1,5)
    mask_tensor = mask_tensor.reshape(image_pad.shape[0],1,image_pad.shape[2],image_pad.shape[3])
    mask_tensor = mask_tensor.permute(0,2,3,1)
    mask = mask_tensor.cpu().numpy()

    # rectify
    image_tensor, [height,width], angle = rectify(image_pad, mask[0])

    return image_tensor.cuda(), [height,width], angle


def obtain_wm_blocks(image, targetH=512, targetW=512):
    """
    return rectified watermarked blocks
    """
    image_tensor1, [height, width], angle = pad_split_seg_rectify(image, targetH=targetH, targetW=targetW)
    image = F.interpolate(image, (math.ceil(image.shape[2]/(height/128)), math.ceil(image.shape[3]/(width/128))), mode=interpolatemode)
    image_tensor2, [height, width], angle = pad_split_seg_rectify(image, targetH=targetH, targetW=targetW)
    
    image_tensor = torch.vstack([image_tensor1, image_tensor2])

    return image_tensor.cuda()
    

