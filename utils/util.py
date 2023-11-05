# code ported from https://github.com/jzyustc/MBRS/blob/main/network/Network.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torchvision import utils as vutils
import numpy as np
import torch


def setup_seed(seed):
    """
    set random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_images(images, path):
    images = images.cpu().data
    for i, image in enumerate(images):
        vutils.save_image(image, path)


def decoded_message_error_rate_bit(message, decoded_message):
    length = message.shape[0]
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_message(message, decoded_message):
    length = message.shape[0]
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    error_rate = (1 - int(torch.all(message == decoded_message)))
    return error_rate


def decoded_message_error_rate_bit_batch(messages, decoded_messages, mode='mean'):
    if messages.shape[0] == 1:
        messages = messages.repeat((decoded_messages.shape[0], 1))
    elif messages.shape[0] == decoded_messages.shape[0]:
        pass
    else:
        print('messages size not match, messages:{}, decoded_messages:{}'.format(messages.shape, decoded_messages.shape))
    batch_size = len(messages)
    error_rate = []
    for i in range(batch_size):
        err = decoded_message_error_rate_bit(messages[i], decoded_messages[i])
        error_rate.append(err)
    if mode == 'mean':
        error_rate = np.mean(error_rate)
    elif mode == 'min':
        error_rate = np.min(error_rate)
    elif mode == 'fusion':
        error_rate = decoded_message_error_rate_bit(messages[0], decoded_messages[0])
    else:
        print('decoded_message_error_rate_bit_batch: mode:{} is not right'.format(mode))
        return
    return error_rate


def decoded_message_error_rate_message_batch(messages, decoded_messages, mode='mean'):
    if messages.shape[0] == 1:
        messages = messages.repeat((decoded_messages.shape[0],1))
    elif messages.shape[0] == decoded_messages.shape[0]:
        pass
    else:
        print('messages size not match, messages:{}, decoded_messages:{}'.format(messages.shape, decoded_messages.shape))
    error_rate = []
    batch_size = len(messages)
    for i in range(batch_size):
        err = decoded_message_error_rate_message(messages[i], decoded_messages[i])
        error_rate.append(err)
    if mode == 'mean':
        error_rate = np.mean(error_rate)
    elif mode == 'min':
        error_rate = np.min(error_rate)
    elif mode == 'fusion':
        error_rate = decoded_message_error_rate_message(messages[0], decoded_messages[0])
    else:
        print('decoded_message_error_rate_bit_batch: mode:{} is not right'.format(mode))
        return
    return error_rate


def bit_err_count(message, target_message):
    message1 = message.gt(0.5)
    target_message1 = target_message.gt(0.5)
    return torch.sum(torch.logical_xor(message1, target_message1), dim=-1)


def get_similar_message_list(message, threshold=5):
    batch_size = message.shape[0]
    candidate = []
    for i in range(batch_size):
        # obtain similar message within given threshold
        simlist_tmp = []
        for j in range(batch_size):
            err = bit_err_count(message[i:i+1],message[j:j+1])
            if err <= threshold:
                simlist_tmp.append(message[j:j+1])

        # insert the mean of simlist into candidate, and sort candidate according to the size of simlist
        if len(simlist_tmp) >= 2:
            t = torch.vstack(simlist_tmp)
            if len(candidate) == 0:
                candidate.append([len(simlist_tmp), torch.mean(t.float(), dim=0, keepdim=True)])
            else:
                ll = len(candidate)
                for n in range(ll):
                    if candidate[n][0] < len(simlist_tmp):
                        candidate.insert(n, [len(simlist_tmp), torch.mean(t.float(), dim=0, keepdim=True)])
    if len(candidate) != 0:
        result = [a[1] for a in candidate]
    else:
        result = []

    return result


def messgae_fusion(messages, depth=2):
    """
    return a final result based on the consistency among messages
    """
    if messages.shape[0] == 1:
        return messages
    if messages.shape[0] == 2:
        final = torch.mean(messages.float(), dim=0, keepdim=True)
        return final

    final = 0
    for i in range(5):
        candidate = get_similar_message_list(messages, threshold=i)
        if len(candidate) == 0:
            continue
        elif len(candidate) == 1:
            final = candidate[0]
        elif depth > 1:
            final = messgae_fusion(torch.vstack(candidate), depth=1)
        else:
            final = candidate[0]
        return final

    if isinstance(final, int):
        final = torch.mean(messages.float(), dim=0, keepdim=True)

    return final


def is_intersect(x1, y1, x2, y2, w, h):
    if abs(x1-x2) < w and abs(y1-y2) < h:
        return 1
    else:
        return -1


def generate_random_coor(height, width, splitSize):
    """
    generate random coordinates
    """
    i = 0
    h_coor = []
    w_coor = []

    if height < 128 or width < 128:
        splitSize = min(height // 2 * 2, width // 2 * 2)
        h_coor = [height // 2]
        w_coor = [width // 2]
        return h_coor, w_coor, splitSize

    cc = (height * width * 0.25) // (splitSize * splitSize)
    if cc >= 3:
        cc = floor(cc)
    else:
        cc = round(cc)
    count = max(2, min(cc, 20))
    splitsizeH = max(splitSize * 1.3, min(height // 4, splitSize * 4))
    splitsizeW = max(splitSize * 1.3, min(width // 4,  splitSize * 4))
    try:
        while len(h_coor) < count:
            h = np.random.randint(0, height)
            w = np.random.randint(0, width)
            i += 1
            if h - splitSize // 2 > 0 and h + splitSize // 2 < height and w - splitSize // 2 > 0 and w + splitSize // 2 < width:
                if len(h_coor) == 0:
                    h_coor.append(h)
                    w_coor.append(w)
                else:
                    flag = True
                    lens = len(h_coor)
                    for j in range(lens):
                        if is_intersect(w, h, w_coor[j], h_coor[j], splitsizeW, splitsizeH) > 0:
                            flag = False
                            break
                    if flag:
                        h_coor.append(h)
                        w_coor.append(w)
            if i == 200:
                break
    except:
        pass

    if len(h_coor) == 0:
        h_coor = [height // 2]
        w_coor = [width // 2]
        splitSize = min(height, min(width, splitSize)) // 2 * 2

    return h_coor, w_coor, splitSize

