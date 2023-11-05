# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import time
import kornia
from tqdm import tqdm
import copy
import os
from argparse import ArgumentParser
from networks.models.Discriminator import Discriminator
from networks.models.EncoderDecoder import EncoderDecoder
from networks.models.Noiser import Noise
from utils.util import *
from utils.img import psnr_clip
from utils.util import setup_seed, save_images
from utils.seg import obtain_wm_blocks
from utils.crc import crc

setup_seed(30)


def encode(encoder, images, messages, splitSize=128, inputSize=128, h_coor=[], w_coor=[], psnr=35):
    """
    Encode image blocks based on random coordinates
    """
    with torch.no_grad():
        if isinstance(messages, np.ndarray):
            messages = torch.Tensor(messages)
            messages = messages.to(device)
        
        # obtain image blocks
        tmp_blocks = []
        for i in range(len(h_coor)):
            x1 = h_coor[i]-splitSize//2
            x2 = h_coor[i]+splitSize//2
            y1 = w_coor[i]-splitSize//2
            y2 = w_coor[i]+splitSize//2
            if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                tmp_block = images[:, :, x1:x2, y1:y2]
                tmp_blocks.append(tmp_block)
        tmp_blocks = torch.vstack(tmp_blocks)
        tmp_blocks_bak = tmp_blocks.clone()
        if splitSize != inputSize:
            tmp_blocks = F.interpolate(tmp_blocks, (inputSize,inputSize),mode='bicubic')
        
        # encode image blocks
        messages = messages.repeat((tmp_blocks.shape[0],1))
        tmp_encode_blocks = encoder(tmp_blocks, messages)
        tmp_noise = tmp_encode_blocks - tmp_blocks
        tmp_noise = torch.clamp(tmp_noise, -0.2, 0.2)
        if splitSize != inputSize:
            tmp_noise = F.interpolate(tmp_noise, (splitSize, splitSize),mode='bicubic')

        # combined encoded blocks into watermarked image
        watermarked_images = images.clone().detach_()
        for i in range(len(h_coor)):
            x1 = h_coor[i]-splitSize//2
            x2 = h_coor[i]+splitSize//2
            y1 = w_coor[i]-splitSize//2
            y2 = w_coor[i]+splitSize//2
            if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                ori_block = tmp_blocks_bak[i:i+1, :, :, :]
                en_block = ori_block + tmp_noise[i:i+1, :, :, :]
                en_block = psnr_clip(en_block, ori_block, psnr)
                watermarked_images[:, :, x1:x2, y1:y2] = en_block

        return watermarked_images


def decode(decoder, noised_images):
    """
    Decode images or noised images
    """
    with torch.no_grad():
        noised_blocks = obtain_wm_blocks(noised_images)
        decode_messages = []
        for _ in range(0, len(noised_blocks), 32):
            decode_messages.append(decoder(noised_blocks[_:_+32]))
        decode_messages = torch.vstack(decode_messages)
    
    return decode_messages


def image_quality_evalute(images, encode_images):
    """
    evaluate the visual quality
    """
    # calculate psnr and ssim
    psnr = -kornia.losses.psnr_loss(encode_images.detach(), images, max_val=2.).item()
    ssim = 1 - 2 * kornia.losses.ssim_loss(encode_images.detach(), images, max_val=1., window_size=5, reduction="mean").item()
  
    return psnr, ssim


def message_evaluate(messages, decode_messages, mode='mean'):
    """
    mode: mean, min, fusion
    return the bit error rate between messages and decode_messages
    """
    error_rate_bit = decoded_message_error_rate_bit_batch(messages, decode_messages, mode)

    return error_rate_bit


def search_dir_file(rootdir):
    """
    return image file path
    """
    allfile = []
    for dir_or_file in os.listdir(rootdir):
        filePath = os.path.join(rootdir, dir_or_file)
        if os.path.isfile(filePath):
            if os.path.basename(filePath).endswith('.jpg') or os.path.basename(filePath).endswith('.png') or os.path.basename(filePath).endswith('.jpeg'):
                allfile.append(filePath)
            else:
                continue
        elif os.path.isdir(filePath):
            allfile.extend(search_dir_file(filePath))
        else:
            print('not file and dir '+os.path.basename(filePath))
    return allfile


def crc_evaluate(ori_decode_messages):
    """
    return the crc result, i.e., bit_check_accuracy
    """
    decode_messages = (ori_decode_messages.gt(0.5)).int()
    decode_messages = decode_messages.cpu().numpy()
    decode_messages[decode_messages != 0] = 1
    flag = 0
    for j in range(len(decode_messages)):
        flag = crc(decode_messages[j:j+1, :], 'decode')
        if flag == 1:
            break
    return flag


if __name__ == '__main__':
    parser = ArgumentParser(description='Running code')
    parser.add_argument('--ori_path', type=str, default='./')
    parser.add_argument('--pth_path', type=str, default='./')
    parser.add_argument('--out_path', type=str, default='./')
    args = parser.parse_args()

    localtime = time.asctime(time.localtime(time.time()))
    print('Evaluate Time:', localtime)

    H = 128
    W = 128
    message_length = 30
    batch_size = 1

    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    # Build model...
    default_noise_layer = ["Combined([Identity()])"]
    encoder_decoder = EncoderDecoder(H=H, W=W, message_length=message_length, noise_layers=default_noise_layer)
    discriminator = Discriminator()

    # attack list
    noise_list = [
                ["Identity()"], ["RandomJpegTest(50,100)"],  ["RandomGN(3,10)"],
                ["RandomGF(3,8)"], ["RandomColor(0.5,1.5)"], ["RandomDropout(0.7,1)"],
                ["RandomResize(0.5,2)"], ["RandomCrop(0.7,1)"],  ["RandomRotate(30)"],
                ["RandomPadding(0,100)"], ["RandomOcclusion(0.25,0.5)"], ["RandomPIP(1,2)"],
                ["Joint([RandomColor(0.5,1.5), RandomJpegTest(50,100)])"],
                ["Joint([RandomCrop(0.7,1), RandomJpegTest(50,100)])"],
                ["Joint([RandomCrop(0.7,1), RandomResize(0.5,2)])"],
                ["Joint([RandomOcclusion(0.25,0.5), RandomJpegTest(50,100)])"],
                ["Joint([RandomCrop(0.7,1), RandomResize(0.5,2), RandomJpegTest(50,100)])"],
                ["Joint([RandomCrop(0.7,1),RandomOcclusion(0.25,0.5),RandomJpegTest(50,100)])"],
                ]
    noise_list_bak = copy.deepcopy(noise_list)
    noise_layers = []
    for noise in noise_list:
        noise_layers.append(Noise(noise))

    # load checkpoint
    encoder_decoder.encoder.load_state_dict(torch.load(args.pth_path+'/encoder_best.pth'))
    encoder_decoder.decoder.load_state_dict(torch.load(args.pth_path+'/decoder_best.pth'))
    encoder_decoder.to(device)
    discriminator.to(device)
    encoder_decoder.encoder.eval()
    encoder_decoder.decoder.eval()

    ori_path = args.ori_path
    out_path = args.out_path
    val_loader = search_dir_file(ori_path)
    print("================Encoding================")
    messages_all = []
    use_crc = True
    for batch_idx, batch_data in enumerate(tqdm(val_loader)):
        images = [transform(Image.open(idx).convert('RGB')) for idx in [batch_data]]
        images = torch.stack(images, dim=0)
        images = images.to(device)

        if use_crc:
            messages = np.random.choice([0, 1], (images.shape[0], message_length-8))
            messages = crc(messages, 'encode')
            ori_messages = torch.Tensor(np.copy(messages)).to(device)
        else:
            messages = np.random.choice([0, 1], (images.shape[0], message_length))
            ori_messages = torch.Tensor(np.copy(messages)).to(device)
        messages_all.append(ori_messages)

        # add watermark
        h_coor, w_coor, splitSize = generate_random_coor(images.shape[2], images.shape[3], 128)
        encoded_images = encode(encoder_decoder.encoder, images, messages, splitSize=splitSize,
                                inputSize=H, h_coor=h_coor, w_coor=w_coor, psnr=35)
        encoded_images = torch.clamp(encoded_images, -1, 1)
        
        if os.path.exists(out_path) == False:
            os.makedirs(out_path)
        name = batch_data.split('/')[-1]
        name = name.split('.')[0]
        save_images((encoded_images+1)/2, out_path+'/{}.png'.format(name))

    messages_all = torch.vstack(messages_all)
    torch.save(messages_all, out_path+'/msg.pth')
    
    print("================Decoding================")
    messages_all = torch.load(out_path+'/msg.pth')
    ori_val_loader = search_dir_file(ori_path)
    val_loader = search_dir_file(out_path)
    for i, val_noiser in enumerate(noise_layers):
        print("================Test:", noise_list_bak[i], "===========")
        psnr_all = []
        ssim_all = []
        bit_accuracy_min = []
        bit_accuracy_mean = []
        bit_accuracy_adapt = []
        bit_check_accuracy_min = []
        bit_check_accuracy_mean = []
        bit_check_accuracy_adapt = []
        for batch_idx, batch_data in enumerate(tqdm(ori_val_loader)):
            ori_messages = messages_all[batch_idx:batch_idx+1]

            images = [transform(Image.open(idx).convert('RGB')) for idx in [batch_data]]
            images = torch.stack(images, dim=0)
            images = images.to(device)

            encoded_images = [transform(Image.open(idx).convert('RGB')) for idx in val_loader[batch_idx:batch_idx+1]]
            encoded_images = torch.stack(encoded_images, dim=0)
            encoded_images = encoded_images.to(device)

            # attack watermarked image
            noised_images = val_noiser([encoded_images,images])

            # decode
            decode_messages = decode(encoder_decoder.decoder, noised_images)

            # evaluate
            bit_error = message_evaluate(ori_messages, decode_messages, mode='min')
            bit_accuracy_min.append(1-bit_error)

            bit_error = message_evaluate(ori_messages, decode_messages, mode='mean')
            bit_accuracy_mean.append(1-bit_error)

            decode_messages_fusion = messgae_fusion(decode_messages)
            bit_error = message_evaluate(ori_messages, decode_messages_fusion, mode='fusion')
            bit_accuracy_adapt.append(1-bit_error)

            flag = crc_evaluate(decode_messages)
            bit_check_accuracy_mean.append(flag)
            bit_check_accuracy_min.append(flag)

            flag = crc_evaluate(torch.vstack([decode_messages, decode_messages_fusion]))
            bit_check_accuracy_adapt.append(flag)

            psnr, ssim = image_quality_evalute(images, encoded_images)
            psnr_all.append(psnr)
            ssim_all.append(ssim)

        print('========result:{}========'.format(noise_list_bak[i]))
        print('psnr: {}\t ssim: {}\t len:{}'.format(np.mean(psnr_all), np.mean(ssim_all), len(psnr_all)))
        print('MIN:','bit_acc: {}\t bit_check_acc: {}'.format(np.mean(bit_accuracy_min),  np.mean(bit_check_accuracy_min)))
        print('MEAN:','bit_acc: {}\t bit_check_acc: {}'.format(np.mean(bit_accuracy_mean), np.mean(bit_check_accuracy_mean)))
        print('FUSION:','bit_acc: {}\t bit_check_acc: {}'.format(np.mean(bit_accuracy_adapt), np.mean(bit_check_accuracy_adapt)))
