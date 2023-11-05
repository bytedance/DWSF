# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
from tqdm import tqdm
import cv2
from argparse import ArgumentParser
from networks.models.Discriminator import Discriminator
from networks.models.EncoderDecoder import EncoderDecoder
from utils.util import generate_random_coor, setup_seed
from utils.img import psnr_clip, draw_mask
from utils.dataset import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
setup_seed(16)


def padding_save(img, mask, batch_idx, out_path):
    h, w, c = img.shape
    if h > 384 and w > 384:
        tmp_img = np.zeros((h + 128, w + 128, 3))
        tmp_mask = np.zeros((h + 128, w + 128))
    elif h <= 384 and w > 384:
        tmp_img = np.zeros((512, w + 128, 3))
        tmp_mask = np.zeros((512, w + 128))
    elif h > 384 and w <= 384:
        tmp_img = np.zeros((h + 128, 512, 3))
        tmp_mask = np.zeros((h + 128, 512))
    else:
        tmp_img = np.zeros((512, 512, 3))
        tmp_mask = np.zeros((512, 512))
    tmp_img[:h, :w, :] = img
    tmp_mask[:h, :w] = mask
    if os.path.exists('{}/img/'.format(out_path)) == False:
        os.makedirs('{}/img/'.format(out_path))
    if os.path.exists('{}/mask/'.format(out_path)) == False:
        os.makedirs('{}/mask/'.format(out_path))
    cv2.imwrite('{}/img/{}.png'.format(out_path, batch_idx), tmp_img[..., ::-1])
    cv2.imwrite('{}/mask/{}.png'.format(out_path, batch_idx), tmp_mask)


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
            x1 = h_coor[i] - splitSize // 2
            x2 = h_coor[i] + splitSize // 2
            y1 = w_coor[i] - splitSize // 2
            y2 = w_coor[i] + splitSize // 2
            if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                tmp_block = images[:, :, x1:x2, y1:y2]
                tmp_blocks.append(tmp_block)
        tmp_blocks = torch.vstack(tmp_blocks)
        tmp_blocks_bak = tmp_blocks.clone()
        if splitSize != inputSize:
            tmp_blocks = F.interpolate(tmp_blocks, (inputSize, inputSize), mode='bicubic')

        # encode image blocks
        messages = messages.repeat((tmp_blocks.shape[0], 1))
        tmp_encode_blocks = encoder(tmp_blocks, messages)
        tmp_noise = tmp_encode_blocks - tmp_blocks
        tmp_noise = torch.clamp(tmp_noise, -0.2, 0.2)
        if splitSize != inputSize:
            tmp_noise = F.interpolate(tmp_noise, (splitSize, splitSize), mode='bicubic')

        # combined encoded blocks into watermarked image
        watermarked_images = images.clone().detach_()
        for i in range(len(h_coor)):
            x1 = h_coor[i] - splitSize // 2
            x2 = h_coor[i] + splitSize // 2
            y1 = w_coor[i] - splitSize // 2
            y2 = w_coor[i] + splitSize // 2
            if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                ori_block = tmp_blocks_bak[i:i + 1, :, :, :]
                en_block = ori_block + tmp_noise[i:i + 1, :, :, :]
                en_block = psnr_clip(en_block, ori_block, psnr)
                watermarked_images[:, :, x1:x2, y1:y2] = en_block

        return watermarked_images


if __name__ == '__main__':
    parser = ArgumentParser(description='Running code')
    parser.add_argument('--img_path', type=str, default="./")
    parser.add_argument('--out_path', type=str, default="./")
    parser.add_argument('--weight_path', type=str, default="./")
    args = parser.parse_args()

    H = 128
    W = 128
    message_length = 30
    batch_size = 1

    # select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_dataset_path = args.img_path
    val_dataset = EdDataset(val_dataset_path, transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # Build model...
    train_noise_layer = ["Combined([Identity()])"]
    encoder_decoder = EncoderDecoder(H=H, W=W, message_length=message_length, noise_layers=train_noise_layer)
    discriminator = Discriminator()
    encoder_decoder.encoder.load_state_dict(torch.load('{}/encoder_best.pth'.format(args.weight_path)))
    encoder_decoder.decoder.load_state_dict(torch.load('{}/decoder_best.pth'.format(args.weight_path)))
    encoder_decoder.to(device)
    discriminator.to(device)
    encoder_decoder.encoder.eval()
    encoder_decoder.decoder.eval()


    test_count = 40000
    for batch_idx, batch_data in enumerate(tqdm(val_loader)):
        if batch_idx > test_count:
            break

        images = batch_data
        images = images.to(device)
        messages = np.random.choice([0, 1], (images.shape[0], message_length))
        ori_messages = torch.Tensor(np.copy(messages)).to(device)
        h, w = images.shape[2], images.shape[3]

        h_coor, w_coor, splitSize = generate_random_coor(images.shape[2], images.shape[3], 128)
        encoded_images = encode(encoder_decoder.encoder, images, messages, splitSize=splitSize, inputSize=128, h_coor=h_coor, w_coor=w_coor)
        encoded_images = torch.clamp(encoded_images, -1, 1)

        en_mask = draw_mask(images.shape[2], images.shape[3], h_coor, w_coor, splitSize)
        ori_mask = np.zeros_like(en_mask)

        images = (images.cpu().numpy() + 1) / 2 * 255
        images = np.transpose(images, (0, 2, 3, 1))[0]
        images = images.astype(np.uint8)

        encoded_images = (encoded_images.cpu().numpy() + 1) / 2 * 255
        encoded_images = np.transpose(encoded_images, (0, 2, 3, 1))[0]
        encoded_images = encoded_images.astype(np.uint8)

        if images.shape[0] > 1000 or images.shape[1] > 1000:
            s = min(1000, min(images.shape[0], images.shape[1]))
            h = np.random.randint(0, max(0, images.shape[0] - s)) if max(0, images.shape[0] - s) > 0 else 0
            w = np.random.randint(0, max(0, images.shape[1] - s)) if max(0, images.shape[1] - s) > 0 else 0
            images = images[h:h + s, w:w + s]
            ori_mask = ori_mask[h:h + s, w:w + s]
            encoded_images = encoded_images[h:h + s, w:w + s]
            en_mask = en_mask[h:h + s, w:w + s]

        padding_save(encoded_images, en_mask, batch_idx, args.out_path)
