# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import time
import kornia
from argparse import ArgumentParser
from networks.models.EncoderDecoder import EncoderDecoder
from networks.models.Discriminator import Discriminator
from networks.models.Noiser import Noise
from utils.util import setup_seed, save_images, decoded_message_error_rate_message_batch,  decoded_message_error_rate_bit_batch
from utils.dataset import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
setup_seed(100)

if __name__ == '__main__':
    parser = ArgumentParser(description='Running code')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_dataset_path', type=str, default='./')
    parser.add_argument('--val_dataset_path', type=str, default='./')
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    H = 128
    W = 128
    message_length = 30
    batch_size = args.batch_size

    save_pth_path = args.save_path + '/pth/'
    save_image_path = args.save_path + '/image/'
    if os.path.exists(save_pth_path) == False:
        os.makedirs(save_pth_path)
    if os.path.exists(save_image_path) == False:
        os.makedirs(save_image_path)

    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop((H, W), pad_if_needed=True, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # build dataloader
    train_dataset_path = args.train_dataset_path
    val_dataset_path = args.val_dataset_path
    train_dataset = EdDataset(train_dataset_path, transform)
    val_dataset = EdDataset(val_dataset_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Build model...
    train_noise_layer = ["Combined([Identity(),RandomJpegMask(50,100, padding=True), RandomJpeg(50,100,padding=True), RandomJpegSS(50,100,padding=True), RandomGN(3,10), RandomGF(3,8), RandomColor(0.5,1.5), RandomDropout(0.7,1),\
                                    Identity(),RandomJpegMask(50,100, padding=True), RandomJpeg(50,100,padding=True), RandomJpegSS(50,100,padding=True), RandomGN(3,10), RandomGF(3,8), RandomColor(0.5,1.5), RandomDropout(0.7,1), \
                                        RandomRotate(10), RandomResize(target_size=128), RandomCrop(0.9, 1, target_size=128), RandomPIP(1,1.1, target_size=128)])"]
    val_noise_layer = ["Combined([Identity(), RandomJpegTest(50,100),  RandomGN(3,10), RandomGF(3,8), RandomColor(0.5,1.5), RandomDropout(0.7,1), \
                                    Identity(), RandomJpegTest(50,100),  RandomGN(3,10), RandomGF(3,8), RandomColor(0.5,1.5), RandomDropout(0.7,1), \
                                    RandomRotate(10), RandomResize(target_size=128), RandomCrop(0.9, 1, target_size=128), RandomPIP(1,1.1, target_size=128)])"]
    encoder_decoder = EncoderDecoder(H=H, W=W, message_length=message_length, noise_layers=train_noise_layer)
    discriminator = Discriminator()
    val_noiser = Noise(val_noise_layer)
    encoder_decoder.to(device)
    discriminator.to(device)
    val_noiser.to(device)

    optimizer = torch.optim.Adamw(encoder_decoder.parameters(), lr=1e-4)
    optimizer_dis = torch.optim.Adamw(discriminator.parameters(), lr=1e-4)

    mseloss = torch.nn.MSELoss().to(device)
    binaryloss = torch.nn.BCEWithLogitsLoss().to(device)
    ssim_loss = kornia.losses.MS_SSIMLoss(data_range=2, alpha=0.5).to(device)

    min_loss = 1000000
    dis_weight = 1e-3
    encode_weight = 0.2
    decode_weight = 1
    best_epoch = 0
    label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
    label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
    for epoch in range(100):
        start = time.time()
        train_en_loss_tmp = 0
        train_de_loss_tmp = 0
        train_en_dis_loss_tmp = 0
        train_dis_lossT_tmp = 0
        train_dis_lossF_tmp = 0
        train_loss_tmp = 0
        encoder_decoder.encoder.train()
        encoder_decoder.decoder.train()
        discriminator.train()
        print("========training=============")
        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data
            images = images.to(device)
            ori_images = images.clone().detach()

            # generate random message tensor
            messages = np.random.choice([0, 1], (images.shape[0], message_length))
            ori_messages = np.copy(messages)
            ori_messages = torch.Tensor(ori_messages).to(device)
            messages = torch.Tensor(messages).to(device)

            # encode image
            encode_images = encoder_decoder.encoder(images, messages)
            encode_images = torch.clamp(encode_images, -1, 1)

            # distort encoded image
            noised_images = encoder_decoder.noise([encode_images, images])

            # decode message
            decode_messages = encoder_decoder.decoder(noised_images)

            # optimizer discriminator
            d_cover = discriminator(images)
            train_d_cover_loss = binaryloss(d_cover, label_cover[:d_cover.shape[0]])
            d_encoded = discriminator(encode_images.detach())
            train_d_encoded_loss = binaryloss(d_encoded, label_encoded[:d_encoded.shape[0]])

            train_d_loss = train_d_encoded_loss + train_d_cover_loss

            optimizer_dis.zero_grad()
            train_d_loss.backward()
            optimizer_dis.step()

            train_dis_lossF_tmp += train_d_encoded_loss
            train_dis_lossT_tmp += train_d_cover_loss

            # optimizer encoder_decoder
            g_encoded = discriminator(encode_images)
            train_gen_loss = binaryloss(g_encoded, label_cover[:g_encoded.shape[0]])
            train_mse_loss = mseloss(images, encode_images)
            train_msssim_loss = ssim_loss(images, encode_images)
            train_encode_loss = train_mse_loss + 0.005 * train_msssim_loss
            train_decode_loss = mseloss(decode_messages, messages)

            train_loss = dis_weight * train_gen_loss + encode_weight * train_encode_loss + decode_weight * train_decode_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_en_dis_loss_tmp += train_gen_loss
            train_en_loss_tmp += train_encode_loss
            train_de_loss_tmp += train_decode_loss
            train_loss_tmp += train_loss

            if batch_idx == 0:
                save_images((images + 1) / 2., save_image_path + '/epoch-{}-{}-ori.png'.format(epoch, batch_idx))
                save_images((encode_images + 1) / 2., save_image_path + '/epoch-{}-{}-en.png'.format(epoch, batch_idx))
                save_images((noised_images + 1) / 2., save_image_path + '/epoch-{}-{}-en_dis.png'.format(epoch, batch_idx))
                save_images((encode_images - images + 1) / 2., save_image_path + '/epoch-{}-{}-noise.png'.format(epoch, batch_idx))
            if batch_idx % 100 == 0:
                print(
                    'Train Epoch: {} [{}/{}]\tD_loss: {} = DisF:{} + DisT:{}\n\tEn_De_loss: {} = Gen:{} + En:{}=MSE-{}+MSSIM-{}, De:{}'.format(
                        epoch, batch_idx, len(train_dataset) // batch_size,
                                          train_d_encoded_loss.item() + train_d_cover_loss.item(),
                        train_d_encoded_loss.item(), train_d_cover_loss.item(), train_loss.item(),
                        train_gen_loss.item(), train_encode_loss.item(),
                        train_mse_loss.item(), train_msssim_loss, train_decode_loss.item()))

        print('>>>>>>>Train Epoch: {} \tD_loss: = DisF:{} + DisT:{}\n\tEn_De_loss: {} = Gen:{} + En:{} + De:{}'.format(
            epoch, train_dis_lossF_tmp, train_dis_lossT_tmp,
            train_loss_tmp, train_en_dis_loss_tmp, train_en_loss_tmp, train_de_loss_tmp))

        print("========evaluating=============")
        error_rate_bit_all = 0
        error_rate_message_all = 0
        psnr_all = 0
        ssim_all = 0
        val_loss_tmp = 0.
        val_en_loss_tmp = 0.
        val_de_loss_tmp = 0.
        val_en_dis_loss_tmp = 0.
        val_loss_tmp = 0.
        val_count = 0
        encoder_decoder.encoder.eval()
        encoder_decoder.decoder.eval()
        discriminator.eval()
        with torch.no_grad():
            for batch_idx_val, batch_data_val in enumerate(val_loader):
                val_count += 1
                images = batch_data_val
                images = images.to(device)
                ori_images = images.clone().detach()

                # generate random message tensor
                messages = np.random.choice([0, 1], (images.shape[0], message_length))
                ori_messages = np.copy(messages)
                ori_messages = torch.Tensor(ori_messages).to(device)
                messages = torch.Tensor(messages)
                messages = messages.to(device)

                encode_images = encoder_decoder.encoder(images, messages)
                encode_images = torch.clamp(encode_images, -1, 1)
                encode_images = torch.clamp((encode_images + 1) * 127.5, 0, 255).int()
                encode_images = (encode_images / 255. - 0.5) / 0.5

                # distort encoded image
                noised_images = val_noiser([encode_images, images])
                noised_images = torch.clamp((noised_images + 1) * 127.5, 0, 255).int()
                noised_images = (noised_images / 255. - 0.5) / 0.5

                # decode message
                decode_messages = encoder_decoder.decoder(noised_images)

                g_encoded = discriminator(encode_images)
                val_gen_loss = binaryloss(g_encoded, label_cover[:g_encoded.shape[0]])
                val_mse_loss = mseloss(images, encode_images)
                val_msssim_loss = ssim_loss(images, encode_images)
                val_encode_loss = val_mse_loss + 0.005 * val_msssim_loss
                val_decode_loss = mseloss(decode_messages, messages)

                val_loss = dis_weight * val_gen_loss + encode_weight * val_encode_loss + decode_weight * val_decode_loss

                val_loss_tmp += val_loss
                val_en_dis_loss_tmp += val_gen_loss
                val_en_loss_tmp += val_encode_loss
                val_de_loss_tmp += val_decode_loss

                if batch_idx_val == 0:
                    print(messages[0])
                    print(decode_messages[0])
                    print('val loss:', val_loss, ' gen loss:', val_gen_loss, 'encode loss:', val_encode_loss,
                          'decode loss:', val_decode_loss)

                # error rate
                error_rate_bit_all += decoded_message_error_rate_bit_batch(messages, decode_messages)
                error_rate_message_all += decoded_message_error_rate_message_batch(messages, decode_messages)

                # psnr & ssim
                psnr_all += -kornia.losses.psnr_loss(encode_images.detach(), images, max_val=2.0)
                ssim_all += 1 - 2 * kornia.losses.ssim_loss(encode_images.detach(), images, max_val=1.0, window_size=5, reduction="mean")

                if batch_idx_val == 0:
                    save_images((images + 1) / 2., save_image_path + '/val_epoch-{}-ori.png'.format(epoch))
                    save_images((encode_images + 1) / 2., save_image_path + '/val_epoch-{}-en.png'.format(epoch))
                    save_images((noised_images + 1) / 2., save_image_path + '/val_epoch-{}-en_dis.png'.format(epoch))
                    save_images((encode_images - images + 1) / 2.,
                                save_image_path + '/val_epoch-{}-noise.png'.format(epoch))

            print('loss: {}'.format(val_loss_tmp / val_count))
            print('error_bit: {}'.format(error_rate_bit_all / val_count))
            print('error_message: {}'.format(error_rate_message_all / val_count))
            print('psnr: {}'.format(psnr_all / val_count))
            print('ssim: {}'.format(ssim_all / val_count))
            torch.save(encoder_decoder.encoder.state_dict(), save_pth_path + '/encoder_{}.pth'.format(epoch))
            torch.save(encoder_decoder.decoder.state_dict(), save_pth_path + '/decoder_{}.pth'.format(epoch))
            torch.save(discriminator.state_dict(), save_pth_path + '/discriminator_{}.pth'.format(epoch))

        # save best model
        if val_loss_tmp < min_loss:
            min_loss = val_loss_tmp
            torch.save(encoder_decoder.encoder.state_dict(), save_pth_path + '/encoder_best.pth')
            torch.save(encoder_decoder.decoder.state_dict(), save_pth_path + '/decoder_best.pth')
            torch.save(discriminator.state_dict(), save_pth_path + '/discriminator_best.pth')
            best_epoch = epoch
            print('save on epoch-{}'.format(epoch))

        end = time.time()
        print('Time(epoch-{}):{}'.format(epoch, end - start))


