import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--ground-truth', '-g', metavar='GROUND_TRUTH', nargs='+', help='Filenames of ground truth masks')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

# python unet/Pytorch-Unet/predict.py -i "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test_common/img" -m "/Users/ziyaoshang/Desktop/MEproject/me744_project/unet/Pytorch-UNet/checkpoint/mixed_epoch15_retrain.pth" -o "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test_common/results" -c 2 -g "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test_common/seg" --viz

# python unet/Pytorch-Unet/predict.py -i "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test/img" -m "/Users/ziyaoshang/Desktop/MEproject/me744_project/unet/Pytorch-UNet/checkpoint/mixed_epoch15_retrain.pth" -o "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test/results" -c 2 -g "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test/seg" --viz

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # if is file:
    print(args.input)
    if args.input[0].endswith('.png') or args.input[0].endswith('.jpg') or args.input[0].endswith('.jpeg'):
        in_files = args.input
    else:
        in_files = sorted(glob.glob(os.path.join(args.input[0], "*.*"), recursive=True))
        print(f"Found {len(in_files)} input files")
    out_files = get_output_filenames(args)

    # load GT
    if args.ground_truth:
        if args.ground_truth[0].endswith('.png') or args.ground_truth[0].endswith('.jpg') or args.ground_truth[0].endswith('.npy'):
            gt_files = args.ground_truth
        else:
            gt_files = sorted(glob.glob(os.path.join(args.ground_truth[0], "*.npy"), recursive=True)) 
            print(f"Found {len(gt_files)} ground truth files")
        if len(gt_files) != len(in_files):
            logging.error("Mismatched number of ground truth files and input files")
            gt_files = None

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    print(in_files)
    dices = []
    mious = []
    save_outputs = False
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        if filename.endswith('.npy'):
            img = Image.fromarray(np.load(filename))
        else:
            assert filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')
            img = Image.open(filename).convert('RGB')

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask, mask_values)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask, i)

        # compute scores if ground truth available
        if args.ground_truth and gt_files:
            gt_mask = np.load(gt_files[i])
            gt_mask = gt_mask == 1  # binary mask for fruit class

            # compute mIoU
            intersection = np.logical_and(gt_mask, mask)
            union = np.logical_or(gt_mask, mask)
            iou_score = np.sum(intersection) / np.sum(union)
            mious.append(iou_score)
            logging.info(f"mIoU for image {filename}: {iou_score:.4f}")

            # compute Dice
            intersection = np.logical_and(gt_mask, mask)
            dice_score = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(mask))
            dices.append(dice_score) 
            logging.info(f"Dice coefficient for image {filename}: {dice_score:.4f}")
        
        if save_outputs: # save segmentation to a npy file
            out_filename = os.path.join(args.output[0], os.path.basename(filename)) 
            np.save(out_filename, mask.astype(np.uint8))
            logging.info(f'Saved predicted mask to {out_filename}')


    
    # save dices and mious and print average
    if dices:
        avg_dice = sum(dices) / len(dices)
        logging.info(f"Average Dice coefficient over dataset: {avg_dice:.4f}")
        if save_outputs:
            np.save(os.path.join(args.output[0], "dice_scores.npy"), np.array(dices))
    if mious:
        avg_miou = sum(mious) / len(mious)
        logging.info(f"Average mIoU over dataset: {avg_miou:.4f}")
        if save_outputs:
            np.save(os.path.join(args.output[0], "miou_scores.npy"), np.array(mious))

        




# mixed (what testset???)
# INFO: Average Dice coefficient over dataset: 0.9366
# INFO: Average mIoU over dataset: 0.8819

# INFO: Average Dice coefficient over dataset: 0.9116
# INFO: Average mIoU over dataset: 0.8403


# single (what testset???)
# INFO: Average Dice coefficient over dataset: 0.9257
# INFO: Average mIoU over dataset: 0.8628

# INFO: Average Dice coefficient over dataset: 0.9049
# INFO: Average mIoU over dataset: 0.8276