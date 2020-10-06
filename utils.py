import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def project_results(slides, slides_indices, save_dir, slides_probs=None, color=(255, 255, 255),
                    on_thumbnail=True, slides_labels=None, postfix='projected'):
    patch_size = 512  # the size of patches extracted from 20x svs image
    target_mag = 20
    slides_info = pd.read_pickle('./data_for_train/slides_size_info.pickle')
    for slide_name in tqdm(sorted(slides.Slide_name)):
        slide_info = slides_info[slides_info.Slide_name == slide_name]
        if slide_name not in slides_indices.keys():
            continue
        indices = np.array(slides_indices[slide_name])

        mag, h, w = slide_info['Magnification'].values[0], slide_info['Height'].values[0], slide_info['Width'].values[0]
        extract_patch_size = int(patch_size * mag / target_mag)

        thumbnail = io.imread('../data/20x/thumbnail/{:s}_thumbnail.png'.format(slide_name))
        N_patch_row = h // extract_patch_size
        # N_patch_col = w // extract_patch_size
        stride = int(float(thumbnail.shape[0]) / N_patch_row)

        if slides_probs is None:
            probs = np.ones(indices.shape).astype(np.float)
            color_mask = np.zeros(thumbnail.shape, dtype=np.float)
        else:
            probs = slides_probs[slide_name]
            color_mask = np.ones(thumbnail.shape, dtype=np.float)

        for j in range(0, thumbnail.shape[1], stride):
            for i in range(0, thumbnail.shape[0], stride):
                index = (j//stride) * N_patch_row + (i//stride) + 1
                if index in indices:
                    prob = probs[indices == index]
                    color_mask[i:i + stride, j:j + stride, :] = np.array(color) / 255 * prob

        if on_thumbnail:
            thumbnail = thumbnail.astype(np.float) / 255
            result = thumbnail * color_mask
            # result = 0.5 * thumbnail + 0.5 * color_mask
        else:
            result = color_mask

        if slides_labels is not None:
            label = int(slides_labels[slide_name])
            io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, slide_name, label, postfix), (result*255).astype(np.uint8))
        else:
            io.imsave('{:s}/{:s}_{:s}.png'.format(save_dir, slide_name, postfix), (result*255).astype(np.uint8))


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)

    # set up logger for each result
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch   Train_loss  train_acc  || Test_loss   test_acc  test_auc')

    return logger, logger_results


def get_thumbnails():
    import os
    from openslide import OpenSlide

    data_dir = '/media/hui/Local Disk1/work/Data/Liver_Multi_Modality/Pathology/SlideImages/20-1018 PURU_VINOD'
    save_dir = '/media/hui/Local Disk1/work/Data/Liver_Multi_Modality/Pathology/thumbnails'
    os.makedirs(save_dir, exist_ok=True)
    filelist = os.listdir(data_dir)

    for file in filelist:
        wsi_file = OpenSlide('{:s}/{:s}'.format(data_dir, file))
        thumbnail = wsi_file.get_thumbnail((1000, 1000))
        thumbnail.save('{:s}/{:s}.png'.format(save_dir, file.split('.')[0]))
