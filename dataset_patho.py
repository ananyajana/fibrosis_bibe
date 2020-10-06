import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
import torch


def is_hdf5_file(filename):
    return filename.lower().endswith('.h5')


def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())


def h5_loader(data):
    he_data = data['HE']
    trichrome_data = data['Trichrome']
    fib_score = data['Fibrosis'][()]
#     nas_total_score = data['NAS_total'][()]
    nas_stea_score = data['NAS_stea'][()]
    nas_lob_score = data['NAS_lob'][()]
    nas_balloon_score = data['NAS_balloon'][()]

    he_imgs = []
    trichrome_imgs = []
    for key in he_data.keys():
        img = he_data[key][()]
        he_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))
    for key in trichrome_data.keys():
        img = trichrome_data[key][()]
        trichrome_imgs.append(Image.fromarray(img.astype('uint8'), 'RGB'))

    if fib_score == 0:  # 0: 0
        fib_label = 0
    elif fib_score < 3:  # 1: [1, 2, 2.5]
        fib_label = 1
    else:               # 2: [3, 3.5, 4]
        fib_label = 2
    # fib_label = 0 if fib_score < 3 else 1

    if nas_total_score < 3:
        nas_total_label = 0
    elif nas_total_score >= 5:
        nas_total_label = 2
    else:
        nas_total_label = 1

    nas_stea_label = 0 if nas_stea_score < 2 else 1
    nas_lob_label = nas_lob_score if nas_lob_score < 2 else 2
    # nas_lob_label = 0 if nas_lob_score < 2 else 1
    nas_balloon_label = nas_balloon_score

    return he_imgs, trichrome_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label


class LiverDataset(data.Dataset):
    def __init__(self, hdf5_path, data_transform):
        super(LiverDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.data_transform = data_transform
        self.keys = get_keys(self.hdf5_path)

    def __getitem__(self, index):
        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[index]]
        he_imgs, trichrome_imgs, fib_label, nas_stea_label, nas_lob_label, nas_balloon_label = h5_loader(slide_data)
        he_tensor, trichrome_tensor = [], []
        for i in range(len(he_imgs)):
            he_tensor.append(self.data_transform(he_imgs[i]).unsqueeze(0))
        for i in range(len(trichrome_imgs)):
            trichrome_tensor.append(self.data_transform(trichrome_imgs[i]).unsqueeze(0))

        return torch.cat(he_tensor, dim=0), torch.cat(trichrome_tensor, dim=0), \
                torch.tensor(fib_label).unsqueeze(0).long(), torch.tensor(nas_stea_label).unsqueeze(0).long(), \
                torch.tensor(nas_lob_label).unsqueeze(0).long(), torch.tensor(nas_balloon_label).unsqueeze(0).long()

    def __len__(self):
        return len(self.keys)

