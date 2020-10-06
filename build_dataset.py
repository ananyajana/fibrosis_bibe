import os, h5py
import numpy as np
from skimage import io
from tqdm import tqdm
from glob import glob
import pandas as pd
import json

def load_labels():
    df = pd.read_excel('AI_LiverBx_NAS_breakdown_20200206.xlsx', sheet_name='Sheet2')
    df = df.iloc[:, 1:]
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    patient_labels = {'ID': list(df['STUDY_ID'].values),
                      'Fibrosis': list(df['FIBROSIS'].values),
                      'NAS-total': list(df['NAS_TOTAL'].values),
                      'NAS-steatosis': list(df['NAS_STEATOSIS'].values),
                      'NAS-lob': list(df['NAS_LOB_INFL'].values),
                      'NAS-balloon': list(df['NAS_BALLOON'].values)}
    return patient_labels


def build_h5_file(patient_labels, original_data_dir1, original_data_dir2, save_dir, save_filename):
    dataset = h5py.File('{:s}/{:s}'.format(save_dir, save_filename), 'w')
    patients = patient_labels['ID']
    for i in tqdm(range(len(patients))):
    # for i in tqdm(range(0, 33)):
        pat = str(patients[i])
        if pat in ['033_A1', '048_B2']:
            continue
        fibrosis_label = patient_labels['Fibrosis'][i]
        nas_label = patient_labels['NAS-total'][i]
        nas_stea_label = patient_labels['NAS-steatosis'][i]
        nas_lob_label = patient_labels['NAS-lob'][i]
        nas_balloon_label = patient_labels['NAS-balloon'][i]
        assert nas_label == nas_stea_label + nas_lob_label + nas_balloon_label

        dataset.create_group('{:s}/HE'.format(pat))
        dataset.create_group('{:s}/Trichrome'.format(pat))
        dataset.create_dataset('{:s}/Fibrosis'.format(pat), data=fibrosis_label)
        dataset.create_dataset('{:s}/NAS_total'.format(pat), data=nas_label)
        dataset.create_dataset('{:s}/NAS_stea'.format(pat), data=nas_stea_label)
        dataset.create_dataset('{:s}/NAS_lob'.format(pat), data=nas_lob_label)
        dataset.create_dataset('{:s}/NAS_balloon'.format(pat), data=nas_balloon_label)

        HE_files = glob('{:s}/{:s}-HE*/*'.format(original_data_dir1, pat))
        Trichrome_files = glob('{:s}/{:s}-Trichrome*/*'.format(original_data_dir1, pat))
        if len(HE_files) == 0:
            HE_files = glob('{:s}/{:s}_HE/*'.format(original_data_dir2, pat))
            Trichrome_files = glob('{:s}/{:s}_TRI/*'.format(original_data_dir2, pat))

        for HE_file in HE_files:
            img_name = HE_file.split('/')[-1].split('.')[0]
            img = io.imread(HE_file)
            h, w, c = img.shape
            if h < 224:
                img = np.concatenate([img, np.ones((224-h, w, c), dtype=np.uint8)*255], axis=0)
            if w < 224:
                img = np.concatenate([img, np.ones((224, 224-w, c), dtype=np.uint8)*255], axis=1)

            dataset.create_dataset('{:s}/HE/{:s}'.format(pat, img_name), data=img)
        for Trichrome_file in Trichrome_files:
            img_name = Trichrome_file.split('/')[-1].split('.')[0]
            img = io.imread(Trichrome_file)
            h, w, c = img.shape
            if h < 224:
                img = np.concatenate([img, np.ones((224-h, w, c), dtype=np.uint8)*255], axis=0)
            if w < 224:
                img = np.concatenate([img, np.ones((224, 224-w, c), dtype=np.uint8)*255], axis=1)

            dataset.create_dataset('{:s}/Trichrome/{:s}'.format(pat, img_name), data=img)

    print((len(dataset.keys())))
    print(list(dataset.keys()))
    dataset.close()


def split_train_test(h5_filepath, save_dir):
    import random

    os.makedirs(save_dir, exist_ok=True)
    h5_file = h5py.File(h5_filepath, 'r')

    keys = list(h5_file.keys())
    indices = list(range(len(keys)))
    random.seed(2)
    random.shuffle(keys)

    N_fold = 3
    l = int(np.ceil(len(keys) / N_fold))

    # cross validation: fold i
    for fold in range(N_fold):
        print('Fold {:d}'.format(fold+1))
        end = l*(fold+1) if l*(fold+1) <= len(indices) else len(indices)
        test_indices = list(range(l*fold, end))
        train_indices = [idx for idx in indices if idx not in test_indices]
        print('test indices')
        print(sorted(test_indices))
        print('train indices')
        print(sorted(train_indices))

        train_file = h5py.File('{:s}/train{:d}.h5'.format(save_dir, fold+1), 'w')
        test_file = h5py.File('{:s}/test{:d}.h5'.format(save_dir, fold+1), 'w')
        for idx in train_indices:
            h5_file.copy(keys[idx], train_file)
        for idx in test_indices:
            h5_file.copy(keys[idx], test_file)
        train_file.close()
        test_file.close()

    h5_file.close()


def split_dataset_according_list(h5_filepath, train_list, test_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    h5_file = h5py.File(h5_filepath, 'r')

    keys = list(h5_file.keys())

    # cross validation: fold i
    for fold in range(len(train_list)):
        print('Fold {:d}'.format(fold+1))

        train_file = h5py.File('{:s}/train{:d}.h5'.format(save_dir, fold+1), 'w')
        test_file = h5py.File('{:s}/test{:d}.h5'.format(save_dir, fold+1), 'w')
        for id in train_list[fold]:
            h5_file.copy(id, train_file)
        for id in test_list[fold]:
            h5_file.copy(id, test_file)

        print(len(train_file))
        print(len(test_file))
        train_file.close()
        test_file.close()

    h5_file.close()

def get_fib_nas_from_data(h5_file):
    fib = []
    nas = []
    nas_stea = []
    nas_lob = []
    nas_balloon = []
    for key in h5_file.keys():
        fib.append(h5_file[key]['Fibrosis'][()])
        nas.append(h5_file[key]['NAS_total'][()])
        nas_stea.append(h5_file[key]['NAS_stea'][()])
        nas_lob.append(h5_file[key]['NAS_lob'][()])
        nas_balloon.append(h5_file[key]['NAS_balloon'][()])
    return fib, nas, nas_stea, nas_lob, nas_balloon

# patient_labels = load_labels()
# # np.save('../data/labels.npy', patient_labels)
#
# original_data_dir1 = ''
# original_data_dir2 = ''
# save_dir = ''
# # build_h5_file(patient_labels, original_data_dir1, original_data_dir2, save_dir, 'slides_20x_both_color_normed.h5')
# # split_train_test('{:s}/slides_20x_both_color_normed.h5'.format(save_dir), '../data_20x_both_color_normed')
#
# split_train_test('{:s}/slides_5x_both_color_normed_features_resnet101.h5'.format(save_dir), '../data_5x_both_normed_features_resnet101')


# train_list = [
#     ['1', '11', '12', '13', '17_2', '18', '20', '22', '25', '26', '27', '28', '29', '3', '32', '4', '5', '7', '8', '9'],
#     ['10', '11', '13', '14', '16', '17_2', '19', '2', '20', '21', '23', '25', '27', '28', '29', '3', '30', '31', '6', '8'],
#     ['1', '10', '12', '14', '16', '18', '19', '2', '21', '22', '23', '26', '30', '31', '32', '4', '5', '6', '7', '9']
# ]
#
# test_list = [
#     ['10', '14', '16', '19', '2', '21', '23', '30', '31', '6'],
#     ['1', '12', '18', '22', '26', '32', '4', '5', '7', '9'],
#     ['11', '13', '17_2', '20', '25', '27', '28', '29', '3', '8']
# ]
# split_dataset_according_list('{:s}/slides_5x_vsi.h5'.format(save_dir), train_list, test_list, '../data_vsi')

