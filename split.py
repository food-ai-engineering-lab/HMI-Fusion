import os
from sklearn.model_selection import train_test_split

DATASET_DIR = '/mnt/projects/bhatta70/VBNC-Detection/outputs/Salmonella_serovars_rgb_full_normalized'

TRAIN_DIR = '/mnt/projects/bhatta70/VBNC-Detection/data_rgb/train/'
TEST_DIR = '/mnt/projects/bhatta70/VBNC-Detection/data_rgb/test/'
TEST_SIZE = 0.3
RANDOM_STATE = 42


def move_file(fname, id,  train=False):
    classname = fname.split('/')[-2]

    img_fnam =  fname + '.png'
    mask_fnam = fname + '_mask.npy'
    spec_fnam =fname + '_spectra_mean.npy'
    single_spec_fnam = fname + '_single_cell_spectra.npy'

    new_fname = classname + '_' + str(id)
    if train:
        new_fname += '_train'
    else:
        new_fname += '_test'


   # copy to train or test directory
    destination = os.path.join(TRAIN_DIR, classname) if train else TEST_DIR
    os.makedirs(destination, exist_ok=True)
    cmd = f'cp {img_fnam} {destination}/{new_fname}.png'
    os.system(cmd)
    cmd = f'cp {mask_fnam} {destination}/{new_fname}_mask.npy'
    os.system(cmd)
    cmd = f'cp {spec_fnam} {destination}/{new_fname}_spectra_mean.npy'
    os.system(cmd)
    cmd = f'cp {single_spec_fnam} {destination}/{new_fname}_single_cell_spectra.npy'
    os.system(cmd)


def split_files(base_dir, test_size=.2, random_state=42):
    classnames = os.listdir(base_dir)
    sample_count = {classname: [0,0] for classname in classnames}
    for classname in classnames:
        class_dir = os.path.join(base_dir, classname)
        if not os.path.isdir(class_dir):
            continue
        files = [fname.replace('.png', '') for fname in os.listdir(class_dir) if fname.endswith('.png')]
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
        for i, fname in enumerate(train_files):
            move_file(os.path.join(class_dir, fname), i, train=True)
        for i, fname in enumerate(test_files):
            move_file(os.path.join(class_dir, fname), i, train=False)
        sample_count[classname] = [len(train_files), len(test_files)]
    print('Files split into train and test sets')
    for classname, counts in sample_count.items():
        print(f'{classname}: {counts[0]} train samples and {counts[1]} test samples')



if __name__ == '__main__':
    split_files(DATASET_DIR, test_size=TEST_SIZE, random_state=RANDOM_STATE)