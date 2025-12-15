import os
print(os.getcwd())

import sys
print(sys.path)
sys.path.append(os.getcwd())
import glob
from utils.utils_video import scandir
from utils.utils_lmdb import make_lmdb_from_imgs




def create_lmdb_flare():
    """Create lmdb files for DVD dataset.

    Usage:
        We take two folders for example:
            GT
            input
        Remember to modify opt configurations according to your settings.
    """
    # train_reflective_flare_free
    folder_path = r'/wangjunqiao/temp/autodl_data_240p/trainset/hq'
    lmdb_path = r'/wangjunqiao/temp/autodl_data_240p/vflare_240p/train_gt.lmdb'
    img_path_list, keys = prepare_keys_flare(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_with_reflective_flare
    folder_path = r'/wangjunqiao/temp/autodl_data_240p/trainset/lq'
    lmdb_path = r'/wangjunqiao/temp/autodl_data_240p/vflare_240p/train_lq.lmdb'
    img_path_list, keys = prepare_keys_flare(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_flare(folder_path):
    """Prepare image path list and keys for DVD dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def generate_meta_info_txt(data_path, meta_info_path):
    '''
    generate meta_info.txt for VFLARE
    :param data_path: dataset path.
    :return: None
    '''

    f = open(meta_info_path, "w+")
    file_list = sorted(glob.glob(os.path.join(data_path, '*')))
    total_frames = 0
    for path in file_list:
        name = os.path.basename(path)
        frames = sorted(glob.glob(os.path.join(path, '*')))
        start_frame = os.path.basename(frames[0]).split('.')[0]

        # print(name, len(frames), start_frame)
        total_frames += len(frames)

        f.write(f"{name} {len(frames)} (240,320,3) {start_frame}\r\n")

    # assert total_frames == 15360, f'DVD training+Validation set should have 6708 images, but got {total_frames} images'


def main():
    print(sys.version)
    create_lmdb_flare()


if __name__ == '__main__':
    # main()
    # mv /wangjunqiao/temp/total_data_spilt/train/vflare_v2/ /wangjunqiao/vflare_removal/code/trainsets/
    generate_meta_info_txt(r'/wangjunqiao/temp/autodl_data_240p/trainset/hq', r'/wangjunqiao/temp/autodl_data_240p/vflare_240p/train_gt.lmdb/meta_info_4.txt')