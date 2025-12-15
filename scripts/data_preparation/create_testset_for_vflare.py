import os
import shutil
import random


def get_subdirectories(directory):
    """
    获取目录下的所有子文件夹的相对路径。

    :param directory: 目录路径
    :return: 子文件夹的相对路径列表
    """
    sub_dirs = os.listdir(directory)
    return sub_dirs


def copy_subdirectories(src_dir, dest_dir, subdirs):
    """
    复制指定子文件夹到目标目录。

    :param src_dir: 源目录路径
    :param dest_dir: 目标目录路径
    :param subdirs: 要复制的子文件夹相对路径列表
    """
    os.makedirs(dest_dir, exist_ok=True)
    for subdir in subdirs:
        src_path = os.path.join(src_dir, subdir)
        dest_path = os.path.join(dest_dir, subdir)

        os.makedirs(dest_path, exist_ok=True)
        for item in os.listdir(src_path):
            src_item_path = os.path.join(src_path, item)
            dest_item_path = os.path.join(dest_path, item)

            if os.path.isdir(src_item_path):
                shutil.copytree(src_item_path, dest_item_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item_path, dest_item_path)


def main():
    source_directory_A = '/wangjunqiao01/Vflare_removal/data/WithFlare_Reds_use_flare-R/only_with_sourcelight'
    source_directory_B = '/wangjunqiao01/Vflare_removal/data/WithFlare_Reds_use_flare-R/with_scattering_and_reflection_light'
    destination_directory_C = '/wangjunqiao01/Vflare_removal/project_code/KAIR/testsets/vflare/test_origal'
    destination_directory_D = '/wangjunqiao01/Vflare_removal/project_code/KAIR/testsets/vflare/test_withflare'


    number_of_folders = 400  # 需要随机选取的子文件夹数量
    random_seed = 2  # 设置随机数种子

    # 设置随机数种子
    random.seed(random_seed)

    # 获取所有子文件夹的相对路径
    sub_dirs_A = get_subdirectories(source_directory_A)
    sub_dirs_B = get_subdirectories(source_directory_B)


    # 确保 A 和 B 的子文件夹列表一致
    assert set(sub_dirs_A) == set(sub_dirs_B), "Subdirectories in A and B do not match"

    # 随机选择 n 个子文件夹
    selected_dirs = random.sample(sub_dirs_A, number_of_folders)

    # 打印选择的子文件夹路径
    print("Selected directories:")
    for dir_path in selected_dirs:
        print(dir_path)

    # 将 A 中的子文件夹复制到 C
    copy_subdirectories(source_directory_A, destination_directory_C, selected_dirs)

    # 将 B 中的子文件夹复制到 D
    copy_subdirectories(source_directory_B, destination_directory_D, selected_dirs)


def test():
    a = os.listdir('/wangjunqiao01/Vflare_removal/project_code/KAIR/testsets/vflare/test_origal')
    b = os.listdir('/wangjunqiao01/Vflare_removal/project_code/KAIR/testsets/vflare/test_withflare')
    print(len(a))
    print(len(b))
    assert set(a) == set(b), "Subdirectories in A and B do not match"


if __name__ == '__main__':
    # main()
    test()
