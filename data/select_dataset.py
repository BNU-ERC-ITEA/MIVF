'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# updated by Wang Jq, 2025.7.17
# --------------------------------------------
'''

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # -----------------------------------------
    # video restoration
    # -----------------------------------------
    if dataset_type in ['videorecurrenttraindataset']:
        from data.dataset_video_train import VideoRecurrentTrainDataset as D
    elif dataset_type in ['videorecurrenttestdataset']:
        from data.dataset_video_test import VideoRecurrentTestDataset as D
    elif dataset_type in ['singlevideorecurrenttestdataset']:
        from data.dataset_video_test import SingleVideoRecurrentTestDataset as D
    elif dataset_type in ['videotestvimeo90kdataset']:
        from data.dataset_video_test import VideoTestVimeo90KDataset as D
    elif dataset_type in ['vfi_davis']:
        from data.dataset_video_test import VFI_DAVIS as D
    elif dataset_type in ['vfi_ucf101']:
        from data.dataset_video_test import VFI_UCF101 as D
    elif dataset_type in ['vfi_vid4']:
        from data.dataset_video_test import VFI_Vid4 as D

    # -----------------------------------------
    # Video Flare Removal with Masks
    # -----------------------------------------
    elif dataset_type in ['vfbm_train']:
        from data.dataset_video_train import VFBMTrainDataset as D
    elif dataset_type in ['vfbm_train_nocrop']:
        from data.dataset_video_train import VFBMTrainDataset_no_crop as D
    elif dataset_type in ['vfbm_test']:
        from data.dataset_video_test import VSBMTestDataset as D
    elif dataset_type in ['vfbm_test_mask_free']:
        from data.dataset_video_test import VideoRecurrentTestDataset as D

    # -----------------------------------------
    # Image Flare Removal
    # -----------------------------------------
    elif dataset_type in ['restormer_train']:
        from data.data_image_train import ImagePairedDataset as D
    elif dataset_type in ['restormer_test']:
        from data.data_image_test import ImagePairedDataset as D
    elif dataset_type in ['restormer_test_single']:
        from data.data_image_test import ImageDataset as D


    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
