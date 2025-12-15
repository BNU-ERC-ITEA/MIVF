
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'image_crop':  # one crop image input L
        from models.model_image_crop import ModelCropImageRestoration as M

    elif model == 'model_restormer':  # one image input L, for restormer
        from models.model_restormer import Model_Restormer as M

    elif model == 'basicvsrpp':  # one image input L, for restormer
        from models.model_basicvsrpp import ModelBasicvsrpp as M


    elif model == 'mambavr':  # one image input L, for mambavr
        from models.model_mamba import ModelMambaVR as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
