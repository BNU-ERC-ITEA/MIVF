import functools
import torch
from torch.nn import init

"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # deflare task
    # ----------------------------------------

    # ----------------------------------------
    # RVRT
    # ----------------------------------------
    if net_type == 'rvrt':
        from models.network_rvrt import RVRT as net
        netG = net(upscale=opt_net['upscale'],
                   clip_size=opt_net['clip_size'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   num_blocks=opt_net['num_blocks'],
                   depths=opt_net['depths'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   inputconv_groups=opt_net['inputconv_groups'],
                   spynet_path=opt_net['spynet_path'],
                   deformable_groups=opt_net['deformable_groups'],
                   attention_heads=opt_net['attention_heads'],
                   attention_window=opt_net['attention_window'],
                   # nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'],
                   cpu_cache_length=opt_net['cpu_cache_length'])

    # ----------------------------------------
    # RVRT FOR VFLARE_REMOVAL
    # ----------------------------------------

    elif net_type == 'rvrt_vfbm':
        from models.network_vflare_removal import RVRT_VFBM as net
        netG = net(upscale=opt_net['upscale'],
                   clip_size=opt_net['clip_size'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   num_blocks=opt_net['num_blocks'],
                   depths=opt_net['depths'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   inputconv_groups=opt_net['inputconv_groups'],
                   spynet_path=opt_net['spynet_path'],
                   deformable_groups=opt_net['deformable_groups'],
                   attention_heads=opt_net['attention_heads'],
                   attention_window=opt_net['attention_window'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'],
                   cpu_cache_length=opt_net['cpu_cache_length'])
        


    # ----------------------------------------
    # restormer
    # ----------------------------------------
    elif net_type == 'restormer':
        from models.network_restormer_vflare_removal_original import Restormer as net
        # netG = net(inp_channels=3,
        #            out_channels=3,
        #            dim=48,
        #            num_blocks=[4, 6, 6, 8],
        #            num_refinement_blocks=4,
        #            heads=[1, 2, 4, 8],
        #            ffn_expansion_factor=2.66,
        #            bias=False,
        #            LayerNorm_type='WithBias',  ## Other option 'BiasFree'
        #            dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6)
        #            )
        netG = net(inp_channels=opt_net['inp_channels'],
            out_channels=opt_net['out_channels'],
            dim=opt_net['dim'],
            num_blocks=opt_net['num_blocks'],
            num_refinement_blocks=opt_net['num_refinement_blocks'],
            heads=opt_net['heads'],
            ffn_expansion_factor=opt_net['ffn_expansion_factor'],
            bias=opt_net['bias'],
            LayerNorm_type=opt_net['LayerNorm_type'],  ## Other option 'BiasFree'
            dual_pixel_task=opt_net['dual_pixel_task']  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6)
            )
        
    # ----------------------------------------
    # basicvsrpp
    # ----------------------------------------
    elif net_type == 'basicvsrpp':
        from models.basicvsrpp.network_basicvsrpp_vflare_removal import BasicVSRPlusPlus as net
        netG = net(mid_channels=opt_net['mid_channels'],
            num_blocks=opt_net['num_blocks'],
            max_residue_magnitude=opt_net['max_residue_magnitude'],
            is_low_res_input=opt_net['is_low_res_input'],
            spynet_pretrained=opt_net['spynet_pretrained'],
            cpu_cache_length=opt_net['cpu_cache_length'],
            )

        
   
    # ----------------------------------------
    # ours
    # ----------------------------------------  
    elif net_type == 'mambavr':
        from models.MIVF.network_mivf import MambaVR as net
        netG = net(upscale=opt_net['upscale'],
            in_chans=opt_net['in_chans'],
            img_size=opt_net['img_size'],
            img_range=opt_net['img_range'],
            embed_dim=opt_net['embed_dim'],
            d_state=opt_net['d_state'],
            depths=opt_net['depths'],
            num_heads=opt_net['num_heads'],
            window_size=opt_net['window_size'],
            inner_rank=opt_net['inner_rank'],
            num_tokens=opt_net['num_tokens'],
            convffn_kernel_size=opt_net['convffn_kernel_size'],
            mlp_ratio=opt_net['mlp_ratio'],
            num_frames=opt_net['num_frames']
            )

    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG

"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition! (mmcv needed)')
