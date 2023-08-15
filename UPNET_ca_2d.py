"""Real-time Spatial Temporal Transformer.
""" 
import numpy as np
import torch
import functools
import math
import einops
import torch.nn as nn
import torch.nn.functional as F
from .layers_ca_fcb_a3 import (make_layer)
from .swin_utils_hfnet_2d import Swin_backbone_unConv,Swin_backbone_unConv_z,Swin_backbone_unConv_d,InputProj, Downsample, Upsample
from timm.models.layers import trunc_normal_


def positionalencoding1d(d_model, length, ratio):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1) * ratio
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class RSTT(nn.Module):
    def __init__(self, in_chans=1, embed_dim=16,
                 depths=[2, 2, 4, 4, 2, 2], 
                 num_heads=[4, 4, 4, 4, 4, 4], num_frames=4,
                 window_sizes=[4, 4, 4, 4, 4, 4], 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, 
                 back_RBs=0):
        """

        Args:
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            embed_dim (int, optional): Number of projection output channels. Defaults to 32.
            depths (list[int], optional): Depths of each Transformer stage. Defaults to [2, 2, 2, 2, 2, 2, 2, 2].
            num_heads (list[int], optional): Number of attention head of each stage. Defaults to [2, 4, 8, 16, 16, 8, 4, 2].
            num_frames (int, optional): Number of input frames. Defaults to 4.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            back_RBs (int, optional): Number of residual blocks for super resolution. Defaults to 10.
        """
        super().__init__()

        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.scale = 2 ** (self.num_enc_layers - 1)
        dec_depths = depths[self.num_enc_layers:]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_in_frames = num_frames
        self.num_out_frames = 5 * num_frames - 4
        self.c_last = 8
        img_size = 256
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth 
        enc_dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        dec_dpr = enc_dpr[::-1]

        self.input_proj = InputProj(in_channels=in_chans, embed_dim=embed_dim,
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_enc_layers):
            encoder_layer = Swin_backbone_unConv_z(img_size=img_size//(2**i_layer), embed_dim=self.num_in_frames*embed_dim,
                                             depth=[8,8,8,8], num_heads=[8,8,8,8],window_size=window_sizes[i_layer],
                                            mlp_ratio=mlp_ratio)
            # encoder_layer = EncoderLayer(
            #     dim=embed_dim,
            #     depth=depths[i_layer], num_heads=num_heads[i_layer],
            #     num_frames=num_frames, window_size=window_sizes[i_layer], mlp_ratio=mlp_ratio,
            #     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            #     attn_drop=attn_drop_rate,
            #     drop_path=enc_dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #     norm_layer=norm_layer, num_in_frames=self.num_in_frames
            # )
            downsample = Downsample(self.num_in_frames*embed_dim, self.num_in_frames*embed_dim)
            self.encoder_layers.append(encoder_layer)
            self.downsample.append(downsample)
        
        # postencoding
        
        
        self.positions_z = positionalencoding1d(self.embed_dim, self.num_out_frames, 1).cuda()
        self.positions_z = self.positions_z.unsqueeze(2).unsqueeze(2)
        
        slice_list = list(range(self.num_out_frames))
        vis_list = slice_list[:num_frames]
        mask_list = slice_list[num_frames:]
        re_list = []

        while len(vis_list) != 0:
            if len(re_list) % 5 == 0:
                re_list.append(vis_list.pop(0))
            else:
                re_list.append(mask_list.pop(0))

        self.slice_sequence = torch.tensor(re_list)
        
        self.D_patch = 8
        img_size = 64
        T_embed = self.embed_dim * self.D_patch
        img_size_l = 256
        I_embed = self.embed_dim
        exec('''self.Decoder_T = Swin_backbone_unConv(img_size=(self.num_out_frames, img_size), embed_dim=T_embed,
                                         depths=[8,8,8,8], num_heads=[8,8,8,8], window_size=4,
                                         mlp_ratio=4)''')
        exec('''self.Decoder_I = Swin_backbone_unConv_z(img_size=img_size, embed_dim=I_embed,
                                            depths=[4,4], num_heads=[8,8], window_size=8,
                                            mlp_ratio=4)''')
        

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.cat_channel = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            decoder_layer = Swin_backbone_unConv_d(img_size=img_size//(2**(self.num_enc_layers-i_layer-1)), embed_dim=self.num_out_frames*embed_dim+self.num_in_frames*embed_dim,
                                             depth=[8,8,8,8], num_heads=[8,8,8,8],window_size=window_sizes[i_layer],
                                            mlp_ratio=mlp_ratio)
            # embed_dim=self.num_out_frames*embed_dim+self.num_in_frames*embed_dim,
            #decoder_layer = DecoderLayer(
            #         dim=embed_dim,
            #         depth=depths[i_layer + self.num_enc_layers],
            #         num_heads=num_heads[i_layer + self.num_enc_layers],
            #         num_frames=num_frames, window_size=window_sizes[i_layer + self.num_enc_layers], mlp_ratio=mlp_ratio,
            #         qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            #         attn_drop=attn_drop_rate,
            #         drop_path=dec_dpr[sum(dec_depths[:i_layer]):sum(dec_depths[:i_layer + 1])],
            #         norm_layer=norm_layer,num_out_frames=self.num_out_frames
            # )
            cat_channel = nn.Conv2d((self.num_out_frames+self.num_in_frames)*embed_dim,self.num_out_frames*embed_dim,1,1,0)
            self.decoder_layers.append(decoder_layer)
            self.cat_channel.append(cat_channel)
            if i_layer != self.num_dec_layers - 1:
                upsample = Upsample(self.num_out_frames*embed_dim, self.num_out_frames*embed_dim)
                self.upsample.append(upsample)


        
        # Reconstruction block
        #ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=embed_dim)
        #self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        # Upsampling
        # self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(embed_dim, 64 * 4, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        #self.HRconv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)
        #self.conv_last = nn.Conv2d(embed_dim, 1, 3, 1, 1, bias=True)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

        self.conv_before_upsample = nn.Sequential(nn.Conv3d(self.embed_dim, 16, 1, 1, 0),
                                                  nn.LeakyReLU(inplace=True))
        self.conv_last_3d = nn.Conv3d(16, 1, 1, 1, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def cal_xy(self, x, D,h,w):
        x_in = x.reshape(-1, self.embed_dim, h, w)

        x_in_sag = einops.rearrange(x_in, 'dn c h (wn wp) -> wn (c wp) dn h', wp=self.D_patch)
        x_out_sag = D.forward_features(x_in_sag)
        x_out_sag = einops.rearrange(x_out_sag, 'wn (c wp) dn h -> dn c h (wn wp)', wp=self.D_patch)

        x_in_cor = einops.rearrange(x_in, 'dn c (hn hp) w -> hn (c hp) dn w', hp=self.D_patch)
        x_out_cor = D.forward_features(x_in_cor)
        x_out_cor = einops.rearrange(x_out_cor, 'hn (c hp) dn w -> dn c (hn hp) w', hp=self.D_patch)

        x_out = x_out_sag + x_out_cor
        x_out = x_out.unsqueeze(0)

        return x_out
    


    def forward(self, x):
        # 1 D H W
        #print(x.shape)#torch.Size([1, 1, 4, 256, 256])
        x = x.permute(0, 2, 1, 3, 4)
        B, D, C, H, W = x.size()  # 1 d 1 h w
        x = x.permute(0, 2, 1, 3, 4) # B C D H W
        upsample_x = F.interpolate(x, (5*D-4, H, W), mode='trilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3, 4)
        #print(x.shape)

        x = self.input_proj(x) # B, D, C, H, W
        #print(x.shape)
        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))
        x = x.reshape(1, -1, H, W)
        encoder_features = []
        for i_layer in range(self.num_enc_layers):
            #print(x.shape)
            x = self.encoder_layers[i_layer].forward_features(x)
            #print(x.shape)
            encoder_features.append(x)
            #print(x.shape)
            if i_layer != self.num_enc_layers - 1:
                x = self.downsample[i_layer](x)
        
        #print(x.shape)
        _, _, h, w = x.size()
        x = x.reshape(1, D, self.embed_dim, h, w)
        _, _, C, h, w = x.size()
        # TODO: Use interpolation for queries
        x_patch_vis = x.squeeze(0)
        # #print(x_patch_vis.shape)
        ###x_patch_mask = torch.nn.Parameter(torch.zeros(self.num_out_frames - self.num_in_frames, self.embed_dim, h, w)).cuda()
        ###x_patch_embed = torch.cat([x_patch_vis, x_patch_mask], dim=0)
        ###y = x_patch_embed[self.slice_sequence].unsqueeze(0)
        y = torch.zeros((B, self.num_out_frames, C, h, w), device=x.device)
        for i in range(self.num_out_frames):
            if i % 5 == 0:
                y[:, i, :, :, :] = x[:, i//5]
            else:
                y[:, i, :, :, :] = ((1-((i//5)/5))*x[:, i//5] + ((i//5)/5)*x[:, i//5 + 1]) / 2
                
        
        y = y + self.positions_z.unsqueeze(0)
        #y = self.cal_xy(y, self.Decoder_T,h,w)
        y = y.reshape(1, -1, h, w)
        #print(y.shape)
        for i_layer in range(self.num_dec_layers):
            #print(y.shape)
            #print(encoder_features[-i_layer - 1].shape)
            y = torch.cat([y, encoder_features[-i_layer - 1]], 1)
            #y = self.cat_channel[i_layer](y)
            y = self.decoder_layers[i_layer].forward_features(y)
            y = self.cat_channel[i_layer](y)
            if i_layer != self.num_dec_layers - 1:
                y = self.upsample[i_layer](y)

        
        trans_output = y.reshape(1, C,self.num_out_frames, H, W)
        #outs = y.permute(0, 2, 1, 3, 4)# B, c, D, H, W
        outs = self.conv_last_3d(self.conv_before_upsample(trans_output))
        
        return outs[:, :, 3:-3]
