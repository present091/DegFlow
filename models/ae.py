from torch import nn
from einops import rearrange
from torch import nn

import numbers
import torch
import torch.nn.functional as F


##########################################################################
# Normalization


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

##########################################################################


class ResidualBlock(nn.Module) :
    def __init__(self, in_dim, out_dim, padding_mode="zeros") :
        super(ResidualBlock, self).__init__()
        
        self.conv0 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)
        self.convOut = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False) if in_dim != out_dim else nn.Identity()
        
        self.norm0 = LayerNorm(in_dim, "WithBias")
        self.norm1 = LayerNorm(out_dim, "WithBias")
        
        self.act = nn.GELU()

    def forward(self, inp, use_skip=True) :
        x = inp
        x = self.conv0(self.act(self.norm0(x)))
        x = self.conv1(self.act(self.norm1(x)))    

        return x + self.convOut(inp) if use_skip else x


class ResidualBlockModule(nn.Module) :
    def __init__(self, in_dim, out_dim, isDown) :
        super(ResidualBlockModule, self).__init__()
        
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1) if isDown else nn.Identity()
        self.RB0 = ResidualBlock(out_dim, out_dim)
        self.RB1 = ResidualBlock(out_dim, out_dim)

        self.beta = nn.Parameter(torch.zeros((1, out_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, out_dim, 1, 1)), requires_grad=True)

    def forward(self, inp) :
        inp = self.down(inp)

        x = inp
        x = self.RB0(x)
        y = inp + self.beta * x
        x = self.RB1(y)
        x = y + self.gamma * x
        
        return x
    

class Upsample(nn.Module) :
    def __init__(self, in_dim, out_dim, isUp) :
        super(Upsample, self).__init__()
        
        self.isUp = isUp
                
        self.RB0 = ResidualBlock(in_dim, in_dim)
        self.RB1 = ResidualBlock(in_dim, in_dim)
        self.up = nn.Sequential(nn.Conv2d(in_dim, in_dim*4, kernel_size=1), 
                                nn.PixelShuffle(2),
                                nn.Conv2d(in_dim, out_dim, kernel_size=1)
                                ) if self.isUp else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, in_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_dim, 1, 1)), requires_grad=True)

    def forward(self, inp) :
        x = inp
        x = self.RB0(x)
        y = inp + self.beta * x
        x = self.RB1(y)
        x = y + self.gamma * x

        x = self.up(x)

        return x



class Encoder(nn.Module) :
    def __init__(self, in_dim, base_dim, latent_dim) :
        super(Encoder, self).__init__()

        self.stem = nn.Conv2d(in_dim, base_dim, kernel_size=3, stride=1, padding=1)

        self.EB0 = ResidualBlockModule(base_dim, base_dim, False)
        self.EB1 = ResidualBlockModule(base_dim, base_dim, True)
        self.EB2 = ResidualBlockModule(base_dim, base_dim*2, True)
        self.EB3 = ResidualBlockModule(base_dim*2, base_dim*4, True)


        self.conv_out = torch.nn.Conv2d(base_dim*4,
                                        latent_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
    def forward(self, inp):
        x = self.stem(inp)

        x = self.EB0(x)
        x = self.EB1(x)
        x = self.EB2(x)
        x = self.EB3(x)

        x = self.conv_out(x)
        
        return x


class Decoder(nn.Module) :
    def __init__(self, in_dim, base_dim, latent_dim) :
        super().__init__()

        self.conv_in = torch.nn.Conv2d(latent_dim,
                                       base_dim*4,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.DB0 = Upsample(base_dim*4, base_dim*2, isUp=True)
        self.DB1 = Upsample(base_dim*2, base_dim, isUp=True)
        self.DB2 = Upsample(base_dim, base_dim, isUp=True)
        self.DB3 = Upsample(base_dim, base_dim, isUp=False)

        self.refine = nn.Sequential(nn.GELU(),
                                    nn.Conv2d(base_dim, in_dim, kernel_size=3, padding=1, padding_mode="zeros"))
        
    def forward(self, inp) :
        x = self.conv_in(inp)
        x = self.DB0(x)
        x = self.DB1(x)
        x = self.DB2(x)
        x = self.DB3(x)   
        
        x = self.refine(x)
        
        return x
    

class Encoder_skip(nn.Module):
    def __init__(self, in_dim, base_dim, latent_dim) :
        super(Encoder_skip, self).__init__()

        self.stem = nn.Conv2d(in_dim, base_dim, kernel_size=3, stride=1, padding=1)
        
        self.EB0 = ResidualBlockModule(in_dim, base_dim, False)
        self.EB1 = ResidualBlockModule(base_dim, base_dim, True)
        self.EB2 = ResidualBlockModule(base_dim, base_dim*2, True)
        self.EB3 = ResidualBlockModule(base_dim*2, base_dim*4, True)

        self.conv_out = torch.nn.Conv2d(base_dim*4,
                                        latent_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = self.stem(x)
        f0 = self.EB0(x)
        f1 = self.EB1(f0)
        f2 = self.EB2(f1)
        x = self.EB3(f2)
        z = self.conv_out(x)
        return z, (f0, f1, f2)

class Decoder_skip(nn.Module):
    def __init__(self, in_dim, base_dim, latent_dim) :
        super().__init__()

        self.conv_in = torch.nn.Conv2d(latent_dim,
                                       base_dim*4,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.DB0 = Upsample(base_dim*4, base_dim*2, isUp=True)
        self.conv_cat2 = nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=1)
        self.DB1 = Upsample(base_dim*2, base_dim, isUp=True)
        self.conv_cat1 = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.DB2 = Upsample(base_dim, base_dim, isUp=True)
        self.conv_cat0 = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.DB3 = Upsample(base_dim, base_dim, isUp=False)

        self.refine = nn.Sequential(nn.GELU(),
                                    nn.Conv2d(base_dim, in_dim, kernel_size=3, padding=1, padding_mode="zeros"))

    def forward(self, z, features):
        f0, f1, f2 = features
        x = self.conv_in(z)
        x = self.DB0(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv_cat2(x)

        x = self.DB1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv_cat1(x)

        x = self.DB2(x)
        x = torch.cat([x, f0], dim=1)
        x = self.conv_cat0(x)

        x = self.DB3(x)
        x = self.refine(x)
        return x


class Autoencoder(nn.Module) :
    def __init__(self, in_dim, base_dim, latent_dim, use_skip) :
        super().__init__()
        if use_skip:
            self.encoder = Encoder_skip(in_dim, base_dim, latent_dim)
            self.decoder = Decoder_skip(in_dim, base_dim, latent_dim)
        else:
            self.encoder = Encoder(in_dim, base_dim, latent_dim)
            self.decoder = Decoder(in_dim, base_dim, latent_dim)
