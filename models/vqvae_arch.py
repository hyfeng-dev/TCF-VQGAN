import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .modules import Normalize, nonlinearity, SEGateBlock, DCTGateBlock, DCTResnetBlock, SEResnetBlock


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        with torch.no_grad():
            d = torch.cdist(z_flattened, self.embedding.weight)

        min_encoding_indices = torch.argmin(d, dim=1)
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).to(z)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # .........\end

        ## 对 z_q 约束小
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # 对 z 约束小
        # loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
        perplexity = torch.exp(entropy)

        # loss = loss - 0.1 * entropy
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # print(" * Perplexity: ", perplexity.item(), "Entropy: ", entropy.item())
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d)

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, decay=0.99, eps=1e-5, gamma=0.1, for_codebook=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.gamma = gamma
        self.for_codebook = for_codebook
        if self.training and self.for_codebook:
            print(" * Using EMA codebook update")
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding.weight.data.uniform_(-0.05, 0.05)
        self.embedding.weight.requires_grad = False

        self.register_buffer("_cluster_size", torch.zeros(n_e))
        self.register_buffer("_embed_avg", torch.zeros(n_e, e_dim))
        
    def forward(self, z: torch.Tensor):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        with torch.no_grad():
            d = torch.cdist(z_flattened, self.embedding.weight)

        min_encoding_indices = torch.argmin(d, dim=1)
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).to(z)

        if self.training and self.for_codebook:
            # 更新统计量
            with torch.no_grad():
                encodings_sum = min_encodings.sum(0)  # [n_e]
                embed_sum = torch.matmul(min_encodings.t(), z_flattened)  # [n_e, e_dim]

            self._cluster_size.mul_(self.decay).add_((1 - self.decay) * encodings_sum)
            self._embed_avg.mul_(self.decay).add_((1 - self.decay) * embed_sum)

            # laplace smoothing
            n = self._cluster_size.sum()
            cluster_size = (self._cluster_size + self.eps) * (n / (n + self.eps * self.n_e))

            # 更新码本
            embed_normalized = self._embed_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
        perplexity = torch.exp(entropy)

        # No entropy
        # loss = self.beta * F.mse_loss(z_q.detach(), z)
        # With entropy
        # loss = self.beta * F.mse_loss(z_q.detach(), z) - self.gamma * entropy
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # if self.training:
        #     print(" * Perplexity: ", perplexity.item(), "Entropy: ", entropy.item())
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d)


class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        conv_shorcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()      
        self.block = SEGateBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shorcut,
            dropout=dropout,
            temb_channels=temb_channels,
        )
    def forward(self, x, temb):
        h = self.block(x, temb)
        return h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            
            # 2×2 kernel, stride=2, padding=1
            # self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert (in_channels % head_size == 0), "The size of head should be divided by the number of channels."

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.num = 0

    def forward(self, x, y=None):
        h_ = x
        h_ = self.norm1(h_)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, self.head_size, self.att_size, h * w)
        q = q.permute(0, 3, 1, 2)  # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size, h * w)
        k = k.permute(0, 3, 1, 2)

        v = v.reshape(b, self.head_size, self.att_size, h * w)
        v = v.permute(0, 3, 1, 2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        scale = int(self.att_size) ** (-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)
        w_ = F.dropout(w_, p=0., training=self.training)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous()  # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2)

        w_ = self.proj_out(w_)

        return x + w_


class MultiHeadEncoder(nn.Module):
    def __init__(
        self,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=512,
        z_channels=256,
        double_z=True,
        enable_mid=True,
        head_size=1,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ConvBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ConvBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ConvBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        hs = {}
        temb = None

        h = self.conv_in(x)
        hs["in"] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                hs["block_" + str(i_level)] = h
                h = self.down[i_level].downsample(h)

        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs["block_" + str(i_level) + "_atten"] = h
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            hs["mid_atten"] = h

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        hs["out"] = h
        return hs


class MultiHeadDecoder(nn.Module):
    def __init__(
        self,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=16,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=512,
        z_channels=256,
        give_pre_end=False,
        enable_mid=True,
        head_size=1,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ConvBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ConvBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ConvBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class MultiHeadDecoderTransformer(nn.Module):
    def __init__(
        self,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=16,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=512,
        z_channels=256,
        give_pre_end=False,
        enable_mid=True,
        head_size=1,
        with_skip=True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid
        self.with_skip = with_skip

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        if self.with_skip:
            print(" * Using skip connections in decoder.")
            # skip connections
            self.skip_conv = nn.ModuleList()
            for channel_offset in reversed(range(self.num_resolutions)):
                if channel_offset == 0:
                    break
                self.skip_conv.append(nn.Conv2d(ch * ch_mult[channel_offset - 1], ch * ch_mult[channel_offset], kernel_size=1, stride=1, padding=0))
            self.skip_conv = nn.ModuleList(reversed(self.skip_conv))
        else:
            print(" * Not using skip connections in decoder.")
        
        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ConvBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ConvBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ConvBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z, hs):
        temb = None
        skip_connections = list(hs.values())[:-1]

        # z to block_in
        h = self.conv_in(z)
        # middle
        if self.enable_mid:
            if self.with_skip:
                # h = h + skip_connections[-1]
                pass
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h, hs["mid_atten"])
            h = self.mid.block_2(h, temb)

        skip_connections = skip_connections[:-1]     
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            
            # skip connection
            if self.with_skip:
                if i_level == self.num_resolutions - 1:
                    # h = h + skip_connections[i_level + 1]
                    pass
                else:
                    h = h + self.skip_conv[i_level](skip_connections[i_level + 1])
                
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    if "block_" + str(i_level) + "_atten" in hs:
                        h = self.up[i_level].attn[i_block](h, hs["block_" + str(i_level) + "_atten"])
                    else:
                        h = self.up[i_level].attn[i_block](h, hs["block_" + str(i_level)])
            if i_level != 0:
                h = self.up[i_level].upsample(h)
    
        if self.give_pre_end:
            return h

        if self.with_skip:
            h = h + skip_connections[0]
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQVAEGAN(nn.Module):
    def __init__(
        self,
        n_embed=1024,
        embed_dim=256,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=16,
        dropout=0.0,
        in_channels=3,
        resolution=512,
        z_channels=256,
        double_z=False,
        enable_mid=True,
        fix_decoder=False,
        fix_codebook=False,
        head_size=1,
        quantize_type="ste",
    ):
        super(VQVAEGAN, self).__init__()

        self.encoder = MultiHeadEncoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            enable_mid=enable_mid,
            head_size=head_size,
        )
        self.decoder = MultiHeadDecoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            enable_mid=enable_mid,
            head_size=head_size,
        )

        if quantize_type == "ste":
            print(" * Using straight-through estimator")
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        elif quantize_type == "ema":
            print(" * Using EMA quantization")
            self.quantize = VectorQuantizerEMA(n_embed, embed_dim, beta=0.25, decay=0.99, eps=1e-5, for_codebook=not fix_codebook)
        else:
            raise ValueError("Unknown quantize type: {}".format(quantize_type))

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        
        # initialize_weights(self)
        
    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs["out"])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        quant, diff, info, hs = self.encode(input)
        dec = self.decode(quant)

        return dec, diff, info, hs


class VQVAEGANMultiHeadTransformer(nn.Module):
    def __init__(
        self,
        n_embed=1024,
        embed_dim=256,
        ch=64,
        out_ch=3,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        in_channels=3,
        resolution=512,
        z_channels=256,
        double_z=False,
        enable_mid=True,
        fix_decoder=False,
        fix_codebook=True,
        fix_encoder=False,
        head_size=4,
        quantize_type="ste",
        ex_multi_scale_num=1,
    ):
        super(VQVAEGANMultiHeadTransformer, self).__init__()

        self.encoder = MultiHeadEncoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            enable_mid=enable_mid,
            head_size=head_size,
        )
        for i in range(ex_multi_scale_num):
            attn_resolutions = [attn_resolutions[0], attn_resolutions[-1] * 2]
        self.decoder = MultiHeadDecoderTransformer(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            enable_mid=enable_mid,
            head_size=head_size,
            with_skip=True
        )

        if quantize_type == "ste":
            print(" * Using straight-through estimator")
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        elif quantize_type == "ema":
            print(" * Using EMA quantization")
            self.quantize = VectorQuantizerEMA(n_embed, embed_dim, beta=0.25, decay=0.99, eps=1e-5)
        else:
            raise ValueError("Unknown quantize type: {}".format(quantize_type))

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            print(" * Fixing decoder")
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            print(" * Fixing encoder")
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False

        # initialize_weights(self)

    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs["out"])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant, hs):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, hs)

        return dec

    def forward(self, input):
        quant, diff, info, hs = self.encode(input)
        dec = self.decode(quant, hs)

        return dec, diff, info, hs
    

def initialize_weights(model, method="kaiming"):
    for m in model.modules():
        # 卷积层
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if method == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
            elif method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        # 线性层
        elif isinstance(m, nn.Linear):
            if method == "kaiming":
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        # 归一化层
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

