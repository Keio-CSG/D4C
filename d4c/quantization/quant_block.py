import torch.nn as nn
import torch
import math
from clip.model import Bottleneck, AttentionPool2d, ModifiedResNet, ResidualAttentionBlock, Transformer, VisionTransformer, CLIP
from d4c.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer


class QuantBottleneck(QuantizedBlock):
    """
    Quantized Bottleneck Block used in ResNet for CLIP.
    """
    def __init__(self, org_module: Bottleneck, w_qconfig, a_qconfig):
        super().__init__()
        self.conv1 = QuantizedLayer(org_module.conv1, None, w_qconfig, a_qconfig)
        self.bn1 = org_module.bn1
        self.relu1 = org_module.relu1

        self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig)
        self.bn2 = org_module.bn2
        self.relu2 = org_module.relu2

        self.avgpool = org_module.avgpool

        self.conv3 = QuantizedLayer(org_module.conv3, None, w_qconfig, a_qconfig)
        self.bn3 = org_module.bn3
        self.relu3 = org_module.relu3

        self.downsample = org_module.downsample if org_module.downsample is not None else None

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class QuantAttentionPool2d(nn.Module):
    """
    Quantized AttentionPool2d Block used in ResNet for CLIP.
    """
    def __init__(self, org_module: AttentionPool2d, w_qconfig, a_qconfig):
        super().__init__()
        self.positional_embedding = org_module.positional_embedding
        self.num_heads = org_module.num_heads

        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig)
        self.c_proj = QuantizedLayer(org_module.c_proj, None, w_qconfig, a_qconfig)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.softmax_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        x = x.permute(1, 0, 2)
        q = self.q_proj(x[:, :1, :]).permute(1, 0, 2)
        k = self.k_proj(x).permute(1, 0, 2)
        v = self.v_proj(x).permute(1, 0, 2)

        L, N, C = q.shape
        head_dim = C // self.num_heads

        def shape_projection(t):
            L_local, N_local, C_local = t.shape
            head_dim_local = C_local // self.num_heads
            t = t.view(L_local, N_local, self.num_heads, head_dim_local)
            t = t.permute(1, 2, 0, 3).contiguous()
            return t.view(N_local * self.num_heads, L_local, head_dim_local)

        q = shape_projection(q)
        k = shape_projection(k)
        v = shape_projection(v)

        scale_factor = 1 / math.sqrt(q.size(-1))

        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0, train=True)

        attn_weight = self.softmax_post_act_fake_quantize(attn_weight)

        attn_output = attn_weight @ v

        def reshape_output(t):
            t = t.view(N, self.num_heads, L, head_dim)
            t = t.permute(2, 0, 1, 3).contiguous()
            return t.view(L, N, C)

        attn_output = reshape_output(attn_output)
        attn_output = attn_output.permute(1, 0, 2)
        out = self.c_proj(attn_output).permute(1, 0, 2)

        return out.squeeze(0)


class QuantModifiedResNet(nn.Module):
    """
    Quantized ModifiedResNet for CLIP.
    """
    def __init__(self, org_module: ModifiedResNet, w_qconfig, a_qconfig):
        super().__init__()
        self.output_dim = org_module.output_dim
        self.input_resolution = org_module.input_resolution

        self.conv1 = org_module.conv1
        self.bn1 = org_module.bn1
        self.relu1 = org_module.relu1
        self.conv2 = org_module.conv2
        self.bn2 = org_module.bn2
        self.relu2 = org_module.relu2
        self.conv3 = org_module.conv3
        self.bn3 = org_module.bn3
        self.relu3 = org_module.relu3

        self.avgpool = nn.AvgPool2d(2)

        self.layer1 = self._replace_layer(org_module.layer1, w_qconfig, a_qconfig)
        self.layer2 = self._replace_layer(org_module.layer2, w_qconfig, a_qconfig)
        self.layer3 = self._replace_layer(org_module.layer3, w_qconfig, a_qconfig)
        self.layer4 = self._replace_layer(org_module.layer4, w_qconfig, a_qconfig)

        self.attnpool = QuantAttentionPool2d(org_module.attnpool, w_qconfig, a_qconfig)

    def _replace_layer(self, layer_seq, w_qconfig, a_qconfig):
        new_layer = []
        for block in layer_seq:
            new_layer.append(QuantBottleneck(block, w_qconfig, a_qconfig))
        return nn.Sequential(*new_layer)

    def forward(self, x):

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class QuantMultiheadAttention(QuantizedBlock):
    def __init__(self, org_module: nn.MultiheadAttention, w_qconfig, a_qconfig):
        """
        Quantized MultiheadAttention Block for CLIP.
        """
        super().__init__()
        self.embed_dim = org_module.embed_dim
        self.num_heads = org_module.num_heads
        self.dropout = org_module.dropout
        self.head_dim = self.embed_dim // self.num_heads

        use_bias = org_module.in_proj_bias is not None
        q_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        k_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        v_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        o_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)

        q_linear.weight.data.copy_(org_module.in_proj_weight[:self.embed_dim, :])
        k_linear.weight.data.copy_(org_module.in_proj_weight[self.embed_dim:2 * self.embed_dim, :])
        v_linear.weight.data.copy_(org_module.in_proj_weight[2 * self.embed_dim:, :])
        o_linear.weight.data.copy_(org_module.out_proj.weight)
        if use_bias:
            q_linear.bias.data.copy_(org_module.in_proj_bias[:self.embed_dim])
            k_linear.bias.data.copy_(org_module.in_proj_bias[self.embed_dim:2 * self.embed_dim])
            v_linear.bias.data.copy_(org_module.in_proj_bias[2 * self.embed_dim:])
            o_linear.bias.data.copy_(org_module.out_proj.bias)

        self.q_proj = QuantizedLayer(q_linear, None, w_qconfig, a_qconfig)
        self.k_proj = QuantizedLayer(k_linear, None, w_qconfig, a_qconfig)
        self.v_proj = QuantizedLayer(v_linear, None, w_qconfig, a_qconfig)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.softmax_post_act_fake_quantize = Quantizer(None, a_qconfig)

        self.out_proj = QuantizedLayer(o_linear, None, w_qconfig, a_qconfig)

    def forward(self, x, attn_mask=None):
        N, L, _ = x.shape

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        def shape_projection(t):
            t = t.view(N, L, self.num_heads, self.head_dim)  # [N, L, H, Hd]
            t = t.permute(0, 2, 1, 3).contiguous()  # [N, H, L, Hd]
            return t.view(N * self.num_heads, L, self.head_dim)  # [N*H, L, Hd]

        query = shape_projection(query)
        key = shape_projection(key)
        value = shape_projection(value)

        attn_bias = torch.zeros(query.size(-2), key.size(-2), dtype=query.dtype, device=query.device)
        if attn_mask is not None and attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(query.dtype)

        scale_factor = 1 / math.sqrt(query.size(-1))
        if attn_mask is not None:
            attn_bias = attn_mask + attn_bias

        query = self.q_post_act_fake_quantize(query)
        key = self.k_post_act_fake_quantize(key)
        value = self.v_post_act_fake_quantize(value)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, train=True)
        attn_weight = self.softmax_post_act_fake_quantize(attn_weight)

        attn_output = attn_weight @ value

        def reshape_output(t):
            t = t.view(N, self.num_heads, L, self.head_dim)  # [N, H, L, Hd]
            t = t.permute(0, 2, 1, 3).contiguous()  # [N, L, H, Hd]
            return t.view(N, L, self.embed_dim)

        attn_output = reshape_output(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output


class QuantMLPBlock(QuantizedBlock):
    """
    Quantized MLP Block used in Transformer for CLIP.
    """
    def __init__(self, org_module: nn.Sequential, w_qconfig, a_qconfig):
        super().__init__()
        self.c_fc = QuantizedLayer(org_module.c_fc, None, w_qconfig, a_qconfig)
        self.gelu = org_module.gelu
        self.c_proj = QuantizedLayer(org_module.c_proj, None, w_qconfig, a_qconfig)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QuantResidualAttentionBlock(nn.Module):
    """
    Quantized Residual Attention Block used in Transformer for CLIP.
    """
    def __init__(self, org_module, w_qconfig, a_qconfig):
        super().__init__()

        self.attn = QuantMultiheadAttention(org_module.attn, w_qconfig, a_qconfig)
        self.ln_1 = org_module.ln_1
        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig)
        self.ln_2 = org_module.ln_2
        self.attn_mask = org_module.attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class QuantTransformer(nn.Module):
    """
    Quantized Transformer for CLIP.
    """
    def __init__(self, org_module: Transformer, w_qconfig, a_qconfig):
        super().__init__()
        self.width = org_module.width
        self.layers = org_module.layers
        self.resblocks = nn.Sequential(*[QuantResidualAttentionBlock(org_module.resblocks[i], w_qconfig, a_qconfig) for i in range(self.layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class QuantVisionTransformer(nn.Module):
    """
    Quantized VisionTransformer for CLIP.
    """
    def __init__(self, org_module: VisionTransformer, w_qconfig, a_qconfig):
        super().__init__()
        self.input_resolution = org_module.input_resolution
        self.output_dim = org_module.output_dim
        self.conv1 = org_module.conv1

        self.class_embedding = org_module.class_embedding
        self.positional_embedding = org_module.positional_embedding
        self.ln_pre = org_module.ln_pre

        self.transformer = QuantTransformer(org_module.transformer, w_qconfig, a_qconfig)

        self.ln_post = org_module.ln_post
        self.proj = org_module.proj

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class QuantCLIP(nn.Module):
    """
    Quantized CLIP Model.
    """
    def __init__(self, org_module: CLIP, w_qconfig, a_qconfig):
        super().__init__()
        self.context_length = org_module.context_length

        if isinstance(org_module.visual, ModifiedResNet):
            self.visual = QuantModifiedResNet(org_module.visual, w_qconfig, a_qconfig)
        elif isinstance(org_module.visual, VisionTransformer):
            self.visual = QuantVisionTransformer(org_module.visual, w_qconfig, a_qconfig)
        self.transformer = QuantTransformer(org_module.transformer, w_qconfig, a_qconfig)

        self.vocab_size = org_module.vocab_size
        self.token_embedding = org_module.token_embedding
        self.positional_embedding = org_module.positional_embedding
        self.ln_final = org_module.ln_final

        self.text_projection = org_module.text_projection
        self.logit_scale = org_module.logit_scale

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


specials = {
    CLIP: QuantCLIP
}
