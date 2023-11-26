import math
from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import (
    PatchEmbed, Mlp, GluMlp, 
    SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat,
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, 
    resample_patch_embed, resample_abs_pos_embed, 
    to_2tuple, use_fused_attn,
)
from timm.models.eva import EvaAttention, EvaBlock, EvaBlockPostNorm, Eva

from .transformer import LayerNorm, QuickGELU, Attention, TextTransformer

class EvaAttention_IVLP(EvaAttention):
    def __init__(
        self, 
        dim, 
        add_prompt=False, 
        text_layer=False, 
        i=0, 
        design_details=None, 
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
    ):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
                         attn_drop=attn_drop, proj_drop=proj_drop, attn_head_dim=attn_head_dim, norm_layer=norm_layer)
        
        self.text_layer = text_layer
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, dim)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, dim)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False
            
    def forward(self, x, rope=None, attn_mask=None):
        if self.add_prompt:
            x = x.permute(1,0,2)
            # Remove the outputs produced by learnable tokens of previous layer
            prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
            # Create/configure learnable tokens of this layer
            visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
            # Add the learnable tokens of this layer with the input, by replacing the previous
            # layer learnable tokens
            x = torch.cat([prefix, visual_context], dim=0)
            x = x.permute(1,0,2)
        
        return super().forward(x, rope=rope, attn_mask=attn_mask)
        
class EvaBlock_IVLP(EvaBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        design_details=None,
        add_prompt=False,
        text_layer=False, 
        i=0, 
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,  
    ):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
                         mlp_ratio=mlp_ratio, swiglu_mlp=swiglu_mlp, scale_mlp=scale_mlp,
                         scale_attn_inner=scale_attn_inner, proj_drop=proj_drop, attn_drop=attn_drop,
                         drop_path=drop_path, init_values=init_values, act_layer=act_layer,
                         norm_layer=norm_layer, attn_head_dim=attn_head_dim)
        self.attn = EvaAttention_IVLP(dim, add_prompt=add_prompt, text_layer=text_layer,
                                      i=i, design_details=design_details,
                                      num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
                                      attn_drop=attn_drop, proj_drop=proj_drop, attn_head_dim=attn_head_dim,
                                      norm_layer=norm_layer if scale_attn_inner else None,
                                     )
        
    def forward(self, x, rope=None, attn_mask=None):
        return super().forward(x, rope=rope, attn_mask=attn_mask)
    
class EvaBlockPostNorm_IVLP(EvaBlockPostNorm):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        design_details=None,
        add_prompt=False,
        text_layer=False, 
        i=0, 
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,
    ):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
                         mlp_ratio=mlp_ratio, swiglu_mlp=swiglu_mlp, scale_mlp=scale_mlp,
                         scale_attn_inner=scale_attn_inner, proj_drop=proj_drop, attn_drop=attn_drop,
                         drop_path=drop_path, init_values=init_values, act_layer=act_layer,
                         norm_layer=norm_layer, attn_head_dim=attn_head_dim)
        self.attn = EvaAttention_IVLP(dim, add_prompt=add_prompt, text_layer=text_layer,
                                      i=i, design_details=design_details,
                                      num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused,
                                      attn_drop=attn_drop, proj_drop=proj_drop, attn_head_dim=attn_head_dim,
                                      norm_layer=norm_layer if scale_attn_inner else None,
                                     )
    
    def forward(self, x, rope=None, attn_mask=None):
        return super().forward(x, rope=rope, attn_mask=attn_mask)
    
class EVAVisionTransformer_IVLP(Eva):
    def __init__(
        self,
        design_details=None,
        img_size = 224,
        patch_size = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = 'avg',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: Callable = LayerNorm,
        init_values: Optional[float] = None,
        class_token: bool = True,
        use_abs_pos_emb: bool = True,
        use_rot_pos_emb: bool = False,
        use_post_norm: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
        head_init_scale: float = 0.001
    ):
        super().__init__(img_size=img_size, patch_size=patch_size, 
                         embed_dim=embed_dim, num_classes=num_classes,
                         depth=depth, num_heads=num_heads, 
                         qkv_fused=qkv_fused, mlp_ratio=mlp_ratio,
                         swiglu_mlp=swiglu_mlp, scale_mlp=scale_mlp, use_rot_pos_emb=use_rot_pos_emb,
                         ref_feat_shape=ref_feat_shape, global_pool=global_pool)
        self.input_resolution = img_size
        self.design_details = design_details
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            n_ctx = design_details["vision_ctx"]
            ctx_vectors = torch.empty(n_ctx, embed_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            
        self.prompt_till_layer_visual = design_details["vision_depth"]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        block_fn = EvaBlockPostNorm_IVLP if isinstance(self.blocks[0], EvaBlockPostNorm) else EvaBlock_IVLP
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                design_details=design_details,
                add_prompt=self.prompt_till_layer_visual,
                text_layer=False, 
                i=i,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])
        self.fix_init_weight()
        
    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if pos_embed is not None:
            x = x + pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        return x, rot_pos_embed
        
    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
            rot_dim = rot_pos_embed.size(-1)
            n_ctx = self.design_details["vision_ctx"]
            rot_pos_embed = torch.cat([rot_pos_embed, torch.zeros(n_ctx, rot_dim).to(
                            dtype=rot_pos_embed.dtype, device=rot_pos_embed.device)])
            
        else:
            assert self.prompt_till_layer_visual == 0
        
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x
    
    def forward(self, x):
        return super().forward(x)
    
### EVA-CLIP    
@dataclass
class CLIPVisionCfg:
    img_size = 224
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    qkv_fused = False
    mlp_ratio = 4 * 2 / 3
    swiglu_mlp = True
    scale_mlp = True
    scale_attn_inner = True
    use_rot_pos_emb = True
    ref_feat_shape = (16, 16)
    global_pool = "token"
    num_classes = 512
    
@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    masked_language_modeling: bool = False
    fusedLN: bool = False
    xattn: bool = False
    attn_mask: bool = True
    embed_dim: int = 512
    
def get_cast_dtype(precision):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

def _build_vision_tower(
    design_details,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None
):
    
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
        
    visual = EVAVisionTransformer_IVLP(
        design_details,
        img_size=vision_cfg.img_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.embed_dim,
        depth=vision_cfg.depth,
        num_heads=vision_cfg.num_heads,
        qkv_fused=vision_cfg.qkv_fused,
        mlp_ratio=vision_cfg.mlp_ratio,
        swiglu_mlp=vision_cfg.swiglu_mlp,
        scale_mlp=vision_cfg.scale_mlp,
        scale_attn_inner=vision_cfg.scale_attn_inner,
        use_rot_pos_emb=vision_cfg.use_rot_pos_emb,
        ref_feat_shape=vision_cfg.ref_feat_shape,
        global_pool=vision_cfg.global_pool,
        num_classes=vision_cfg.num_classes
    )
    return visual
        
def _build_text_tower(
    design_details,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    cast_dtype = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNorm
    FusedLayerNorm = LayerNorm
    
    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        design_details=design_details,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=text_cfg.embed_dim,
        act_layer=act_layer,
        norm_layer= FusedLayerNorm if text_cfg.fusedLN else norm_layer,
        xattn=text_cfg.xattn,
        attn_mask=text_cfg.attn_mask,
    )
    return text

class EvaCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        design_details = None,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(design_details, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(design_details, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}
    
    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()
    
if __name__ == "__main__":
    from open_clip import create_model_from_pretrained, get_tokenizer
    from collections import defaultdict
    model, preprocess = create_model_from_pretrained('hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k')
    state_dict = model.visual.trunk.state_dict()
    
    design_details = {}
    design_details["vision_depth"] = 2
    design_details["vision_ctx"] = 4
    vision_cfg = CLIPVisionCfg
    model = _build_vision_tower(design_details, vision_cfg)