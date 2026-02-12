"""
Flux2 Fun ControlNet for ComfyUI

ComfyUI implementation of FLUX.2-dev-Fun-Controlnet-Union from Alibaba's VideoX-Fun.
Supports pose, canny, depth, HED, MLSD, tile control modes.

Optimized for low VRAM systems (8-12GB):
- Lazy loading: Model stays on CPU until needed
- Automatic offloading: Integrates with ComfyUI's memory management
- Chunked processing: Minimizes peak memory usage

Usage:
1. Place this folder in ComfyUI/custom_nodes/
2. Download FLUX.2-dev-Fun-Controlnet-Union.safetensors to ComfyUI/models/controlnet/
3. Use "Load Flux2 Fun ControlNet" and "Apply Flux2 Fun ControlNet" nodes

Model: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any

import folder_paths
import comfy.utils
import comfy.model_management as mm

# Apply monkey patch on import
from . import flux_patch
flux_patch.apply_patch()


# =============================================================================
# Memory Management Utilities
# =============================================================================

def get_vram_state() -> Tuple[bool, bool, str]:
    """
    Detect VRAM state from ComfyUI's model management.
    
    Returns:
        Tuple of (is_low_vram, is_no_vram, vram_mode_name)
    """
    try:
        from comfy.model_management import vram_state, VRAMState
        is_low_vram = vram_state in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM)
        is_no_vram = vram_state == VRAMState.NO_VRAM
        mode_name = vram_state.name if hasattr(vram_state, 'name') else str(vram_state)
        return is_low_vram, is_no_vram, mode_name
    except (ImportError, AttributeError):
        return False, False, "UNKNOWN"


def get_free_vram() -> float:
    """Get available VRAM in MB."""
    try:
        return mm.get_free_memory() / (1024 * 1024)
    except:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
        return 0


def soft_empty_cache():
    """Clear GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


# =============================================================================
# Architecture Components
# =============================================================================

def attention_forward(query, key, value, attn_mask=None):
    """Attention using PyTorch's scaled_dot_product_attention."""
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.transpose(1, 2).contiguous()


def apply_rotary_emb(x, freqs_cis, sequence_dim=1):
    """Apply rotary embeddings with sequence length handling."""
    if freqs_cis is None:
        return x
    
    cos, sin = freqs_cis
    seq_len = x.shape[sequence_dim]
    rope_seq_len = cos.shape[0]
    
    # Handle sequence length mismatch
    if seq_len != rope_seq_len:
        if seq_len < rope_seq_len:
            cos = cos[:seq_len]
            sin = sin[:seq_len]
        else:
            pad_len = seq_len - rope_seq_len
            cos = torch.cat([cos, cos[-1:].expand(pad_len, -1)], dim=0)
            sin = torch.cat([sin, sin[-1:].expand(pad_len, -1)], dim=0)
    
    if sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    
    cos, sin = cos.to(x.device, dtype=x.dtype), sin.to(x.device, dtype=x.dtype)
    
    # Handle head dimension mismatch
    if cos.shape[-1] != x.shape[-1]:
        if cos.shape[-1] > x.shape[-1]:
            cos = cos[..., :x.shape[-1]]
            sin = sin[..., :x.shape[-1]]
        else:
            pad_size = x.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
    
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, dim, dim_out=None, mult=3.0, bias=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x):
        return self.linear_out(self.act_fn(self.linear_in(x)))


class Attention(nn.Module):
    """Multi-head attention with optional added KV projections."""
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0, bias=False,
                 added_kv_proj_dim=None, added_proj_bias=True, out_bias=True,
                 eps=1e-5, out_dim=None, elementwise_affine=True):
        super().__init__()
        
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim else dim_head * heads
        self.out_dim = out_dim if out_dim else query_dim
        self.heads = out_dim // dim_head if out_dim else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])

        if added_kv_proj_dim:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

    def forward(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None, **kwargs):
        query = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        if encoder_hidden_states is not None and self.added_kv_proj_dim:
            enc_q = self.norm_added_q(self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1)))
            enc_k = self.norm_added_k(self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1)))
            enc_v = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
            
            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        out = attention_forward(query, key, value).flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None and self.added_kv_proj_dim:
            enc_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = self.to_add_out(out[:, :enc_len])
            out = out[:, enc_len:]

        out = self.to_out[1](self.to_out[0](out))

        if encoder_hidden_states is not None and self.added_kv_proj_dim:
            return out, encoder_hidden_states
        return out


class TransformerBlock(nn.Module):
    """Transformer block with dual-stream modulation."""
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0, eps=1e-6, bias=False):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        
        self.attn = Attention(
            query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim,
            heads=num_attention_heads, out_dim=dim, bias=bias, added_proj_bias=bias,
            out_bias=bias, eps=eps
        )
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)
        
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_params_img,
                temb_mod_params_txt, image_rotary_emb=None, **kwargs):
        
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        norm_h = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
        norm_enc = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa

        attn_out, ctx_attn_out = self.attn(norm_h, norm_enc, image_rotary_emb)

        hidden_states = hidden_states + gate_msa * attn_out
        norm_h = (1 + scale_mlp) * self.norm2(hidden_states) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_h)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * ctx_attn_out
        norm_enc = (1 + c_scale_mlp) * self.norm2_context(encoder_hidden_states) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_enc)

        return encoder_hidden_states, hidden_states


class ControlTransformerBlock(TransformerBlock):
    """Control block with before_proj/after_proj for hint generation."""
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0,
                 eps=1e-6, bias=False, block_id=0):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps, bias)
        self.block_id = block_id
        
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        encoder_hidden_states, c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        
        return encoder_hidden_states, c


# =============================================================================
# ControlNet Model
# =============================================================================

class Flux2FunControlNet(nn.Module):
    """
    Flux2 Fun ControlNet - generates control hints for Flux diffusion.
    
    From Alibaba's VideoX-Fun implementation.
    Supports: pose, canny, depth, HED, MLSD, tile, and inpainting.
    
    Memory optimized:
    - Supports CPU offloading for low VRAM systems
    - Lazy device placement
    """
    
    CONTROL_LAYERS = [0, 2, 4, 6]
    
    def __init__(self, hidden_size=6144, num_attention_heads=48, attention_head_dim=128,
                 mlp_ratio=3.0, control_in_dim=260, num_blocks=4, eps=1e-6,
                 dtype=None, device=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.control_in_dim = control_in_dim
        self.num_blocks = num_blocks
        self.control_layers_mapping = {layer: idx for idx, layer in enumerate(self.CONTROL_LAYERS[:num_blocks])}
        
        self.control_img_in = nn.Linear(control_in_dim, hidden_size)
        
        self.control_transformer_blocks = nn.ModuleList([
            ControlTransformerBlock(
                dim=hidden_size, num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio,
                eps=eps, bias=False, block_id=i
            ) for i in range(num_blocks)
        ])
        
        # Store preferred dtype for inference
        self._dtype = dtype
        
        if dtype: 
            self.to(dtype)
        if device: 
            self.to(device)
    
    @property
    def model_dtype(self):
        """Get the dtype of the model parameters."""
        return next(self.parameters()).dtype
    
    @property  
    def model_device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device
    
    def memory_usage_mb(self) -> float:
        """Calculate model memory usage in MB."""
        return model_size_mb(self)
    
    def forward_control(self, x, control_context, encoder_hidden_states,
                        temb_mod_params_img, temb_mod_params_txt,
                        image_rotary_emb=None,
                        ctrl_h=None, ctrl_w=None, txt_seq_len=None,
                        debug=False) -> List[torch.Tensor]:
        """
        Generate control hints to inject into Flux blocks.
        
        Args:
            x: Hidden states from main model [B, seq, hidden]
            control_context: Control input [B, seq, 260] 
                             (128 control + 4 mask + 128 inpaint)
            encoder_hidden_states: Text embeddings [B, txt_seq, hidden]
            temb_mod_params_img: Modulation params for image stream
            temb_mod_params_txt: Modulation params for text stream
            image_rotary_emb: RoPE embeddings (cos, sin)
            ctrl_h, ctrl_w: Control spatial dimensions
            txt_seq_len: Text sequence length
            debug: Enable debug output
        
        Returns:
            List of hint tensors to add to Flux blocks
        """
        if debug:
            print(f"[Flux2 Fun] forward_control:")
            print(f"  x: {x.shape}, abs_mean={x.abs().mean():.4f}")
            print(f"  control_context: {control_context.shape}, abs_mean={control_context.abs().mean():.4f}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                print(f"  image_rotary_emb: cos={cos.shape}, sin={sin.shape}")
        
        # Project control context to hidden dimension
        c = self.control_img_in(control_context)
        
        if debug:
            print(f"  After control_img_in: {c.shape}, abs_mean={c.abs().mean():.4f}")
            (shift, scale, gate), _ = temb_mod_params_img
            print(f"  Modulation: shift={shift.abs().mean():.4f}, scale={scale.abs().mean():.4f}, gate={gate.abs().mean():.4f}")
        
        kwargs = dict(
            x=x,
            encoder_hidden_states=encoder_hidden_states.clone(),
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=image_rotary_emb,
        )
        
        for i, block in enumerate(self.control_transformer_blocks):
            encoder_hidden_states_out, c = block(c, **kwargs)
            kwargs["encoder_hidden_states"] = encoder_hidden_states_out
            
            if debug:
                hint = c[-2] if c.shape[0] > 1 else c[0]
                print(f"  Block {i}: c_state={c[-1].abs().mean():.4f}, hint={hint.abs().mean():.6f}")
        
        hints = list(torch.unbind(c))[:-1]
        
        if debug:
            print(f"  Final: {len(hints)} hints")
            for i, h in enumerate(hints):
                print(f"    hint[{i}]: abs_mean={h.abs().mean():.6f}")
        
        return hints


# =============================================================================
# Model Container with Memory Management
# =============================================================================

class Flux2FunControlNetContainer:
    """
    Container for Flux2FunControlNet with smart memory management.
    
    Features:
    - Lazy loading: Model stays on CPU until inference
    - Automatic offloading: Moves to CPU when not in use
    - VRAM awareness: Checks available memory before loading
    - ComfyUI integration: Works with --lowvram flag
    """
    
    def __init__(self, controlnet: Flux2FunControlNet, dtype: torch.dtype):
        self.controlnet = controlnet
        self.dtype = dtype
        self._is_on_gpu = False
        self._model_size_mb = controlnet.memory_usage_mb()
        
        # Detect VRAM mode on creation
        self.low_vram, self.no_vram, self.vram_mode = get_vram_state()
        
        # Always keep on CPU initially
        self.controlnet.to('cpu')
        self._is_on_gpu = False
        
    def get_model(self) -> Flux2FunControlNet:
        """Get the underlying model."""
        return self.controlnet
    
    def to_device(self, device: torch.device, force: bool = False) -> Flux2FunControlNet:
        """
        Move model to device with memory checks.
        
        Args:
            device: Target device
            force: Force move even if low on memory
            
        Returns:
            Model on the specified device
        """
        target_device = str(device)
        
        if 'cuda' in target_device:
            # Check if we have enough VRAM
            free_vram = get_free_vram()
            required_mb = self._model_size_mb * 1.1  # 10% overhead
            
            if not force and free_vram < required_mb:
                print(f"[Flux2 Fun] Warning: Low VRAM ({free_vram:.0f}MB free, need {required_mb:.0f}MB). Keeping on CPU.")
                return self.controlnet
            
            if not self._is_on_gpu:
                self.controlnet.to(device=device, dtype=self.dtype)
                self._is_on_gpu = True
                
        else:
            # Moving to CPU
            if self._is_on_gpu:
                self.controlnet.to('cpu')
                self._is_on_gpu = False
                soft_empty_cache()
                
        return self.controlnet
    
    def offload_to_cpu(self):
        """Move model to CPU and free GPU memory."""
        if self._is_on_gpu:
            self.controlnet.to('cpu')
            self._is_on_gpu = False
            soft_empty_cache()
            
    def is_on_gpu(self) -> bool:
        """Check if model is currently on GPU."""
        return self._is_on_gpu
    
    def memory_mb(self) -> float:
        """Get model size in MB."""
        return self._model_size_mb


# =============================================================================
# ComfyUI Integration - ControlNet Wrapper
# =============================================================================

class ControlNetWrapper:
    """
    Wrapper to integrate with ComfyUI's control system.
    
    Features:
    - Supports chaining multiple Flux2Fun controlnets
    - Dynamic CPU<->GPU migration based on VRAM
    - Proper memory requirement reporting for ComfyUI
    - Low VRAM mode support
    """
    
    def __init__(self, container: Flux2FunControlNetContainer, 
                 control_context: torch.Tensor, strength: float, 
                 ctrl_h: int, ctrl_w: int, low_vram: bool = False):
        self.container = container
        self.controlnet = container.get_model()
        self.control_context = control_context
        self.strength = strength
        self.ctrl_h = ctrl_h
        self.ctrl_w = ctrl_w
        self.low_vram = low_vram or container.low_vram
        self.previous_controlnet = None
        
        # In low_vram mode, keep control_context on CPU until needed
        if self.low_vram and control_context is not None:
            self.control_context = control_context.cpu()
        
        class HooksContainer:
            hooks = []
        self.extra_hooks = HooksContainer()
        
    def pre_run(self, model, percent_to_timestep_function):
        """Called before sampling starts."""
        if self.previous_controlnet:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)
    
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
        """
        Called by ComfyUI during sampling to get control signals.
        Sets up transformer_options for the patch to use.
        """
        control_prev = None
        if self.previous_controlnet:
            control_prev = self.previous_controlnet.get_control(
                x_noisy, t, cond, batched_number, transformer_options
            )
        
        if transformer_options:
            # Initialize lists if this is the first Flux2Fun controlnet in the chain
            if 'flux2_fun_controlnets' not in transformer_options:
                transformer_options['flux2_fun_controlnets'] = []
                transformer_options['flux2_fun_control_contexts'] = []
                transformer_options['flux2_fun_control_scales'] = []
                transformer_options['flux2_fun_ctrl_dims'] = []
                transformer_options['flux2_fun_low_vram'] = self.low_vram
                transformer_options['flux2_fun_containers'] = []
            
            # Append this controlnet's data to the lists
            transformer_options['flux2_fun_controlnets'].append(self.controlnet)
            transformer_options['flux2_fun_control_contexts'].append(self.control_context)
            transformer_options['flux2_fun_control_scales'].append(self.strength)
            transformer_options['flux2_fun_ctrl_dims'].append((self.ctrl_h, self.ctrl_w))
            transformer_options['flux2_fun_containers'].append(self.container)
        
        output = {"input": [], "output": []}
        if control_prev:
            output["input"] = control_prev.get("input", [])
            output["output"] = control_prev.get("output", [])
        return output
    
    def copy(self):
        """Create a copy for batched processing."""
        c = ControlNetWrapper(
            self.container, self.control_context, self.strength, 
            self.ctrl_h, self.ctrl_w, self.low_vram
        )
        c.previous_controlnet = self.previous_controlnet
        return c
    
    def cleanup(self):
        """Called after sampling completes."""
        # Offload to CPU in low VRAM mode
        if self.low_vram:
            self.container.offload_to_cpu()
        if self.previous_controlnet:
            self.previous_controlnet.cleanup()
    
    def get_models(self):
        """Return list of models for ComfyUI memory management."""
        return self.previous_controlnet.get_models() if self.previous_controlnet else []
    
    def get_extra_hooks(self):
        """Return extra hooks for ComfyUI."""
        return self.previous_controlnet.get_extra_hooks() if self.previous_controlnet else []
    
    def inference_memory_requirements(self, dtype) -> int:
        """
        Calculate memory requirements for ComfyUI's memory manager.
        
        Returns memory in bytes.
        """
        # Model parameters
        mem = int(self.container.memory_mb() * 1024 * 1024)
        
        # Add estimate for activations (roughly 20% of model size)
        mem = int(mem * 1.2)
        
        # Add previous controlnet requirements
        if self.previous_controlnet:
            mem += self.previous_controlnet.inference_memory_requirements(dtype)
            
        return mem


# =============================================================================
# ComfyUI Nodes
# =============================================================================

class Flux2FunControlNetLoader:
    """
    Load Flux2 Fun ControlNet checkpoint.
    
    Memory optimized:
    - Model loads to CPU first
    - Automatic VRAM mode detection
    - Smart device placement during inference
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            },
            "optional": {
                "low_vram_mode": (["auto", "enabled", "disabled"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("FLUX2_FUN_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"
    DESCRIPTION = """Load Flux2 Fun ControlNet model.

Memory optimized for low VRAM systems:
- auto: Detects ComfyUI's --lowvram flag
- enabled: Force CPU offloading
- disabled: Keep on GPU (needs ~8.3GB VRAM)"""
    
    def load_controlnet(self, controlnet_name, low_vram_mode="auto"):
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        print(f"[Flux2 Fun] Loading: {controlnet_name}")
        
        # Load weights on CPU first (critical for low VRAM)
        state_dict = comfy.utils.load_torch_file(controlnet_path, safe_load=True)
        
        # Detect architecture from weights
        control_in_dim = state_dict["control_img_in.weight"].shape[1]
        hidden_size = state_dict["control_img_in.weight"].shape[0]
        num_blocks = max(int(k.split(".")[1]) for k in state_dict if k.startswith("control_transformer_blocks.")) + 1
        
        print(f"[Flux2 Fun] Architecture: hidden={hidden_size}, ctrl_dim={control_in_dim}, blocks={num_blocks}")
        
        # Detect VRAM state
        is_low_vram, is_no_vram, vram_mode = get_vram_state()
        
        # Determine if we should use low VRAM mode
        if low_vram_mode == "auto":
            use_low_vram = is_low_vram
        elif low_vram_mode == "enabled":
            use_low_vram = True
        else:
            use_low_vram = False
            
        # Choose dtype
        dtype = torch.bfloat16 if mm.should_use_bf16() else torch.float16
        
        # Create model on CPU with the correct dtype
        controlnet = Flux2FunControlNet(
            hidden_size=hidden_size, 
            num_attention_heads=48,
            attention_head_dim=hidden_size // 48, 
            mlp_ratio=3.0,
            control_in_dim=control_in_dim, 
            num_blocks=num_blocks,
            dtype=dtype, 
            device="cpu"  # Always create on CPU first
        )
        
        # Load weights (already on CPU)
        missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
        
        # Free the state dict immediately
        del state_dict
        soft_empty_cache()
        
        if missing:
            print(f"[Flux2 Fun] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Flux2 Fun] Unexpected keys: {len(unexpected)}")
        
        controlnet.eval()
        
        # Wrap in container for memory management
        container = Flux2FunControlNetContainer(controlnet, dtype)
        
        model_size_mb_val = container.memory_mb()
        free_vram_mb = get_free_vram()
        
        print(f"[Flux2 Fun] Model size: {model_size_mb_val:.0f}MB")
        print(f"[Flux2 Fun] Free VRAM: {free_vram_mb:.0f}MB")
        print(f"[Flux2 Fun] VRAM mode: {vram_mode}")
        print(f"[Flux2 Fun] Low VRAM: {use_low_vram}")
        
        # Only move to GPU if we have enough VRAM and not in low VRAM mode
        if not use_low_vram and free_vram_mb > model_size_mb_val * 1.2:
            device = mm.get_torch_device()
            container.to_device(device)
            print(f"[Flux2 Fun] Loaded on GPU")
        else:
            print(f"[Flux2 Fun] Keeping on CPU (will move to GPU during inference)")
        
        return (container,)


class Flux2FunControlNetApply:
    """
    Apply Flux2 Fun ControlNet to conditioning.
    
    Modes:
    - Control only: Provide control_image (pose/canny/depth/etc)
    - Control + Inpaint: Provide control_image + mask + inpaint_image
    
    Memory optimized:
    - Sequential image encoding (not simultaneous)
    - Immediate tensor cleanup
    - Low VRAM mode support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "controlnet": ("FLUX2_FUN_CONTROLNET",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "control_image": ("IMAGE",),
                "mask": ("MASK",),
                "inpaint_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"
    DESCRIPTION = """Apply Flux2 Fun ControlNet to conditioning.

Control modes:
- Control: pose/canny/depth/HED/MLSD/tile
- Control + Inpaint: mask + inpaint_image

Strength: 0.65-0.80 recommended for control"""
    
    @staticmethod
    def _patchify(x):
        """Convert [B, C, H, W] -> [B, C*4, H/2, W/2] by rearranging 2x2 patches."""
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(b, c * 4, h // 2, w // 2)
        return x
    
    def apply_controlnet(self, conditioning, controlnet, vae, strength, 
                         control_image=None, mask=None, inpaint_image=None):
        
        # Get container (new format) or raw model (legacy)
        if isinstance(controlnet, Flux2FunControlNetContainer):
            container = controlnet
            cn_model = container.get_model()
            low_vram = container.low_vram
        else:
            # Legacy support: wrap raw model
            dtype = next(controlnet.parameters()).dtype
            container = Flux2FunControlNetContainer(controlnet, dtype)
            cn_model = controlnet
            low_vram, _, _ = get_vram_state()
        
        dtype = cn_model.model_dtype
        
        # Determine dimensions from available image
        if control_image is not None:
            bs, h, w, _ = control_image.shape
        elif inpaint_image is not None:
            bs, h, w, _ = inpaint_image.shape
        else:
            raise ValueError("Must provide either control_image or inpaint_image")
        
        # Ensure dimensions divisible by 16
        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        if h != new_h or w != new_w:
            h, w = new_h, new_w
        
        # Latent dimensions (VAE outputs packed 128ch at h/16, w/16)
        lat_h, lat_w = h // 16, w // 16
        
        # Load VAE to GPU
        mm.load_model_gpu(vae.patcher)
        vae_device = mm.get_torch_device()
        
        # Determine target device for context assembly
        # In low_vram mode or with DisTorch2, VAE outputs on CPU
        # We'll detect actual output device and use CPU as fallback for low_vram
        if low_vram:
            target_device = torch.device('cpu')
        else:
            target_device = vae_device
        
        # =====================================================================
        # Process images SEQUENTIALLY to minimize memory usage
        # =====================================================================
        
        # 1. Process mask first (smallest tensor)
        if mask is not None:
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            mask_binary = (mask >= 0.5).float()
            
            # Downscale mask for context
            mask_unpacked_size = (lat_h * 2, lat_w * 2)
            mask_for_context = F.interpolate(
                mask_binary.to(device=target_device, dtype=dtype), 
                mask_unpacked_size, 
                mode='nearest'
            )
            mask_for_context = 1.0 - mask_for_context  # Invert
            mask_for_context = self._patchify(mask_for_context)
            mask_flat = mask_for_context.flatten(2).permute(0, 2, 1)
            
            # Upscale for image masking (use vae_device for VAE input)
            mask_for_img = F.interpolate(
                mask_binary.to(device=vae_device, dtype=dtype), 
                (h, w), 
                mode='nearest'
            )
            
            # Free intermediate tensors
            del mask_for_context, mask_binary
        else:
            mask_flat = torch.zeros((bs, lat_h * lat_w, 4), device=target_device, dtype=dtype)
            mask_for_img = None
        
        soft_empty_cache()
        
        # 2. Encode inpaint image (with masked region zeroed)
        if inpaint_image is not None:
            inp_img = inpaint_image[:, :, :, :3].to(vae_device)
            if inp_img.shape[1:3] != (h, w):
                inp_img = F.interpolate(
                    inp_img.permute(0, 3, 1, 2), (h, w), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Zero out inpaint region before encoding
            if mask_for_img is not None:
                keep_mask = (mask_for_img < 0.5).float()
                inp_img = inp_img * keep_mask.permute(0, 2, 3, 1)
                del keep_mask
            
            with torch.no_grad():
                inpaint_latents = vae.encode(inp_img)
            
            # Move to target device for consistent concatenation
            inpaint_flat = inpaint_latents.to(device=target_device, dtype=dtype).flatten(2).permute(0, 2, 1)
            
            # Free immediately
            del inpaint_latents, inp_img
            soft_empty_cache()
        else:
            inpaint_flat = torch.zeros((bs, lat_h * lat_w, 128), device=target_device, dtype=dtype)
        
        # 3. Encode control image
        if control_image is not None:
            ctrl_img = control_image[:, :, :, :3].to(vae_device)
            if ctrl_img.shape[1:3] != (h, w):
                ctrl_img = F.interpolate(
                    ctrl_img.permute(0, 3, 1, 2), (h, w), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            
            with torch.no_grad():
                control_latents = vae.encode(ctrl_img)
            
            # Move to target device for consistent concatenation
            control_flat = control_latents.to(device=target_device, dtype=dtype).flatten(2).permute(0, 2, 1)
            
            # Free immediately
            del control_latents, ctrl_img
            soft_empty_cache()
        else:
            control_flat = torch.zeros((bs, lat_h * lat_w, 128), device=target_device, dtype=dtype)
        
        # Free mask_for_img if we used it
        if mask_for_img is not None:
            del mask_for_img
        
        # =====================================================================
        # Build control context: [control(128), mask(4), inpaint(128)] = 260
        # =====================================================================
        control_context = torch.cat([control_flat, mask_flat, inpaint_flat], dim=2)
        
        # Free components
        del control_flat, mask_flat, inpaint_flat
        soft_empty_cache()
        
        # In low VRAM mode, move context to CPU (will be moved back during inference)
        if low_vram:
            control_context = control_context.cpu()
        
        # Determine mode for logging
        if control_image is not None and inpaint_image is not None:
            mode = "control+inpaint"
        elif control_image is not None:
            mode = "control"
        else:
            mode = "inpaint"
        
        print(f"[Flux2 Fun] Mode: {mode}, strength: {strength}, low_vram: {low_vram}")
        
        # Create wrapper for ComfyUI's control system
        wrapper = ControlNetWrapper(container, control_context, strength, lat_h, lat_w, low_vram)
        
        # Apply to conditioning
        c = [[t[0], t[1].copy()] for t in conditioning]
        for t in c:
            # Chain with existing control if present
            existing_control = t[1].get('control', None)
            if existing_control is not None:
                wrapper.previous_controlnet = existing_control
            t[1]['control'] = wrapper
            t[1]['control_apply_to_uncond'] = True
        
        return (c,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "Flux2FunControlNetLoader": Flux2FunControlNetLoader,
    "Flux2FunControlNetApply": Flux2FunControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2FunControlNetLoader": "Load Flux2 Fun ControlNet",
    "Flux2FunControlNetApply": "Apply Flux2 Fun ControlNet",
}

__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS', 
    'Flux2FunControlNet',
    'Flux2FunControlNetContainer',
]
