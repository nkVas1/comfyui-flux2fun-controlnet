# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-02-12

### Fixed
- **Device mismatch error with DisTorch2/MultiGPU**: Fixed `RuntimeError: Expected all tensors to be on the same device`
  - Root cause: VAE with DisTorch2 outputs on CPU, but mask was created on `cuda:0`
  - Solution: Introduced `target_device` variable that ensures all tensors are on the same device before concatenation
  - In `low_vram` mode, all tensors now stay on CPU during context assembly
- Added explicit `.to(device=target_device)` calls after VAE encoding
- Mask processing now uses `target_device` instead of `vae_device`

## [2.0.0] - 2025-01-10

### Added
- **Lazy Loading Architecture**: Complete rewrite of memory management
  - `Flux2FunControlNetContainer` wrapper for smart device placement
  - Model stays on CPU until inference, preventing OOM on load
  - Automatic VRAM state detection from ComfyUI (`--lowvram`, `--novram`)
- **New loader parameter**: `low_vram_mode` with options:
  - `auto`: Detect from ComfyUI flags (default)
  - `enabled`: Force CPU offloading
  - `disabled`: Keep model on GPU
- **Memory management utilities**:
  - `get_vram_state()`: Detect ComfyUI VRAM mode
  - `get_free_vram()`: Query available GPU memory
  - `soft_empty_cache()`: Safe cache clearing
  - `model_size_mb()`: Calculate model memory footprint
- **Sequential image encoding**: Apply node processes images one at a time
  - Reduces peak memory during VAE encoding
  - Immediate tensor cleanup after each encode

### Changed
- Loader now returns `Flux2FunControlNetContainer` instead of raw model
  - Backward compatible: Apply node handles both formats
- `ControlNetWrapper` now works with containers for proper memory tracking
- `inference_memory_requirements()` returns accurate memory estimate for ComfyUI

### Fixed
- **Critical OOM fix**: Model no longer loaded to GPU immediately
  - Previously: `controlnet.to(device)` at line 481 caused OOM when VRAM nearly full
  - Now: Model created on CPU, moved to GPU only when free VRAM available
- Memory leaks from intermediate tensors during image encoding
- Control context kept on CPU in low VRAM mode until forward pass

### Technical
- `flux_patch.py` updated to use containers and `_soft_empty_cache()`
- Container tracks `_is_on_gpu` state for efficient device management
- Added `to_device()` method with VRAM check before GPU migration

## [1.1.0] - 2025-01-09

### Added
- Low VRAM mode with CPU offloading
  - Automatically detects ComfyUI's `--lowvram` flag
  - Moves controlnet to CPU when not in use, reducing peak VRAM
  - Enables users with 8-12GB VRAM to run multi-reference workflows

### Fixed
- Memory cleanup: free VAE latents and intermediate tensors immediately after use
- Free hint tensors after application at each layer
- Add `torch.cuda.empty_cache()` calls to help GPU reclaim memory

### Technical
- Detect `VRAMState.LOW_VRAM` and `VRAMState.NO_VRAM` from `comfy.model_management`
- Control context kept on CPU in low VRAM mode, moved to GPU only during forward pass

## [1.0.2] - 2025-01-07

### Added
- Support for Flux2 reference latent images
- Experimental support for chaining multiple Flux2Fun controlnets
  - Hints from chained controlnets are summed at each injection layer
  - Each controlnet can have independent strength settings

### Technical
- ControlNet hints only applied to main image tokens, not reference latent tokens
- Wrapper supports `previous_controlnet` for chaining via ComfyUI's control system

## [1.0.0] - 2025-01-05

### Added
- Initial release
- `Load Flux2 Fun ControlNet` node for loading FLUX.2-dev-Fun-Controlnet-Union checkpoint
- `Apply Flux2 Fun ControlNet` node for applying control to Flux generation
- Support for multiple control modes:
  - Pose (OpenPose)
  - Canny (edge detection)
  - Depth (depth maps)
  - HED (soft edges)
  - MLSD (line segments)
  - Tile (upscaling/detail)
- Experimental inpainting support via mask and inpaint_image inputs
- Monkey patch system - no ComfyUI core modifications required
- Example workflows

### Technical
- Native architecture implementation matching VideoX-Fun reference
- RoPE (Rotary Position Embedding) handling for ComfyUI compatibility
- VAE batch normalization support
- Hint injection at Flux double stream blocks 0, 2, 4, 6
