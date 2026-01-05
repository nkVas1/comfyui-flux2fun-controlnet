# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
