# Example Workflows

## workflow_control.json
Basic control workflow demonstrating pose/canny/depth control.

**Setup:**
1. Load a Flux.2-dev checkpoint
2. Load flux2-vae.safetensors 
3. Load FLUX.2-dev-Fun-Controlnet-Union.safetensors
4. Provide a control image (pose skeleton, canny edges, or depth map)

**Settings:**
- Strength: 0.75 (adjust 0.65-0.80 for control modes)
- Steps: 35
- CFG: 4.0
- Sampler: euler

## Usage Tips

### Control Images
The model auto-detects the control type from your image:
- **Pose**: OpenPose skeleton (black background, colored joints/limbs)
- **Canny**: Edge detection output (white edges on black)
- **Depth**: Depth map (grayscale, white=close, black=far)
- **HED**: Soft edge detection
- **MLSD**: Line segment detection

### Strength Guidelines
- **Control only**: 0.65 - 0.80
- **Control + Inpaint**: 0.25 - 0.40
