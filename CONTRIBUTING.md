# Contributing

Thank you for your interest in contributing to ComfyUI Flux2 Fun ControlNet!

## Reporting Issues

If you encounter a bug or have a feature request:

1. Check existing [issues](https://github.com/bmcguire/comfyui-flux2fun-controlnet/issues) first
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - ComfyUI version
   - Python version
   - GPU/VRAM info
   - Relevant console output/logs

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Test with ComfyUI
5. Commit with clear messages
6. Push and create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/comfyui-flux2fun-controlnet.git

# Create symlink in ComfyUI custom_nodes (for development)
ln -s /path/to/comfyui-flux2fun-controlnet /path/to/ComfyUI/custom_nodes/

# Test changes by restarting ComfyUI
```

## Code Style

- Follow existing code patterns
- Use type hints where appropriate
- Keep functions focused and documented
- Test with different control types (pose, canny, depth)

## Questions?

Open a [discussion](https://github.com/bmcguire/comfyui-flux2fun-controlnet/discussions) or issue.
