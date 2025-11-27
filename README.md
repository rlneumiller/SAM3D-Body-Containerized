# SAM 3D Body Podman Container

* Use with caution - "works on my machine"

* This readme is WIP - The Dockerfile is (hopefully) the source of truth

This repository contains a Podman container setup for experimenting with [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body).
I've taken the liberty of borrowing from whatever open source projects I needed to borrow from to get this working.

## Prerequisites

* python venv (I used v3.13.5) named .venv (I used `uv venv`)
* Podman installed (not tested with Docker)
* NVIDIA GPU with CUDA support
* NVIDIA Container Toolkit configured for Podman
* Huggingface token set as env var HF_TOKEN

## Build the continuer image

```bash
podman build -t localhost/sam-3d-body --build-arg HF_TOKEN=$HF_TOKEN .
```

## Run the container and open a shell session in the container

```bash
podman run --rm --hooks-dir=/usr/share/containers/oci/hooks.d --device nvidia.com/gpu=all -v "$(pwd)/data":/workspace/data -v "$(pwd)/output":/workspace/output -it localhost/sam-3d-body bash
```

## Run demo.py from within the container

```bash
xvfb-run -s "-screen 0 1024x768x24" python demo.py --image_folder /workspace/data --output_folder /workspace/output --checkpoint_path ./checkpoints/dinov3/model.ckpt --mhr_path ./checkpoints/dinov3/assets/mhr_model.pt
```

-----------------------------------------------------------------

* Other stuff I haven't bothered with yet

-----------------------------------------------------------------

## Other stuff I haven't bothered with yet

## Directory Structure

- `data/` - Place your input images/videos here
- `output` - Where the output will be created
- `Dockerfile` - Container definition

## Notes

- The container runs root user
- GPU access is provided via `--device nvidia.com/gpu=all`
- Volumes are mounted with `:Z` for SELinux compatibility

## Troubleshooting

If you encounter GPU issues:

Check GPU is accessible: `podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

* The container now pins `torchvision==0.20.1+cu121` from the CUDA 12.1 wheels so operators like `torchvision::nms` are available when running the SAM-3D-Body demo.

Keeping in mind that the facebook repo does not use a container: For more information, see the [SAM 3D Body installation guide](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md).
