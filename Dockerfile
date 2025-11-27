# Base image: PyTorch with CUDA 12.1 + cuDNN 9
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system packages: build tools + OpenGL/Mesa
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg \
    libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev \
    build-essential cmake ninja-build python3-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Clone SAM-3D-Body
RUN git clone https://github.com/facebookresearch/sam-3d-body.git
WORKDIR /workspace/sam-3d-body

# Upgrade pip/setuptools/wheel and conda before creating the dedicated env
RUN python -m pip install --upgrade pip setuptools wheel && \
    conda update -n base -c defaults conda

# Build a dedicated conda environment that keeps all python packages compatible
RUN conda create -n sam3d python=3.11 pip -y && \
    conda clean -afy
ENV CONDA_DEFAULT_ENV=sam3d
ENV PATH=/opt/conda/envs/sam3d/bin:$PATH
RUN python -m pip install --upgrade pip setuptools wheel huggingface_hub

# Download model files during build
RUN HF_TOKEN="$HF_TOKEN" hf download facebook/sam-3d-body-dinov3 --local-dir /root/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3

# Install PyMomentum via conda (not available on PyPI yet)
# This provides the Momentum library support for highest quality reconstructions
RUN conda install -n sam3d -c conda-forge pymomentum -y && \
    conda clean -afy

# Install Python dependencies via pip
RUN python -m pip install \
    opencv-python \
    yacs scikit-image einops timm==0.9.12 dill pandas rich \
    hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset \
    chump networkx==3.2.1 roma joblib seaborn wandb appdirs \
    cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools \
    tensorboard \
    moderngl glcontext \
    "numpy<2.0"

# Make sure torchvision matches the CUDA 12.1 PyTorch runtime so operators like nms exist
RUN python -m pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121 torchvision==0.20.1+cu121

RUN python -m pip install pytorch-lightning==2.4.0

# Install Detectron2 with the PosixPath fix
RUN python -m pip install 'git+https://github.com/volks73/detectron2.git@fix-pkg-resource-dep' --no-build-isolation --no-deps


# Detectron2's LazyConfig still assumes a str path, so ensure it converts Path before calling replace
RUN python - <<'PY'
import site
from pathlib import Path

site_packages = site.getsitepackages()[0]
lazy_path = Path(site_packages) / "detectron2" / "config" / "lazy.py"
text = lazy_path.read_text()
import_marker = "from pathlib import Path"
if import_marker not in text:
    insert_after = "from contextlib import contextmanager\n"
    text = text.replace(insert_after, insert_after + import_marker + "\n", 1)
target = "filename = filename.replace(\"/./\", \"/\")  # redundant"
replacement = "filename = str(filename) if isinstance(filename, Path) else filename\n        filename = filename.replace(\"/./\", \"/\")  # redundant"
if replacement not in text:
    text = text.replace(target, replacement, 1)
lazy_path.write_text(text)
PY


# Patch the Detectron2 model_zoo.py to handle Path objects properly
RUN python - <<'PY'
import site
from pathlib import Path

site_packages = site.getsitepackages()[0]
model_zoo = Path(site_packages) / "detectron2" / "model_zoo" / "model_zoo.py"
text = model_zoo.read_text()
text = text.replace('if cfg_file.endswith(".yaml"):', 'if str(cfg_file).endswith(".yaml"):', 1)
text = text.replace('elif cfg_file.endswith(".py"):', 'elif str(cfg_file).endswith(".py"):', 1)
model_zoo.write_text(text)
PY

# Install rendering dependencies
RUN python -m pip install pyrender==0.1.45 trimesh==4.3.0 

# Install MoGe2 (Microsoft monocular geometry estimator)
RUN python -m pip install git+https://github.com/microsoft/MoGe.git

# Copy model files to expected locations
RUN mkdir -p checkpoints/dinov3/assets && \
    cp /root/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/model.ckpt checkpoints/dinov3/model.ckpt && \
    cp /root/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/model_config.yaml checkpoints/dinov3/ && \
    cp -r /root/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/assets/* checkpoints/dinov3/assets/

RUN python -m pip install "numpy<2.0"


# Verify pymomentum installation
RUN python -c "import pymomentum; print('PyMomentum version:', pymomentum.__version__)" || \
    echo "Warning: PyMomentum import test failed, but continuing build"

# Build command:
# podman build -t localhost/sam-3d-body --build-arg HF_TOKEN=$HF_TOKEN .

# podman run --rm --hooks-dir=/usr/share/containers/oci/hooks.d --device nvidia.com/gpu=all \-v $(pwd)/data:/workspace/data -v $(pwd)/output:/workspace/output -it sam-3d-body-functional bash


# Run command:
# podman run --rm --hooks-dir=/usr/share/containers/oci/hooks.d --device nvidia.com/gpu=all \
#   -v "$(pwd)/data":/workspace/data -v "$(pwd)/output":/workspace/output \
#   -it localhost/sam-3d-body bash

# Demo command (inside container):

# xvfb-run -s "-screen 0 1024x768x24" python demo.py --image_folder /workspace/data --output_folder /workspace/output --checkpoint_path ./checkpoints/dinov3/model.ckpt --mhr_path ./checkpoints/dinov3/assets/mhr_model.pt

# xvfb-run -s "-screen 0 1024x768x24" python demo.py \
#   --image_folder /workspace/data --output_folder /workspace/output \
#   --checkpoint_path ./checkpoints/dinov3/model.ckpt \
#   --mhr_path ./checkpoints/dinov3/assets/mhr_model.pt