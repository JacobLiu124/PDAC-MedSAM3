#!/usr/bin/env bash
# Set up MedSAM3-PDAC on a Linux machine with an NVIDIA GPU (CUDA 12.x driver).
#   bash install.sh            -> creates .venv here and clones MedSAM3 into ./MedSAM3
# Afterwards log in to Hugging Face once (facebook/sam3 is gated; request access on its page first):
#   .venv/bin/hf auth login
set -euo pipefail
cd "$(dirname "$0")"
MEDSAM3_COMMIT=84f43118b5b67b50abad81d7c946681ebb97dcda   # the MedSAM3 version these weights were trained with

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d MedSAM3/.git ]; then
  git clone https://github.com/Joey-S-Liu/MedSAM3.git MedSAM3
fi
git -C MedSAM3 checkout -q "$MEDSAM3_COMMIT"

[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# MedSAM3's requirements without dev tools and without its own torch pins
grep -vE '^(#|$)|pytest|black|flake8|openai|^torch[>=<]|^torchvision[>=<]' MedSAM3/requirements.txt > .ms3_requirements.txt
uv pip install --python .venv/bin/python -r .ms3_requirements.txt -r requirements.txt
uv pip install --python .venv/bin/python -e MedSAM3
rm -f .ms3_requirements.txt

.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Installed. Next: request access to https://huggingface.co/facebook/sam3, then run: .venv/bin/hf auth login"
