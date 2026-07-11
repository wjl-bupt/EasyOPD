#!/usr/bin/env bash
# =============================================================================
# install_easyopd_env.sh
# Adapted from the original install_easyopd.sh for this workspace.
#
# This recreates the verified EasyOPD/verl training environment (vLLM + flash-attn)
# that the maintainer validated on the Cross-Tokenizer methods. Because SDPO shares
# the same verl framework, is rollout-engine agnostic, and needs no extra deps,
# this environment also runs SDPO.
#
# What was changed vs. the original script:
#   - conda paths are derived from `conda info --base` (no hardcoded /opt/conda)
#   - EASYOPD_ROOT / PYTHONPATH point to this EasyOPD (has SDPO wiring)
#   - REQUIREMENTS_FILE points at the original full pip-freeze (absolute path)
#
# Environment Summary (unchanged, validated):
#   - Conda env name:  OpenAgentRL-sj
#   - Python:          3.11
#   - PyTorch:         2.6.0+cu124
#   - vLLM:            0.8.5.post1
#   - flash-attn:      2.7.4.post1
#   - xformers:        0.0.29.post2
#   - verl:            0.5.0 (note: the in-repo verl via PYTHONPATH takes priority)
#
# Usage:
#   bash scripts/install_easyopd_env.sh            # create env OpenAgentRL-sj
#   bash scripts/install_easyopd_env.sh --force    # recreate if exists
# =============================================================================
set -euo pipefail

# Mirror all output to a shared-disk log so it can be inspected later.
_INSTALL_LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_logs"
mkdir -p "${_INSTALL_LOG_DIR}"
exec > >(tee "${_INSTALL_LOG_DIR}/install_latest.log" "${_INSTALL_LOG_DIR}/install_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "[install] Full log mirrored to: ${_INSTALL_LOG_DIR}/install_latest.log"

# ======================== Configuration ========================
ENV_NAME="${ENV_NAME:-OpenAgentRL-sj}"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.6.0"
CUDA_TAG="cu124"
VLLM_VERSION="0.8.5.post1"
FLASH_ATTN_VERSION="2.7.4.post1"
XFORMERS_VERSION="0.0.29.post2"
TRITON_VERSION="3.2.0"
VERL_VERSION="0.5.0"

# [CHANGED] conda base is resolved in Step 0 (after ensure_conda), because conda
# may not be installed yet on a fresh machine.

# [CHANGED] Point at THIS workspace (so the env's PYTHONPATH gets the SDPO wiring)
EASYOPD_ROOT="${EASYOPD_ROOT:-/path/to/workspace/EasyOPD}"

# [CHANGED] Reuse the original full, validated pip-freeze (absolute path on shared disk)
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-/path/to/workspace/settings/install/easyopd_requirements.txt}"

# PyPI mirror (Tencent, fast in China)
MIRROR="https://mirrors.tencent.com/pypi/simple/"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

# Parse command-line arguments
FORCE_RECREATE=false
if [[ "${1:-}" == "--force" ]] || [[ "${1:-}" == "-f" ]]; then
    FORCE_RECREATE=true
    echo "  --force: will recreate environment without confirmation"
fi

echo "============================================================"
echo "  EasyOPD Environment Installation (local adaptation)"
echo "============================================================"
echo "  Env name:       ${ENV_NAME}"
echo "  PyTorch:        ${TORCH_VERSION}+${CUDA_TAG}"
echo "  vLLM:           ${VLLM_VERSION}"
echo "  flash-attn:     ${FLASH_ATTN_VERSION}"
echo "  EasyOPD root:   ${EASYOPD_ROOT}"
echo "  requirements:   ${REQUIREMENTS_FILE}"
echo "============================================================"
echo ""

# ======================== Step 0: Check prerequisites ========================
echo "[Step 0/6] Checking prerequisites..."

# Ensure conda is available; auto-install Miniconda if it is missing.
ensure_conda() {
    if command -v conda &>/dev/null; then
        return 0
    fi

    local install_dir="${CONDA_INSTALL_DIR:-${HOME}/miniconda3}"

    if [ -x "${install_dir}/bin/conda" ]; then
        echo "  Found existing conda at ${install_dir} (not in PATH); sourcing it."
    else
        echo "  conda not found. Auto-installing Miniconda to ${install_dir}..."
        local arch mc_arch installer
        arch="$(uname -m)"
        case "${arch}" in
            x86_64)        mc_arch="x86_64" ;;
            aarch64|arm64) mc_arch="aarch64" ;;
            *) echo "  ERROR: unsupported architecture '${arch}'."; exit 1 ;;
        esac

        installer="$(mktemp /tmp/miniconda_XXXXXX.sh)"
        local url_mirror="https://mirrors.tencent.com/anaconda/miniconda/Miniconda3-latest-Linux-${mc_arch}.sh"
        local url_official="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${mc_arch}.sh"

        local dl=""
        if command -v curl &>/dev/null; then
            dl="curl -fsSL -o"
        elif command -v wget &>/dev/null; then
            dl="wget -qO"
        else
            echo "  ERROR: need curl or wget to download Miniconda."; exit 1
        fi

        echo "  Downloading Miniconda (${mc_arch})..."
        ${dl} "${installer}" "${url_mirror}" || ${dl} "${installer}" "${url_official}" || {
            echo "  ERROR: failed to download Miniconda installer."; rm -f "${installer}"; exit 1
        }

        echo "  Running installer (silent, -b)..."
        bash "${installer}" -b -p "${install_dir}"
        rm -f "${installer}"
    fi

    # Make conda usable in THIS shell (conda.sh may reference unset vars -> relax -u).
    set +u
    # shellcheck disable=SC1091
    source "${install_dir}/etc/profile.d/conda.sh"
    set -u

    command -v conda &>/dev/null || { echo "  ERROR: conda still unavailable after install."; exit 1; }
    echo "  (tip: run 'conda init bash' once if you want conda in future shells)"
}

ensure_conda
command -v nvidia-smi &>/dev/null || { echo "  ERROR: nvidia-smi not found."; exit 1; }

# [CHANGED] Resolve conda base now that conda is guaranteed available.
CONDA_BASE="$(conda info --base)"

echo "  conda:          $(conda --version)"
echo "  conda base:     ${CONDA_BASE}"
echo "  NVIDIA driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo ""

# ======================== Step 1: Create conda environment ========================
echo "[Step 1/6] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."

# conda >= 24/26 blocks non-interactive `conda create` on the Anaconda default
# channels until their Terms of Service are accepted. Accept them up-front
# (no-op / harmlessly ignored on older conda that lacks the `tos` subcommand).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if conda env list | grep -q "/${ENV_NAME}$"; then
    echo "  Environment '${ENV_NAME}' already exists."
    if [ "${FORCE_RECREATE}" = "true" ]; then
        echo "  --force: recreating from scratch..."
        conda env remove -n "${ENV_NAME}" -y
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    else
        echo "  Keeping existing environment. Will update packages. (use --force to recreate)"
    fi
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# [CHANGED] env's pip/python derived from conda base
ENV_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
ENV_PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
[ -x "${ENV_PYTHON}" ] || { echo "  ERROR: ${ENV_PYTHON} not found after create."; exit 1; }
echo "  Python: $(${ENV_PYTHON} --version)"
echo "[Step 1/6] Done."
echo ""

# ======================== Step 2: PyTorch stack ========================
echo "[Step 2/6] Installing PyTorch ${TORCH_VERSION}+${CUDA_TAG}..."
${ENV_PIP} install --no-cache-dir \
    "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" "torchvision==0.21.0" \
    "torchdata==0.11.0" "triton==${TRITON_VERSION}" \
    --index-url "${TORCH_INDEX}" --extra-index-url "${MIRROR}"
${ENV_PYTHON} -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.version.cuda}')"
echo "[Step 2/6] Done."
echo ""

# ======================== Step 3: flash-attn & xformers ========================
echo "[Step 3/6] Installing flash-attn and xformers..."
${ENV_PIP} install --no-cache-dir "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation -i "${MIRROR}" || {
    echo "  WARNING: flash-attn wheel not found, building from source (10-30 min)..."
    MAX_JOBS=8 ${ENV_PIP} install --no-cache-dir "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
}
${ENV_PIP} install --no-cache-dir "xformers==${XFORMERS_VERSION}" --index-url "${TORCH_INDEX}" --extra-index-url "${MIRROR}" || {
    ${ENV_PIP} install --no-cache-dir "xformers==${XFORMERS_VERSION}" -i "${MIRROR}"
}
echo "[Step 3/6] Done."
echo ""

# ======================== Step 4: vLLM ========================
echo "[Step 4/6] Installing vLLM ${VLLM_VERSION}..."
${ENV_PIP} install --no-cache-dir "vllm==${VLLM_VERSION}" -i "${MIRROR}"
${ENV_PYTHON} -c "import vllm; print(f'  vllm={vllm.__version__}')"
echo "[Step 4/6] Done."
echo ""

# ======================== Step 5: Remaining packages ========================
echo "[Step 5/6] Installing remaining pip packages from ${REQUIREMENTS_FILE}..."
if [ -f "${REQUIREMENTS_FILE}" ]; then
    grep -vE "^(torch==|torchaudio==|torchvision==|torchdata==|triton==|flash.attn==|xformers==|vllm==)" \
        "${REQUIREMENTS_FILE}" \
        | grep -vE "^packaging @" \
        | grep -vE "^(nvidia-|typing_extensions|typing-inspection|mpmath|MarkupSafe|certifi|charset-normalizer|filelock|fsspec|idna|Jinja2|networkx|pillow|sympy|urllib3|requests|einops|numpy==|pyasn1)" | \
    ${ENV_PIP} install --no-cache-dir -r /dev/stdin -i "${MIRROR}" || {
        echo "  WARNING: some packages failed with deps. Retrying with --no-deps..."
        grep -vE "^(torch==|torchaudio==|torchvision==|torchdata==|triton==|flash.attn==|xformers==|vllm==)" \
            "${REQUIREMENTS_FILE}" \
            | grep -vE "^packaging @" \
            | grep -vE "^(nvidia-|typing_extensions|typing-inspection|mpmath|MarkupSafe|certifi|charset-normalizer|filelock|fsspec|idna|Jinja2|networkx|pillow|sympy|urllib3|requests|einops|numpy==|pyasn1)" | \
        ${ENV_PIP} install --no-cache-dir --no-deps -r /dev/stdin -i "${MIRROR}" || true
    }
else
    echo "  ERROR: ${REQUIREMENTS_FILE} not found."; exit 1
fi
echo "[Step 5/6] Done."
echo ""

# ======================== Step 6: EasyOPD PYTHONPATH hook ========================
echo "[Step 6/6] Setting up EasyOPD PYTHONPATH activate hook..."
ACTIVATE_SCRIPT="${CONDA_BASE}/envs/${ENV_NAME}/etc/conda/activate.d/easyopd.sh"
mkdir -p "$(dirname "${ACTIVATE_SCRIPT}")"
cat > "${ACTIVATE_SCRIPT}" << EOF
#!/bin/bash
# Auto-set EasyOPD PYTHONPATH on conda activate (this workspace)
export PYTHONPATH="${EASYOPD_ROOT}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
EOF
chmod +x "${ACTIVATE_SCRIPT}"
echo "  Created activate hook: ${ACTIVATE_SCRIPT}"
echo "[Step 6/6] Done."
echo ""

# ======================== Verification ========================
echo "============================================================"
echo "  Verifying..."
echo "============================================================"
${ENV_PYTHON} -c "import torch; print(f'  torch:        {torch.__version__} (CUDA {torch.version.cuda})')" 2>/dev/null || echo "  torch:        FAILED"
${ENV_PYTHON} -c "import flash_attn; print(f'  flash_attn:   {flash_attn.__version__}')" 2>/dev/null || echo "  flash_attn:   FAILED"
${ENV_PYTHON} -c "import vllm; print(f'  vllm:         {vllm.__version__}')" 2>/dev/null || echo "  vllm:         FAILED"
${ENV_PYTHON} -c "import ray; print(f'  ray:          {ray.__version__}')" 2>/dev/null || echo "  ray:          FAILED"
${ENV_PYTHON} -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null || echo "  transformers: FAILED"
${ENV_PYTHON} -c "import pandas; print(f'  pandas:       {pandas.__version__}')" 2>/dev/null || echo "  pandas:       FAILED"
echo ""
echo "  Done. Run SDPO with:"
echo "    PYTHON=${ENV_PYTHON} \\"
echo "    bash experiments/04_self_opd/methods/sdpo/launch.sh trainer.total_training_steps=5 trainer.save_freq=5"
echo "============================================================"
