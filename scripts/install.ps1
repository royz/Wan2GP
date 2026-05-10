# This install script is for Windows 11 with RTX 5090

uv venv --python 3.11
.venv\Scripts\activate
python --version

# Manually install cuda toolkit - https://developer.nvidia.com/cuda-13-1-0-download-archive

# install base packages
uv pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt

# Triton
uv pip install triton-windows

# Sage2 Attention
uv pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl

# Sparge Attention
uv pip install https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post4/spas_sage_attn-0.1.0%2Bcu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl

# Flash Attention
uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl

# GGUF llama.cpp CUDA Kernels
uv pip install https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2+torch210cu13py311-cp311-cp311-win_amd64.whl

# INT4 / FP4 quantized support - Lightx2v NVP4
uv pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-win_amd64.whl

# INT4 / FP4 quantized support - Nunchaku INT4/FP4
uv pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl

# Install hf_xet for hugging face download optimization
uv pip install hf_xet

<# 
crete envs.json with the following content:
{
    "active": "uv_venv",
    "envs": {
        "uv_venv": {
            "type": "uv",
            "path": ".\\.venv"
        }
    }
}
#>