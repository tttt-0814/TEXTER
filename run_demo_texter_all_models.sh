#!/usr/bin/env bash
set -euo pipefail

MODELS=("resnet50" "resnet18" "vit" "dino_vits8" "dino_resnet50")
BASE_OUTPUT_DIR="results_demo"
CONCEPTS_DATA_BASE="Concepts/ImageNet/val"
DATA_ROOT="..."
ALIGNER_MODEL_PATH="pretrained_models/aligner"
SAE_MODEL_PATH="pretrained_models/sae"
DEVICE="cuda"

usage() {
  cat <<'EOF'
Usage:
  ./run_demo_texter_all_models.sh [options] [-- <demo_texter.py args...>]

Options:
  --models CSV               Comma-separated model names to run
  --base_output_dir DIR      Base output directory passed to demo_texter.py
  --concepts_data_base DIR   Base concepts dir; uses <DIR>/<model_name> per run
  --data_root DIR            ImageNet root directory (required)
  --aligner_model_path DIR   Base aligner model directory (default: pretrained_models/aligner)
  --sae_model_path PATH      SAE checkpoint path or base directory (default: pretrained_models/sae)
  --device DEVICE            cuda or cpu (default: cuda)
  -h, --help                 Show this help

Examples:
  ./run_demo_texter_all_models.sh --data_root /path/to/ImageNet
  ./run_demo_texter_all_models.sh --models resnet50,vit --data_root /path/to/ImageNet -- --target_classes 20 --images_per_class 2
EOF
}

EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --models)
      IFS=',' read -r -a MODELS <<<"$2"
      shift 2
      ;;
    --base_output_dir)
      BASE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --concepts_data_base)
      CONCEPTS_DATA_BASE="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --aligner_model_path)
      ALIGNER_MODEL_PATH="$2"
      shift 2
      ;;
    --sae_model_path)
      SAE_MODEL_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "[error] --data_root is required"
  usage
  exit 1
fi

echo "[info] Ensuring pretrained models are available..."
bash download_pretrained_models.sh

for model in "${MODELS[@]}"; do
  concepts_dir="${CONCEPTS_DATA_BASE}/${model}"
  echo "[run] model=${model} output_dir=${BASE_OUTPUT_DIR} concepts=${concepts_dir}"
  python demo_texter.py \
    --model_name "${model}" \
    --data_root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --concepts_data_path "${concepts_dir}" \
    --aligner_model_path "${ALIGNER_MODEL_PATH}" \
    --sae_model_path "${SAE_MODEL_PATH}" \
    "${EXTRA_ARGS[@]}"
done

echo "[done] all demo_texter runs completed"
