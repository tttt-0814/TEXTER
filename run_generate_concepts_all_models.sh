#!/usr/bin/env bash
set -euo pipefail

MODELS=("resnet50" "resnet18" "vit" "dino_vits8" "dino_resnet50")
BASE_OUTPUT_DIR="Concepts/ImageNet/val"
DATA_ROOT="..."
OPENAI_API_KEY="..."

usage() {
  cat <<'EOF'
Usage:
  ./run_generate_concepts_all_models.sh [options] [-- <generate_concepts.py args...>]

Options:
  --base_output_dir DIR   Base output directory (default: Concepts/ImageNet/val)
  --models CSV            Comma-separated model names to run
  --data_root DIR         ImageNet root directory passed to generate_concepts.py
  --openai_api_key KEY    OpenAI API key passed to generate_concepts.py
  -h, --help              Show this help

Examples:
  ./run_generate_concepts_all_models.sh -- --num_images 200
  ./run_generate_concepts_all_models.sh --models resnet50,vit -- --num_images 100
  ./run_generate_concepts_all_models.sh --data_root /path/to/ImageNet --openai_api_key sk-... -- --num_images 50
EOF
}

EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --base_output_dir)
      BASE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --models)
      IFS=',' read -r -a MODELS <<<"$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --openai_api_key)
      OPENAI_API_KEY="$2"
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

echo "[info] Ensuring pretrained models are available..."
bash download_pretrained_models.sh

for model in "${MODELS[@]}"; do
  out_dir="${BASE_OUTPUT_DIR}/${model}"
  cmd=(
    python generate_concepts.py
    --model_name "${model}"
    --output_dir "${out_dir}"
  )
  if [[ -n "${DATA_ROOT}" ]]; then
    cmd+=(--data_root "${DATA_ROOT}")
  fi
  if [[ -n "${OPENAI_API_KEY}" ]]; then
    cmd+=(--openai_api_key "${OPENAI_API_KEY}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[run] model=${model} output_dir=${out_dir}"
  "${cmd[@]}"
done

echo "[done] all model runs completed"
