#!/usr/bin/env bash
set -euo pipefail

URL_DEFAULT="https://drive.google.com/drive/folders/1LEL_5L1N6vpvY5A_CP39atcEsaqFDLn6?usp=sharing"
URL="${URL_DEFAULT}"
OUTPUT_DIR="pretrained_models"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  ./download_pretrained_models.sh [options]

Options:
  --url URL         Google Drive folder URL
  --output DIR      Output directory name (default: pretrained_models)
  --force           Re-download and overwrite existing output directory
  -h, --help        Show this help
EOF
}

while (($#)); do
  case "$1" in
    --url)
      URL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]] && [[ "${FORCE}" -ne 1 ]]; then
  echo "[skip] '${OUTPUT_DIR}' already exists. Use --force to re-download."
  exit 0
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "[info] gdown not found. Installing gdown..."
  python -m pip install gdown
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

echo "[info] Downloading from Google Drive folder..."
(
  cd "${tmpdir}"
  gdown --folder "${URL}"
)

source_dir=""

if [[ -d "${tmpdir}/pretrained_models" ]]; then
  source_dir="${tmpdir}/pretrained_models"
else
  mapfile -t top_dirs < <(find "${tmpdir}" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ "${#top_dirs[@]}" -eq 1 ]]; then
    source_dir="${top_dirs[0]}"
  else
    source_dir="${tmpdir}"
  fi
fi

# Flatten one extra level if downloaded files are wrapped in a parent directory.
if [[ "$(basename "${source_dir}")" != "pretrained_models" ]] && [[ -d "${source_dir}/pretrained_models" ]]; then
  source_dir="${source_dir}/pretrained_models"
fi

if [[ "${FORCE}" -eq 1 ]] && [[ -d "${OUTPUT_DIR}" ]]; then
  rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"
cp -a "${source_dir}/." "${OUTPUT_DIR}/"

echo "[done] Downloaded pretrained models to '${OUTPUT_DIR}'"
