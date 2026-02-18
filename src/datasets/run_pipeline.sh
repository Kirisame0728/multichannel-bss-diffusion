cat > run_pipeline.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-click pipeline for generating mix2 (2sp-6ch) reverb dataset.

Required:
  --librispeech_dir PATH     LibriSpeech root (contains test-clean/, train-clean-360/, SPEAKERS.TXT, ...)
  --wham_dir PATH            WHAM noise root
  --out_dir PATH             Output dataset root (will create early/tail/observation under it)

Optional:
  --subsets LIST             Comma-separated LibriSpeech subsets to include in metadata + rendering.
                             Default: "test-clean,train-clean-360"
  --fs INT                   Output sample rate. Default: 8000
  --n_src INT                Number of sources. Default: 2
  --seed_mix INT             Seed for create_mix_metadata.py. Default: 72
  --seed_reverb INT          Seed for create_reverb_params.py. Default: 17
  --dev_test_mixtures INT    Dev/test mixtures count per split (if script uses it). Default: 3000
  --add_noise                Add WHAM noise in rendering stage (requires noise_path/noise_gain in recipe)
  --noise_atten_db FLOAT     Noise attenuation in dB. Default: 12
  --overwrite                Overwrite existing outputs
  --strict                   Enable strict mode (fail fast on missing/invalid items)

Example:
  bash run_pipeline.sh \
    --librispeech_dir /mnt/f/LibriSpeech \
    --wham_dir /mnt/d/datasets/wham_noise/wham_noise \
    --out_dir /mnt/d/datasets/mix2_reverb_6ch \
    --subsets test-clean,train-clean-360 \
    --add_noise --overwrite
EOF
}

# Defaults
SUBSETS="test-clean,train-clean-360"
FS=8000
N_SRC=2
SEED_MIX=72
SEED_REVERB=17
DEV_TEST_MIXTURES=3000
ADD_NOISE=0
NOISE_ATTEN_DB=12
OVERWRITE=0
STRICT=0

LIBRISPEECH_DIR=""
WHAM_DIR=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --librispeech_dir) LIBRISPEECH_DIR="$2"; shift 2;;
    --wham_dir) WHAM_DIR="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --subsets) SUBSETS="$2"; shift 2;;
    --fs) FS="$2"; shift 2;;
    --n_src) N_SRC="$2"; shift 2;;
    --seed_mix) SEED_MIX="$2"; shift 2;;
    --seed_reverb) SEED_REVERB="$2"; shift 2;;
    --dev_test_mixtures) DEV_TEST_MIXTURES="$2"; shift 2;;
    --add_noise) ADD_NOISE=1; shift 1;;
    --noise_atten_db) NOISE_ATTEN_DB="$2"; shift 2;;
    --overwrite) OVERWRITE=1; shift 1;;
    --strict) STRICT=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${LIBRISPEECH_DIR}" || -z "${WHAM_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "[ERROR] --librispeech_dir, --wham_dir, --out_dir are required."
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MD_ROOT="${SCRIPT_DIR}/metadata"
MD_LIBRI="${MD_ROOT}/librispeech"
MD_WHAM="${MD_ROOT}/wham"
MD_MIX="${MD_ROOT}/mix"
PARAMS_DIR="${SCRIPT_DIR}/reverb_params/mix"

mkdir -p "${MD_LIBRI}" "${MD_WHAM}" "${MD_MIX}" "${PARAMS_DIR}"

OW_FLAG=""
[[ "${OVERWRITE}" -eq 1 ]] && OW_FLAG="--overwrite"

STRICT_FLAG=""
[[ "${STRICT}" -eq 1 ]] && STRICT_FLAG="--strict"

echo "[STEP 0] Config"
echo "  LIBRISPEECH_DIR=${LIBRISPEECH_DIR}"
echo "  WHAM_DIR=${WHAM_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SUBSETS=${SUBSETS}"
echo "  FS=${FS}  N_SRC=${N_SRC}"
echo "  SEED_MIX=${SEED_MIX}  SEED_REVERB=${SEED_REVERB}"
echo "  ADD_NOISE=${ADD_NOISE}  NOISE_ATTEN_DB=${NOISE_ATTEN_DB}"
echo "  OVERWRITE=${OVERWRITE}  STRICT=${STRICT}"

echo "[STEP 1] Create LibriSpeech metadata"
# Build explicit subset args: --<subset> <path/to/subset>
IFS=',' read -ra SUB_ARR <<< "${SUBSETS}"
EXPLICIT_ARGS=()
for s in "${SUB_ARR[@]}"; do
  EXPLICIT_ARGS+=( "--${s}" "${LIBRISPEECH_DIR}/${s}" )
done

python create_libri_metadata.py --explicit \
  --librispeech_dir "${LIBRISPEECH_DIR}" \
  "${EXPLICIT_ARGS[@]}" \
  --output_dir "${MD_LIBRI}" \
  ${OW_FLAG} \
  ${STRICT_FLAG}

echo "[STEP 2] Create WHAM metadata"
python create_wham_metadata.py \
  --wham_dir "${WHAM_DIR}" \
  --output_dir "${MD_WHAM}" \
  ${OW_FLAG} \
  ${STRICT_FLAG}

echo "[STEP 3] Create mix recipe metadata (mix2_*.csv)"
python create_mix_metadata.py \
  --librispeech_dir "${LIBRISPEECH_DIR}" \
  --librispeech_md_dir "${MD_LIBRI}" \
  --wham_dir "${WHAM_DIR}" \
  --wham_md_dir "${MD_WHAM}" \
  --metadata_outdir "${MD_MIX}" \
  --n_src "${N_SRC}" \
  --seed "${SEED_MIX}" \
  --dev_test_mixtures "${DEV_TEST_MIXTURES}" \
  ${OW_FLAG} \
  ${STRICT_FLAG}

echo "[STEP 4] Create reverb params (batch: mix2_*.csv)"
python create_reverb_params.py \
  --metadata_dir "${MD_MIX}" \
  --out_dir "${PARAMS_DIR}" \
  --seed "${SEED_REVERB}" \
  ${OW_FLAG} \
  --progress

echo "[STEP 5] Render dataset for each mix2_*.csv"
shopt -s nullglob
MIX_FILES=( "${MD_MIX}"/mix2_*.csv )
if [[ ${#MIX_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No mix2_*.csv found under: ${MD_MIX}"
  exit 1
fi

for md_csv in "${MIX_FILES[@]}"; do
  base="$(basename "${md_csv}")"              # mix2_test-clean.csv
  params_csv="${PARAMS_DIR}/${base}"          # matching params file
  if [[ ! -f "${params_csv}" ]]; then
    echo "[ERROR] Missing params_csv: ${params_csv}"
    exit 1
  fi

  # Derive split_dir from filename: mix2_<split>.csv -> <split>
  split="${base#mix2_}"
  split="${split%.csv}"

  echo "  -> render split_dir=${split}  metadata=${base}"

  RENDER_ARGS=( \
    --librispeech_dir "${LIBRISPEECH_DIR}" \
    --wham_dir "${WHAM_DIR}" \
    --metadata_csv "${md_csv}" \
    --params_csv "${params_csv}" \
    --out_dir "${OUT_DIR}" \
    --split_dir "${split}" \
    --fs "${FS}" \
    --n_src "${N_SRC}" \
  )

  if [[ "${ADD_NOISE}" -eq 1 ]]; then
    RENDER_ARGS+=( --add_noise --noise_atten_db "${NOISE_ATTEN_DB}" )
  fi

  python create_mix_reverb_dataset.py "${RENDER_ARGS[@]}"
done

echo "[DONE] Dataset generated under: ${OUT_DIR}"
BASH

chmod +x run_pipeline.sh
echo "Created: run_pipeline.sh"
