#!/usr/bin/env bash
# Set up the whole project on a new (or wiped) machine, from GitHub + the two Hugging Face repos.
#
#   git clone git@github.com:SakibHossain9999/Brain-Tumor-Segmentation-Mlops.git brain-npr
#   cd brain-npr
#   bash scripts/setup_machine.sh            # everything (GPU machine)
#
# Options:
#   --cpu           install the CPU build of PyTorch (no NVIDIA GPU)
#   --env NAME      conda environment name (default: brain-npr)
#   --skip-env      use the existing conda environment, don't install packages
#   --skip-backup   don't restore the training backup (checkpoints, MLflow, predictions, logs)
#   --skip-data     don't download the MSD dataset (~7 GB; only needed to retrain or re-evaluate)
#   --skip-web      don't install the viewer's Node.js packages
#
# Needs: conda, git and Node.js 22 (for the viewer). No Hugging Face account: both repos are public.
set -euo pipefail

PUBLIC_REPO=MohammadSakibHossain/brain-tumor-segmentation-mlops
BACKUP_REPO=MohammadSakibHossain/brain-tumor-segmentation-training
ENV_NAME=brain-npr
TORCH_INDEX=https://download.pytorch.org/whl/cu128
SKIP_ENV=0 SKIP_BACKUP=0 SKIP_DATA=0 SKIP_WEB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
    --env) ENV_NAME="$2"; shift ;;
    --skip-env) SKIP_ENV=1 ;;
    --skip-backup) SKIP_BACKUP=1 ;;
    --skip-data) SKIP_DATA=1 ;;
    --skip-web) SKIP_WEB=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 1 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."
step() { echo; echo "==== $*"; }

step "1/5 Python environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
if [ "$SKIP_ENV" = 0 ]; then
  conda env list | grep -qE "^$ENV_NAME\s" || conda create -n "$ENV_NAME" python=3.11 -y
  conda activate "$ENV_NAME"
  pip install torch==2.11.0 --index-url "$TORCH_INDEX"
  pip install -r requirements/train.txt
  pip install -e pipeline
else
  conda activate "$ENV_NAME"
  echo "using the existing environment"
fi

step "2/5 Models, sample brains and gallery ($PUBLIC_REPO)"
python scripts/hub.py download --repo "$PUBLIC_REPO"
ln -sfn ../../output/samples web/public/samples   # the viewer offers the samples for download

step "3/5 Training backup: checkpoints, MLflow history, predictions, logs ($BACKUP_REPO)"
if [ "$SKIP_BACKUP" = 1 ]; then echo "skipped"
elif [ -e mlflow.db ] || [ -e mlruns ]; then echo "mlflow.db / mlruns already here: skipped (move them away to restore)"
elif ! python -c "from huggingface_hub import HfApi; HfApi().repo_info('$BACKUP_REPO')" >/dev/null 2>&1; then
  echo "skipped: the backup repo can't be reached (made private, or no network)."
  echo "Everything needed to run the app is already here; the backup only matters for continuing training."
else python scripts/backup.py restore --repo "$BACKUP_REPO" --to .
fi

step "4/5 MSD dataset (~7 GB)"
if [ "$SKIP_DATA" = 1 ]; then echo "skipped"
elif [ -f data/msd/Task01_BrainTumour/dataset.json ]; then echo "already here"
else python scripts/download_msd.py
fi

step "5/5 Viewer packages"
if [ "$SKIP_WEB" = 1 ]; then echo "skipped"
else (cd web && npm ci)
fi

step "Check: unit tests"
python -m pytest -q tests

cat <<EOF

Done. To run the app (two terminals, environment: conda activate $ENV_NAME):
  uvicorn api.main:app --port 8000            # upload service
  cd web && npm run dev                       # viewer: http://localhost:5173
Training history:
  mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000
EOF
