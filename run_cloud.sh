#!/bin/bash
# run_cloud.sh
# ═══════════════════════════════════════════════════════════════
# Script di setup e training per server cloud (Vast.ai / Lambda)
#
# Uso:
#   bash run_cloud.sh setup      # installa dipendenze
#   bash run_cloud.sh gan        # baseline GAN (Sarrut 2019)
#   bash run_cloud.sh nsf        # NSF con reparametrizzazione sferica
#   bash run_cloud.sh cfm        # CFM con reparametrizzazione sferica
#   bash run_cloud.sh all        # tutti e tre in sequenza
#   bash run_cloud.sh export     # esporta .pth per GATE dopo il training
# ═══════════════════════════════════════════════════════════════

set -e
PHASE_SPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PHASE_SPACE_DIR"

DATA_TRAIN="data/elekta_6mv_train.h5"
DATA_EVAL="data/elekta_6mv_eval.h5"

# ── Verifica dati ──────────────────────────────────────────────
check_data() {
    if [ ! -f "$DATA_TRAIN" ]; then
        echo "[ERROR] File di training non trovato: $DATA_TRAIN"
        echo "Eseguire prima: python data/read_iaea_phsp.py --input ... --output $DATA_TRAIN"
        exit 1
    fi
    echo "  ✓ Dati training: $DATA_TRAIN"
}

# ── Setup ──────────────────────────────────────────────────────
cmd_setup() {
    echo "=== Setup dipendenze ==="
    pip install torch nflows zuko torchdiffeq h5py \
                scipy scikit-learn matplotlib SimpleITK --quiet
    python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
    python -c "import nflows, zuko, torchdiffeq; print('flow libs OK')"
    echo "=== Setup completato ==="
}

# ── GAN baseline (Sarrut 2019) ─────────────────────────────────
cmd_gan() {
    check_data
    echo "=== GAN baseline (Sarrut 2019) ==="
    echo "  Parametri: H=400, z_dim=6, lr=1e-5, RMSProp, 80k iter"
    python baseline_gaga.py \
        --hdf5_train "$DATA_TRAIN" \
        --n_epochs   62 \
        --batch_size 10000 \
        --h_dim      400 \
        --z_dim      6 \
        --lr         1e-5 \
        --n_critic   4 \
        --log_every  1000 \
        --save_every 10000 \
        --output_dir outputs/baseline_GAN_iaea
    echo "=== GAN completato ==="
}

# ── NSF (Neural Spline Flow) ───────────────────────────────────
cmd_nsf() {
    check_data
    echo "=== NSF con reparametrizzazione sferica ==="
    echo "  K=12 transforms, 16 bin, spherical (theta,phi), tail_bound=7"
    python train.py \
        --model        nsf \
        --data_path    "$DATA_TRAIN" \
        --spherical \
        --epochs       1500 \
        --batch_size   16384 \
        --lr           3e-5 \
        --n_transforms 12 \
        --n_bins       16 \
        --hidden_dim   256 \
        --tail_bound   7.0 \
        --val_every    10 \
        --save_every   100 \
        --run_name     nsf_iaea_spherical
    echo "=== NSF completato ==="
}

# ── CFM (Conditional Flow Matching) ───────────────────────────
cmd_cfm() {
    check_data
    echo "=== CFM con reparametrizzazione sferica ==="
    echo "  6 layers, hidden_dim=512, lr=5e-5, 800 epoche"
    python train.py \
        --model     cfm \
        --data_path "$DATA_TRAIN" \
        --spherical \
        --epochs    800 \
        --batch_size 16384 \
        --lr        5e-5 \
        --n_layers  6 \
        --hidden_dim 512 \
        --val_every  10 \
        --save_every 50 \
        --run_name  cfm_iaea_spherical
    echo "=== CFM completato ==="
}

# ── Export .pth per GATE ───────────────────────────────────────
cmd_export() {
    echo "=== Export modelli per GATE 10 ==="

    for model in gan nsf cfm; do
        # Trova l'ultima run
        case $model in
            gan) dir="outputs/baseline_GAN_iaea"; ckpt="$dir/best_model.pt" ;;
            nsf) dir=$(ls -d outputs/nsf_iaea_spherical* 2>/dev/null | tail -1); ckpt="$dir/best_model.pt" ;;
            cfm) dir=$(ls -d outputs/cfm_iaea_spherical* 2>/dev/null | tail -1); ckpt="$dir/best_model.pt" ;;
        esac

        if [ ! -f "$ckpt" ]; then
            echo "  [SKIP] $model: checkpoint non trovato ($ckpt)"
            continue
        fi

        stats="$dir/normalization_stats.json"
        out="outputs/gate_models/${model}_elekta_6mv.pth"
        mkdir -p outputs/gate_models

        echo "  Esporto $model → $out"
        python gate_integration/save_for_gate.py \
            --checkpoint "$ckpt" \
            --model      "$model" \
            --stats_path "$stats" \
            --out        "$out"
    done

    echo "=== Export completato ==="
    echo "  File .pth in outputs/gate_models/"
}

# ── All ────────────────────────────────────────────────────────
cmd_all() {
    cmd_setup
    cmd_gan
    cmd_nsf
    cmd_cfm
    cmd_export
}

# ── Dispatch ───────────────────────────────────────────────────
case "${1:-help}" in
    setup)  cmd_setup  ;;
    gan)    cmd_gan    ;;
    nsf)    cmd_nsf    ;;
    cfm)    cmd_cfm    ;;
    export) cmd_export ;;
    all)    cmd_all    ;;
    *)
        echo "Uso: bash run_cloud.sh [setup|gan|nsf|cfm|export|all]"
        ;;
esac
