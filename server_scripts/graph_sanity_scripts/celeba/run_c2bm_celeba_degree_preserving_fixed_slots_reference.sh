#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/causally-reliable-cbm_graph_pertub"
PYTHON_BIN="/home/dani00003/miniconda3/envs/c2bm/bin/python"
CONDA_ENV_BIN="/home/dani00003/miniconda3/envs/c2bm/bin"
DATA_DIR="/home/dani00003/.cache/c2bm/CelebA/celeba"
BASE_GRAPH="$PROJECT_ROOT/learned_graphs/celeba/graph.pkl"
RUN_DIR="$PROJECT_ROOT/outputs/graph_sanity/celeba/degree_preserving_fixed_slots_reference/reference_seed_0/model_seed_1"
REFERENCE_CKPT="$RUN_DIR/fixed_slots_reference.ckpt"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: c2bm Python environment not found: $PYTHON_BIN" >&2
    exit 127
fi
if [[ ! -d "$DATA_DIR/img_align_celeba" || ! -f "$DATA_DIR/list_attr_celeba.txt" || ! -f "$DATA_DIR/list_eval_partition.txt" ]]; then
    echo "ERROR: CelebA images/annotations not found under $DATA_DIR" >&2
    exit 2
fi
if [[ ! -f "$BASE_GRAPH" ]]; then
    echo "ERROR: original learned graph not found: $BASE_GRAPH" >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: fixed-slot reference run directory already exists: $RUN_DIR" >&2
    echo "Move it aside or choose a new reference output directory before rerunning." >&2
    exit 2
fi

cd "$PROJECT_ROOT"
export PATH="$CONDA_ENV_BIN:$PATH"
echo "Training the original-graph all-node fixed-slot reference from scratch."
"$PYTHON_BIN" main.py --config-name default \
    dataset=celeba model=c2bm seed=1 \
    dataset.batch_size=512 dataset.load_embeddings=true \
    +dataset.num_workers=0 +dataset.load_graph=true +dataset.load_true_graph=false \
    ++dataset.graph_path="$BASE_GRAPH" \
    +policy=nodes_pred \
    model.hidden_size=64 model.concept_hidden_size=8 \
    model.concept_loss_weight=0.8 model.prop_type=equations \
    model.fixed_parent_slots=true model.fixed_parent_slots_scope=all \
    engine.optim_kwargs.lr=0.00075 engine.intervention_prob=0.8 \
    trainer.max_epochs=500 trainer.patience=30 "+trainer.devices=[0]" \
    +trainer.logger=null \
    hydra.run.dir="outputs/graph_sanity/celeba/degree_preserving_fixed_slots_reference/reference_seed_0/model_seed_1"

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
BEST_CKPT="$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'epoch=*.ckpt' -print | sort | tail -n 1)"
if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: no best epoch checkpoint found under $CHECKPOINT_DIR" >&2
    exit 3
fi
cp -- "$BEST_CKPT" "$REFERENCE_CKPT"
echo "Fixed-slot reference checkpoint saved at: $REFERENCE_CKPT"
