#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/causally-reliable-cbm_graph_pertub"
PYTHON_BIN="/home/dani00003/miniconda3/envs/c2bm/bin/python"
CONDA_ENV_BIN="/home/dani00003/miniconda3/envs/c2bm/bin"
DATA_DIR="/home/dani00003/.cache/c2bm/CelebA/celeba"
GRAPH_CACHE="/home/dani00003/.cache/c2bm/celeba"
GRAPH_SOURCE="$PROJECT_ROOT/learned_graphs/celeba/graph.pkl"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: c2bm Python environment not found: $PYTHON_BIN" >&2
    exit 127
fi
if [[ ! -d "$DATA_DIR/img_align_celeba" || ! -f "$DATA_DIR/list_attr_celeba.txt" || ! -f "$DATA_DIR/list_eval_partition.txt" ]]; then
    echo "ERROR: CelebA images/annotations not found under $DATA_DIR" >&2
    exit 2
fi
if [[ ! -f "$GRAPH_SOURCE" ]]; then
    echo "ERROR: repository graph not found: $GRAPH_SOURCE" >&2
    exit 2
fi

mkdir -p "$GRAPH_CACHE"
if [[ ! -f "$GRAPH_CACHE/graph.pkl" ]]; then
    cp "$GRAPH_SOURCE" "$GRAPH_CACHE/graph.pkl"
fi

cd "$PROJECT_ROOT"
export PATH="$CONDA_ENV_BIN:$PATH"
echo "HOST=$(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
"$PYTHON_BIN" -V
nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import pytorch_lightning, torchvision, hydra; print('deps_ok=1')"

echo "Running C2BM CelebA reproduction (seeds 1-5)..."
"$PYTHON_BIN" main.py --config-name default --multirun \
    dataset=celeba model=c2bm seed=1 \
    dataset.batch_size=512 \
    dataset.load_embeddings=false \
    +dataset.load_graph=true +dataset.load_true_graph=false \
    +policy=nodes_true \
    model.hidden_size=64 model.concept_hidden_size=8 \
    model.concept_loss_weight=0.8 model.prop_type=equations \
    engine.optim_kwargs.lr=0.00075 engine.intervention_prob=0.8 \
    trainer.max_epochs=500 trainer.patience=30 +trainer.devices='[0]' \
    +trainer.logger=null

echo "CelebA C2BM sweep finished. See $PROJECT_ROOT/outputs/multirun/"
