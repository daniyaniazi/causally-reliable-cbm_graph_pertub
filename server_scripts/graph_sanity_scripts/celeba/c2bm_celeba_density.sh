#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/causally-reliable-cbm_graph_pertub"
PYTHON_BIN="/home/dani00003/miniconda3/envs/c2bm/bin/python"
CONDA_ENV_BIN="/home/dani00003/miniconda3/envs/c2bm/bin"
DATA_DIR="/home/dani00003/.cache/c2bm/CelebA/celeba"
BASE_GRAPH="$PROJECT_ROOT/learned_graphs/celeba/graph.pkl"

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

cd "$PROJECT_ROOT"
export PATH="$CONDA_ENV_BIN:$PATH"
GRAPH_SEED="${1:?Usage: c2bm_celeba_density.sh GRAPH_SEED}"
if ! [[ "$GRAPH_SEED" =~ ^[1-5]$ ]]; then
    echo "ERROR: GRAPH_SEED must be an integer from 1 to 5; got '$GRAPH_SEED'" >&2
    exit 2
fi

echo "HOST=$(hostname)"
echo "Model seed=1; density graph seed=$GRAPH_SEED; DAG density range=0.30-0.80"
"$PYTHON_BIN" -V
nvidia-smi || true
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"$PYTHON_BIN" -c "import pandas, networkx, pytorch_lightning, torchvision, hydra; print('deps_ok=1')"

DENSITY_GRAPH="$PROJECT_ROOT/perturb_graph/celeba/density/random_seed_${GRAPH_SEED}.pkl"
"$PYTHON_BIN" perturb_graph/perturb_density.py \
    --input "$BASE_GRAPH" \
    --output "$DENSITY_GRAPH" \
    --graph-seed "$GRAPH_SEED" \
    --target-node Mouth_Slightly_Open \
    --min-dag-density 0.30 \
    --max-dag-density 0.80 \
    --density-seed 31415 \
    --n-graphs 5

echo "Running C2BM on density graph seed $GRAPH_SEED (model seed 1)..."
"$PYTHON_BIN" main.py --config-name default \
    dataset=celeba model=c2bm seed=1 \
    dataset.batch_size=512 dataset.load_embeddings=true \
    +dataset.num_workers=0 \
    +dataset.load_graph=true +dataset.load_true_graph=false \
    ++dataset.graph_path="$DENSITY_GRAPH" \
    +policy=nodes_pred \
    model.hidden_size=64 model.concept_hidden_size=8 \
    model.concept_loss_weight=0.8 model.prop_type=equations \
    engine.optim_kwargs.lr=0.00075 engine.intervention_prob=0.8 \
    trainer.max_epochs=500 trainer.patience=30 +trainer.devices='[0]' \
    +trainer.logger=null \
    hydra.run.dir="outputs/graph_sanity/celeba/density_seed_${GRAPH_SEED}/model_seed_1"

echo "Done: density graph seed $GRAPH_SEED. Results are under $PROJECT_ROOT/outputs/graph_sanity/celeba/density_seed_${GRAPH_SEED}/"
