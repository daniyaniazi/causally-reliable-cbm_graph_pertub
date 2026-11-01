#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dani00003/causally-reliable-cbm_graph_pertub"
PYTHON_BIN="/home/dani00003/miniconda3/envs/c2bm/bin/python"
CONDA_ENV_BIN="/home/dani00003/miniconda3/envs/c2bm/bin"
DATA_DIR="/home/dani00003/.cache/c2bm/CelebA/celeba"
BASE_GRAPH="$PROJECT_ROOT/learned_graphs/celeba/graph.pkl"
REFERENCE_CKPT="$PROJECT_ROOT/outputs/graph_sanity/celeba/degree_preserving_fixed_slots_reference/reference_seed_0/model_seed_1/fixed_slots_reference.ckpt"

FAMILY="${1:-}"
GRAPH_SEED="${2:-}"
if [[ -z "$FAMILY" || -z "$GRAPH_SEED" ]]; then
    echo "Usage: $0 FAMILY GRAPH_SEED" >&2
    echo "FAMILY: full|frozen_encoder|frozen_cencoder|frozen_upstream|train_encoder_frozen_cencoder|train_cencoder_frozen_scorer" >&2
    exit 2
fi
if ! [[ "$GRAPH_SEED" =~ ^[1-5]$ ]]; then
    echo "ERROR: GRAPH_SEED must be 1-5; got '$GRAPH_SEED'" >&2
    exit 2
fi
case "$FAMILY" in
    full)
        OUTPUT_FAMILY="full"
        FREEZE_ENCODER=false; FREEZE_CENCODER=false; FREEZE_SCORER=false
        SKIP_TRAINING=false
        ;;
    frozen_encoder)
        OUTPUT_FAMILY="frozen_encoder"
        FREEZE_ENCODER=true; FREEZE_CENCODER=false; FREEZE_SCORER=false
        SKIP_TRAINING=false
        ;;
    frozen_cencoder)
        OUTPUT_FAMILY="frozen_cencoder"
        FREEZE_ENCODER=true; FREEZE_CENCODER=true; FREEZE_SCORER=false
        SKIP_TRAINING=false
        ;;
    frozen_upstream)
        OUTPUT_FAMILY="frozen_upstream"
        FREEZE_ENCODER=true; FREEZE_CENCODER=true; FREEZE_SCORER=true
        # With the propagators also frozen this is a pure graph-mask evaluation.
        SKIP_TRAINING=true
        ;;
    train_encoder_frozen_cencoder)
        OUTPUT_FAMILY="train_encoder_frozen_cencoder"
        FREEZE_ENCODER=false; FREEZE_CENCODER=true; FREEZE_SCORER=false
        SKIP_TRAINING=false
        ;;
    train_cencoder_frozen_scorer)
        OUTPUT_FAMILY="train_cencoder_frozen_scorer"
        FREEZE_ENCODER=true; FREEZE_CENCODER=false; FREEZE_SCORER=true
        SKIP_TRAINING=false
        ;;
    *)
        echo "ERROR: unsupported FAMILY '$FAMILY'" >&2
        exit 2
        ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: c2bm Python environment not found: $PYTHON_BIN" >&2
    exit 127
fi
if [[ ! -f "$REFERENCE_CKPT" ]]; then
    echo "ERROR: fixed-slot reference checkpoint missing: $REFERENCE_CKPT" >&2
    echo "Submit c2bm_celeba_degree_preserving_fixed_slots_reference.sub first and wait for it to finish." >&2
    exit 2
fi
if [[ ! -f "$BASE_GRAPH" || ! -d "$DATA_DIR/img_align_celeba" || ! -f "$DATA_DIR/list_attr_celeba.txt" || ! -f "$DATA_DIR/list_eval_partition.txt" ]]; then
    echo "ERROR: learned graph or CelebA data is missing" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
export PATH="$CONDA_ENV_BIN:$PATH"
PERTURBED_GRAPH="$PROJECT_ROOT/perturb_graph/celeba/topology_degree_preserving/random_seed_${GRAPH_SEED}.pkl"
"$PYTHON_BIN" perturb_graph/perturb_topology_degree_preserving.py \
    --input "$BASE_GRAPH" --output "$PERTURBED_GRAPH" --seed "$GRAPH_SEED" \
    --target-node Mouth_Slightly_Open

OUTPUT_DIR="degree_preserving_fixed_slots_fixedprop_${OUTPUT_FAMILY}"
echo "All-node fixed-slot, frozen-propagator run: family=$FAMILY, graph seed=$GRAPH_SEED"
echo "Shared source checkpoint: $REFERENCE_CKPT"
"$PYTHON_BIN" main.py --config-name fixed_slots_frozen_propagator \
    dataset=celeba model=c2bm seed=1 \
    dataset.batch_size=512 dataset.load_embeddings=true \
    +dataset.num_workers=0 +dataset.load_graph=true +dataset.load_true_graph=false \
    ++dataset.graph_path="$PERTURBED_GRAPH" \
    +policy=nodes_pred \
    model.hidden_size=64 model.concept_hidden_size=8 \
    model.concept_loss_weight=0.8 model.prop_type=equations \
    model.fixed_parent_slots=true model.fixed_parent_slots_scope=all \
    engine.optim_kwargs.lr=0.00075 engine.intervention_prob=0.8 \
    trainer.max_epochs=500 trainer.patience=30 "+trainer.devices=[0]" \
    +trainer.logger=null \
    experiment.encoder_checkpoint="'$REFERENCE_CKPT'" \
    experiment.freeze_c2bm_encoder="$FREEZE_ENCODER" \
    experiment.freeze_concept_encoders="$FREEZE_CENCODER" \
    experiment.freeze_concept_scorers="$FREEZE_SCORER" \
    experiment.skip_training="$SKIP_TRAINING" \
    hydra.run.dir="outputs/graph_sanity/celeba/${OUTPUT_DIR}/topology_seed_${GRAPH_SEED}/model_seed_1"

echo "Results: $PROJECT_ROOT/outputs/graph_sanity/celeba/${OUTPUT_DIR}/topology_seed_${GRAPH_SEED}/model_seed_1"
