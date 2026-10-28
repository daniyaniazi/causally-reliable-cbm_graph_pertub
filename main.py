import random
import numpy as np
import torch
import os
import warnings
import hydra
import pickle
# import jpype
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

# pytorch lightning
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# data loading
from src.data.dataset_block import get_dataset

# training and utils
from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams
from src.data.utils import static_graph_collate
from src.metrics import hamming_distance
from src.plots import maybe_plot_graph
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph
from src.utils import finetune_model

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
    
def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_and_freeze_c2bm_modules(
    engine,
    checkpoint_path: str,
    load_input_encoder: bool,
    freeze_input_encoder: bool,
    load_concept_encoders: bool,
    freeze_concept_encoders: bool,
    load_concept_scorers: bool,
    freeze_concept_scorers: bool,
) -> None:
    """Load selected C2BM modules from a Lightning checkpoint and freeze them."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Frozen-module runs require experiment.encoder_checkpoint to point to "
            f"an existing original-model checkpoint; got: {checkpoint_path!r}"
        )
    if not hasattr(engine, "model"):
        raise TypeError("The configured engine does not expose a model")
    if load_input_encoder and not hasattr(engine.model, "encoder"):
        raise TypeError("The configured model does not expose a C2BM input encoder")
    if (load_concept_encoders or freeze_concept_encoders or load_concept_scorers or freeze_concept_scorers) and not hasattr(engine.model, "concept_encoders"):
        raise TypeError("The configured model does not expose C2BM concept encoders")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = {}
    concept_states = {}
    for name, value in state_dict.items():
        if name.startswith("model.encoder."):
            encoder_state[name[len("model.encoder."):]] = value
        elif name.startswith("encoder."):
            encoder_state[name[len("encoder."):]] = value
        elif name.startswith("model.concept_encoders."):
            tail = name[len("model.concept_encoders."):]
            concept_name, separator, parameter_name = tail.partition(".")
            if separator:
                concept_states.setdefault(concept_name, {})[parameter_name] = value
        elif name.startswith("concept_encoders."):
            tail = name[len("concept_encoders."):]
            concept_name, separator, parameter_name = tail.partition(".")
            if separator:
                concept_states.setdefault(concept_name, {})[parameter_name] = value

    if load_input_encoder and not encoder_state:
        raise KeyError(
            "No C2BM input encoder weights found in checkpoint. Expected keys such "
            "as 'state_dict.model.encoder.*'. Check that this is a C2BM checkpoint."
        )
    if (load_concept_encoders or freeze_concept_encoders or load_concept_scorers or freeze_concept_scorers) and not concept_states:
        raise KeyError(
            "No C2BM concept encoder weights found in checkpoint. Expected keys "
            "such as 'state_dict.model.concept_encoders.*'."
        )

    if load_input_encoder:
        engine.model.encoder.load_state_dict(encoder_state, strict=True)
        if freeze_input_encoder:
            engine.model.encoder.requires_grad_(False)
            engine.model.encoder.eval()
            print(f"Loaded and froze C2BM input encoder ({len(encoder_state)} tensors)")
        else:
            print(f"Initialized trainable C2BM input encoder ({len(encoder_state)} tensors)")

    if load_concept_encoders or freeze_concept_encoders or load_concept_scorers or freeze_concept_scorers:
        expected_names = set(engine.model.concept_encoders.keys())
        checkpoint_names = set(concept_states)
        if expected_names != checkpoint_names:
            raise KeyError(
                "Checkpoint concept modules do not match this model. "
                f"Missing={sorted(expected_names - checkpoint_names)}, "
                f"extra={sorted(checkpoint_names - expected_names)}"
            )
        for concept_name, module in engine.model.concept_encoders.items():
            module_state = concept_states[concept_name]
            if load_concept_encoders:
                c_encoder_state = {
                    key[len("c_encoder."):]: value
                    for key, value in module_state.items()
                    if key.startswith("c_encoder.")
                }
                if not c_encoder_state:
                    raise KeyError(f"Checkpoint has no c_encoder weights for {concept_name}")
                module.c_encoder.load_state_dict(c_encoder_state, strict=True)
            if load_concept_scorers:
                c_scorer_state = {
                    key[len("c_scorer."):]: value
                    for key, value in module_state.items()
                    if key.startswith("c_scorer.")
                }
                if not c_scorer_state:
                    raise KeyError(f"Checkpoint has no c_scorer weights for {concept_name}")
                module.c_scorer.load_state_dict(c_scorer_state, strict=True)
            if freeze_concept_encoders:
                module.c_encoder.requires_grad_(False)
            if freeze_concept_scorers:
                module.c_scorer.requires_grad_(False)
        component_names = []
        if freeze_concept_encoders:
            component_names.append("c_encoder MLPs")
        if freeze_concept_scorers:
            component_names.append("c_scorer layers")
        print(f"Loaded {len(concept_states)} concept modules and froze {' and '.join(component_names)}")

@hydra.main(config_path="conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.mkdir('results')
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg)

    # get the causal graph
    if cfg.dataset.load_true_graph:
        graph = true_graph
    else:
        if cfg.dataset.load_graph:
            # Use an explicit graph path for graph robustness experiments; otherwise
            # load the dataset's graph from its normal cache directory.
            graph_path = cfg.dataset.get("graph_path") or os.path.join(dataset_directory, "graph.pkl")
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
        else:
            # These pipelines are optional when a graph is supplied from cache.
            from src.causal_discovery.causal_discovery_block import causal_discovery
            from src.completion.completion_block import complete_graph_with_llm

            # estimate causal graph with causal structural learning algorithms
            predicted_graph = causal_discovery(cfg, dataset, true_graph)
            if true_graph is not None:
                hamming = hamming_distance(true_graph, predicted_graph)
                print('(after CD) structural hamming distance: ', hamming)    

            # complete the causal graph with LLM and RAG
            completed_graph = complete_graph_with_llm(cfg, predicted_graph, cfg.dataset.name)
            if true_graph is not None:
                hamming = hamming_distance(true_graph, completed_graph)
                print('(after LLM + RAG) structural hamming distance: ', hamming)
            graph = completed_graph

            # save graph
            with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
                pickle.dump(graph, f)

    # fix the graph
    # (part 1): remove bidirected and undirected edges + add virtual nodes
    # edge can only be directed at this stage, the following function is just here in 
    # case the CD + LLM + RAG pipeline is modified and could produce bidirected or undirected edges
    graph, dataset = remove_problematic_edges(graph, dataset)
    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1
    # (part 2): remove cycles
    graph = remove_cycles(graph, y_index)

    if true_graph is not None:
        hamming = hamming_distance(true_graph, graph)
        print('(after fix) structural hamming distance: ', hamming)
    maybe_plot_graph(graph, 'fixed_graph')

    # use the graph to define an intervention policy at test time
    interv_policy, ip_names = get_intervention_policy(cfg.policy, graph, true_graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, dataset)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)
    
    ############ model block ########################################################################################
    [dataset.data[split].register_graph(graph) for split in dataset.data]
    train_dataloader = DataLoader(dataset.data['train'], 
                                  batch_size=cfg.dataset.batch_size, 
                                  collate_fn=static_graph_collate,
                                  num_workers=cfg.dataset.num_workers)
    val_dataloader = DataLoader(dataset.data['val'], 
                                batch_size=cfg.dataset.batch_size, 
                                collate_fn=static_graph_collate,
                                num_workers=cfg.dataset.num_workers)
    test_dataloader = DataLoader(dataset.data['test'], 
                                 batch_size=cfg.dataset.batch_size, 
                                 collate_fn=static_graph_collate,
                                 num_workers=cfg.dataset.num_workers)
    
    engine = instantiate(cfg.engine)
    experiment_cfg = cfg.get("experiment", {})
    freeze_input_encoder = experiment_cfg.get("freeze_c2bm_encoder", False)
    load_input_encoder = freeze_input_encoder or experiment_cfg.get("initialize_c2bm_encoder", False)
    freeze_concept_encoders = experiment_cfg.get("freeze_concept_encoders", False)
    freeze_concept_scorers = experiment_cfg.get("freeze_concept_scorers", False)
    initialize_concept_modules = freeze_concept_encoders or freeze_concept_scorers
    load_concept_encoders = experiment_cfg.get("initialize_concept_encoders", initialize_concept_modules)
    load_concept_scorers = experiment_cfg.get("initialize_concept_scorers", initialize_concept_modules)
    if load_input_encoder or load_concept_encoders or freeze_concept_encoders or load_concept_scorers or freeze_concept_scorers:
        initialize_and_freeze_c2bm_modules(
            engine,
            experiment_cfg.get("encoder_checkpoint"),
            load_input_encoder=load_input_encoder,
            freeze_input_encoder=freeze_input_encoder,
            load_concept_encoders=load_concept_encoders,
            freeze_concept_encoders=freeze_concept_encoders,
            load_concept_scorers=load_concept_scorers,
            freeze_concept_scorers=freeze_concept_scorers,
        )
    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        # ---- train
        trainer.fit(engine, train_dataloader, val_dataloader)
        # ---- finetune the encoder (eventually)
        if cfg.dataset.loader.ftune_size > 0: 
            trainer, engine = finetune_model(cfg, engine, dataset)
        # ----- test
        trainer.test(engine, test_dataloader, ckpt_path='best')
        trainer.logger.finalize("success")
    finally:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()
    ############################################################################################


if __name__ == "__main__":
    main()
    print('done')
