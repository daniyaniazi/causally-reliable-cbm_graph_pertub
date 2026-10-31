import torch
import torch.nn as nn
from src.models.layers.base import Dense, MLP
from src.models.layers.c_encoder import ConceptBlock
from src.models.layers.intervention import maybe_intervene
from src.utils import get_graph_levels, get_parents


class C2BM(nn.Module):
    """
    Causal version of the CEM model. 
    It propagates the information through a predefined causal graph.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 concept_hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_concept_encoder=1,
                 n_layers_propagation=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 normalize_concept_loss=False,
                 c_info={},
                 y_info={},
                 graph=None,
                 graph_labels=None,
                 prop_type='linear',
                 cat_latent=False,
                 fixed_parent_slots=False,
                 fixed_parent_slots_scope='all'):
        super(C2BM, self).__init__()
        
        # to be stored for every model
        self.has_concepts = True
        self.is_causal = True
        self.normalize_concept_loss = normalize_concept_loss

        # define concepts info parameters
        self.c_names = c_info['names'] # used later to retrieve which are concepts 
        self.y_names = y_info['names'] # and which are targerts
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        self.combo_info = {'names': c_info['names'] + y_info['names'],
                           'cardinality': c_info['cardinality'] + y_info['cardinality']}
        assert self.c_names + self.y_names == graph_labels

        # Encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)

        # Concept encoders, one for each concept
        self.concept_encoders = nn.ModuleDict()
        for name in self.combo_info['names']:
            self.concept_encoders[name] = ConceptBlock(input_size=hidden_size,
                                                       hidden_size=concept_hidden_size,
                                                       n_layers=n_layers_concept_encoder,
                                                       activation=activation,
                                                       c_cardinality=self.combo_info['cardinality'][self.combo_info['names'].index(name)])
        self.concept_hidden_size = concept_hidden_size
        self.concept_loss_weight = concept_loss_weight

        # get levels
        self.graph = torch.Tensor(graph).int()
        task_index = self.combo_info['names'].index(self.y_names[0])
        graph_levels = get_graph_levels(self.graph, task_index)
        self.roots = graph_levels[0]
        self.roots_info = {'names': [name for i, name in enumerate(self.combo_info['names']) 
                                     if i in self.roots], 
                           'cardinality': [card for i, card in enumerate(self.combo_info['cardinality']) 
                                           if i in self.roots]}
        if self.y_names[0] in self.roots_info['names']:
            raise ValueError('The target variable cannot be a root concept')
        
        # get list of propagators
        self.prop_type = prop_type
        self.cat_latent = cat_latent
        self.fixed_parent_slots = fixed_parent_slots
        self.fixed_parent_slots_scope = fixed_parent_slots_scope
        if self.fixed_parent_slots and prop_type != 'equations':
            raise ValueError("fixed_parent_slots is currently supported only with prop_type='equations'")
        if self.fixed_parent_slots_scope not in ('all', 'task'):
            raise ValueError("fixed_parent_slots_scope must be 'all' or 'task'")
        if prop_type in ['dense', 'mlp', 'embeddings', 'equations']:
            self.propagators = nn.ModuleDict()
            for i in range(1, len(graph_levels)):
                level = graph_levels[i]
                self.propagators[str(i)] = nn.ModuleDict()
                for node in level:
                    node_name = self.combo_info['names'][node]
                    parents = get_parents(self.graph, node).tolist()
                    node_cardinality = self.combo_info['cardinality'][node]
                    parents_cardinality = [self.combo_info['cardinality'][p] for p in parents]
                    use_fixed_parent_slots = (
                        self.fixed_parent_slots and
                        (self.fixed_parent_slots_scope == 'all' or node_name in self.y_names)
                    )
                    if use_fixed_parent_slots:
                        # Stable slots follow the dataset's global concept order.
                        # The task may receive any concept; a concept may receive
                        # any other concept, never itself or the task node.
                        slot_parents = [
                            p for p, possible_parent in enumerate(self.combo_info['names'])
                            if p != node and possible_parent in self.c_names
                        ]
                        self.fixed_parent_slot_indices = getattr(self, 'fixed_parent_slot_indices', {})
                        self.fixed_parent_slot_indices[node_name] = slot_parents
                        parents_cardinality = [self.combo_info['cardinality'][p] for p in slot_parents]
                    if prop_type == 'dense':
                        prop_input_size = sum(parents_cardinality) + hidden_size if cat_latent else sum(parents_cardinality)
                        self.propagators[str(i)][node_name] = Dense(input_size = prop_input_size, 
                                                                    output_size = node_cardinality,
                                                                    activation = activation)
                    elif prop_type == 'mlp':
                        prop_input_size = sum(parents_cardinality) + hidden_size if cat_latent else sum(parents_cardinality)
                        self.propagators[str(i)][node_name] = MLP(input_size = prop_input_size,
                                                                  hidden_size = sum(parents_cardinality)*2,
                                                                  output_size = node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
                    elif prop_type == 'embeddings':
                        self.propagators[str(i)][node_name] = MLP(input_size = len(parents)*concept_hidden_size,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
                        # TODO: embeddings for the latent factors if there are undirected edges
                    elif prop_type == 'equations':
                        self.propagators[str(i)][node_name] = MLP(input_size = node_cardinality*concept_hidden_size,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = sum(parents_cardinality)*node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
        else:
            raise ValueError('invalid prop_type')     
        

    def forward(self, x, c=None, intervention_index=None):
        # Encode input, get the latent features
        x_encoded = self.encoder(x)

        c_embs, c_probs, c_values_emb = {}, {}, {}
        for i, name in enumerate(self.combo_info['names']):
            # create embeddings and probabilities for each root concept
            # this assumes the task is last in the name list
            if name in self.roots_info['names']:
                c_embs[name], c_probs[name] =  self.concept_encoders[name](x_encoded, 
                                                                           c[:,i], 
                                                                           intervention_index[:,i],
                                                                           to_return=['embs', 'probs'])
            else:
                # create latent for each non-root concept    
                # remember not to intervene on the task
                c_values_emb[name] =  self.concept_encoders[name](x_encoded, 
                                                                  c[:,i] if name in self.c_names else None, 
                                                                  intervention_index[:,i] if name in self.c_names else None,
                                                                  to_return=['values_embs'])
                
        # propagate the information through the causal graph
        # loop over le graph levels, starting from the level 1 after the roots
        for _, level in self.propagators.items():
            # update all nodes in the level
            for c_name, propagator in level.items():
                c_index = self.combo_info['names'].index(c_name)
                p_indices = get_parents(self.graph, c_index).tolist()
                p_names = [self.combo_info['names'][p] for p in p_indices]
                
                if self.prop_type =='dense' or self.prop_type == 'mlp':
                    # propagate probabilities
                    c_prob_parents = torch.cat([c_probs[p_name] for p_name in p_names], dim=1)
                    if self.cat_latent: 
                        c_prob_parents = torch.cat([c_prob_parents, x_encoded], dim=1)
                    c_probs[c_name] = torch.softmax(propagator(c_prob_parents), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])

                elif self.prop_type == 'embeddings':
                    c_cardinality = self.combo_info['cardinality'][c_index]
                    # propagate embeddings
                    c_embs_parents = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_probs[c_name] = torch.softmax(propagator(c_embs_parents), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])
                    # compute the weighted average between concepts embedding and probabilities
                    first = c_values_emb[c_name].reshape(-1, c_cardinality, self.concept_hidden_size)
                    second = c_probs[c_name].unsqueeze(-1)
                    c_embs[c_name] = (first * second).sum(dim=1)

                elif self.prop_type == 'equations':
                    c_cardinality = self.combo_info['cardinality'][c_index]
                    if c_name in getattr(self, 'fixed_parent_slot_indices', {}):
                        # Build the same parent-state slots for every graph. An
                        # absent edge contributes a zero vector in that parent's
                        # permanent slot; present parents contribute their
                        # predicted categorical probabilities.
                        parent_vectors = []
                        for parent_index in self.fixed_parent_slot_indices[c_name]:
                            parent_name = self.combo_info['names'][parent_index]
                            parent_cardinality = self.combo_info['cardinality'][parent_index]
                            if self.graph[parent_index, c_index].item() == 1:
                                if parent_name not in c_probs:
                                    raise RuntimeError(
                                        f"Parent {parent_name} of {c_name} was not evaluated before its child"
                                    )
                                parent_vectors.append(c_probs[parent_name])
                            else:
                                parent_vectors.append(x_encoded.new_zeros((x_encoded.shape[0], parent_cardinality)))
                        c_prop_parents = torch.cat(parent_vectors, dim=1).unsqueeze(-1)
                        parent_state_width = c_prop_parents.shape[1]
                    else:
                        # Paper implementation: concatenate only the graph's actual parents.
                        p_cardinality = [self.combo_info['cardinality'][p] for p in p_indices]
                        c_prop_parents = torch.cat([c_probs[p_name] for p_name in p_names], dim=1).unsqueeze(-1)
                        parent_state_width = sum(p_cardinality)
                    weights = propagator(c_values_emb[c_name])
                    weights = weights.reshape(-1, c_cardinality, parent_state_width)
                    c_probs[c_name] = torch.softmax(torch.matmul(weights, c_prop_parents).squeeze(-1), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])                   
                    # first = c_values_emb[c_name].reshape(-1, c_cardinality, self.concept_hidden_size)
                    # second = c_probs[c_name].unsqueeze(-1)
                    # c_embs[c_name] = (first * second).sum(dim=1)

        # Decode, get task logits
        y_hat_probs = c_probs[self.y_names[0]]
        # filter virtual roots
        c_hat_probs = {k:v for k,v in c_probs.items() if k in self.c_names and k not in self.virtual_roots}
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        return y_output, c_output

    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task probabilities
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept probabilities
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()
        # c = c.long() # later to avoid nan to disappear
        loss_form = torch.nn.NLLLoss()

        # -- task loss
        y_hat = torch.log(y_hat + 1e-6)
        task_loss = loss_form(y_hat, y)

        # -- concepts loss
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            c_hat = torch.log(c_hat + 1e-6)
            concept_loss += loss_form(c_hat, c[:,self.c_names.index(name)].long())
        if self.normalize_concept_loss:
            concept_loss /= len(c_hat_dict)

        return self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss