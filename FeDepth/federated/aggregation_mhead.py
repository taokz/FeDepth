import copy
import numpy as np
import torch
from torch import nn

class ModelAccumulator(object):
    """Accumulate models. Client models are sequentially trained and accumulatively added to the
    server (w/ weights). At the end of communication, the server model will be divided
    by summed weights.
    If local_bn is True, a dict of bn layers will be kept for all users.
    Concepts:
        running_model: The model used to train. This is not persistent storage. Load by call
            `load_model` at practice.
        server_state_dict: The current state_dict in server.
        accum_state_dict: The accumulated state_dict which will accumulate the trained results
            from running model and update to server_state_dict when fulfilled.
    Args:
        running_model: Model to init state_dict shape and bn layers.
        n_accum: Number of models to accumulate per round. If retrieve before this value,
            an error will raise.
        num_model: Total number of models. Used if local_bn is True.
        local_bn: Whether to keep local bn for all users.
        raise_err_on_early_accum: Raise error if update model when not all users are accumulated.
    """
    def __init__(self, running_model: nn.Module, n_accum, num_model, local_bn=False,
                 raise_err_on_early_accum=True, resource_weight=False, balance_weight=True):
        self.n_accum = n_accum
        self._cnt = 0
        self.local_bn = local_bn
        self._weight_sum = 0
        self.raise_err_on_early_accum = raise_err_on_early_accum
        self.num_model = num_model
        self.resource_weight = resource_weight
        self.balance_weight = balance_weight
        
        with torch.no_grad():
            self.server_state_dict = {
                k: copy.deepcopy(v) for k, v in running_model.state_dict().items()
            }
            self._accum_state_dict = {
                k: torch.zeros_like(v) for k, v in running_model.state_dict().items()
            }
            if local_bn:
                self.local_state_dict = [{
                    k: copy.deepcopy(v) for k, v in running_model.state_dict().items() if 'bn' in k
                } for _ in range(num_model)]
            else:
                self.local_state_dict = []

    def state_dict(self):
        return {
            'server': self.server_state_dict,
            'clients': self.local_state_dict,
        }

    def load_state_dict(self, state_dict: dict):
        self.server_state_dict = state_dict['server']
        local_state_dict = state_dict['clients']
        if self.local_bn:
            assert len(local_state_dict) > 0, "Not found local state dict when local_bn is set."
            # num_model
            assert len(local_state_dict) == len(self.local_state_dict), \
                f"Load {len(local_state_dict)} local states while expected" \
                f" {len(self.local_state_dict)}"
        else:
            assert len(local_state_dict) == 0, "Found local bn state when local_bn is not set."
        self.local_state_dict = local_state_dict

    def add(self, model_idx, model, weight, client_num=100):
        """Use weight = 1/n_accum to average.
        """
        if self._cnt >= self.n_accum:  # note cnt starts from 0
            raise RuntimeError(f"Try to accumulate {self._cnt}, while only {self.n_accum} models"
                               f" are allowed. Did you forget to reset after accumulated?")

        with torch.no_grad():
            for key in self._accum_state_dict:
                # if (key == 'linear1.weight'):
                #     print(model_idx, key, model.state_dict()[key])
                    # print(model_idx, key, self.local_state_dict[model_idx][key])
                if len(self.local_state_dict) >0 and key in self.local_state_dict[model_idx]:
                    self.local_state_dict[model_idx][key].data.copy_(model.state_dict()[key])
                else:
                    if 'num_batches_tracked' in key:
                        self._accum_state_dict[key].data.copy_(model.state_dict()[key])
                    else:
                        # if ("linear1" not in key) and ("linear2" not in key) and ("linear3" not in key) and ("linear4" not in key) and ("linear5" not in key) and ("linear6" not in key) and ("linear7" not in key) and ("linear10" not in key):
                        #     temp = weight * model.state_dict()[key]
                        #     self._accum_state_dict[key].data.add_(temp)
                        if self.balance_weight:
                            if model_idx < client_num/4*1:
                                if ("linear1" in key) or ("linear5" in key) or ("aux_norm1" in key) or ("aux_norm5" in key) or ("aux_head1" in key) or ("aux_head5" in key):
                                    temp = 4 * weight * model.state_dict()[key]
                                elif ("linear2" in key) or ("linear3" in key) or ("linear4" in key) or ("aux_norm2" in key) or ("aux_norm3" in key) or ("aux_norm4" in key) or ("aux_head2" in key) or ("aux_head3" in key) or ("aux_head4" in key):
                                    temp = 2 * weight * model.state_dict()[key]
                                else:
                                    temp = weight * model.state_dict()[key]
                            elif model_idx < client_num/4*2:
                                if ("linear1" in key) or ("linear3" in key) or ("linear5" in key) or ("aux_norm1" in key) or ("aux_norm3" in key) or ("aux_norm5" in key) or ("aux_head1" in key) or ("aux_head3" in key) or ("aux_head5" in key):
                                    temp = 0 * weight * model.state_dict()[key]
                                elif ("linear2" in key) or ("linear4" in key) or ("aux_norm2" in key) or ("aux_norm4" in key) or ("aux_head2" in key) or ("aux_head4" in key):
                                    temp = 2 * weight * model.state_dict()[key]
                                else:
                                    temp = weight * model.state_dict()[key]
                            elif model_idx < client_num/4*3:
                                if ("linear1" in key) or ("linear2" in key) or ("linear4" in key) or ("linear5" in key) or ("aux_norm1" in key) or ("aux_norm2" in key) or ("aux_norm4" in key) or ("aux_norm5" in key) or ("aux_head1" in key) or ("aux_head2" in key) or ("aux_head4" in key) or ("aux_head5" in key):
                                    temp = 0 * weight * model.state_dict()[key]   
                                elif ("linear3" in key) or ("aux_norm3" in key) or ("aux_head3" in key):
                                    temp = 2 * weight * model.state_dict()[key]
                                else:
                                    temp = weight * model.state_dict()[key]
                            elif model_idx < client_num/4*4:
                                if ("linear1" in key) or ("linear2" in key) or ("linear3" in key) or ("linear4" in key) or ("linear5" in key):
                                    temp = 0 * weight * model.state_dict()[key]   
                                elif ("aux_norm1" in key) or ("aux_norm2" in key) or ("aux_norm3" in key) or ("aux_norm4" in key) or ("aux_norm5" in key) or ("aux_head1" in key) or ("aux_head2" in key) or ("aux_head3" in key) or ("aux_head4" in key) or ("aux_head5" in key):
                                    temp = 0 * weight * model.state_dict()[key]   
                                else:
                                    temp = weight * model.state_dict()[key]
                        else:
                            temp = weight * model.state_dict()[key]
                        self._accum_state_dict[key].data.add_(temp)
                        
        self._cnt += 1  # DO THIS at the END such that start from 0.
        self._weight_sum += weight

    @property
    def accumulated_count(self):
        return self._cnt

    @property
    def accum_state_dict(self):
        self.check_full_accum()
        return self._accum_state_dict

    def load_model(self, running_model: nn.Module, model_idx: int, strict=True):
        """Load server model and local BN states into the given running_model."""
        state_dict = {k: v for k, v in self.server_state_dict.items()}
        if len(self.local_state_dict) > 0:
            for k in self.local_state_dict[model_idx]:
                state_dict[k] = self.local_state_dict[model_idx][k]
        running_model.load_state_dict(state_dict, strict=strict)

    def update_server_and_reset(self):
        """Load accumulated state_dict to server_model and
        reset accumulated values but not local bn."""
        self.check_full_accum()
        weight_norm = 1. / self._weight_sum
        with torch.no_grad():
            # update server
            for k in self.server_state_dict:
                if 'num_batches_tracked' in k:
                    self.server_state_dict[k].data.copy_(self._accum_state_dict[k].data)
                else:
                    self.server_state_dict[k].data.copy_(
                        self._accum_state_dict[k].data * weight_norm)

            # reset
            self._cnt = 0
            self._weight_sum = 0
            for k in self._accum_state_dict:
                self._accum_state_dict[k].data.zero_()

    def check_full_accum(self):
        """Check if the number of accumulated models reaches the expected value (n_accum)."""
        if self.raise_err_on_early_accum:
            assert self._cnt == self.n_accum, f"Retrieve before all models are accumulated. " \
                                              f"Expect to accumulate {self.n_accum} but only" \
                                              f" get {self._cnt}"
