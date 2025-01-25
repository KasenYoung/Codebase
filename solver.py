from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))




class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev



class EulerSolver:

    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50):
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (np.arange(1, euler_timesteps + 1) *
                                self.step_ratio).round().astype(np.int64) - 1
        self.euler_timesteps_prev = np.asarray(
            [0] + self.euler_timesteps[:-1].tolist())
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas_prev = np.asarray(
            [sigmas[0]] + sigmas[self.euler_timesteps[:-1]].tolist()
        )  # either use sigma0 or 0

        self.euler_timesteps = torch.from_numpy(self.euler_timesteps).long()
        self.euler_timesteps_prev = torch.from_numpy(
            self.euler_timesteps_prev).long()
        self.sigmas = torch.from_numpy(self.sigmas)
        self.sigmas_prev = torch.from_numpy(self.sigmas_prev)

    def to(self, device):
        self.euler_timesteps = self.euler_timesteps.to(device)
        self.euler_timesteps_prev = self.euler_timesteps_prev.to(device)

        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        return self

    def euler_step(self, sample, model_pred, timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index,
                                    model_pred.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index,
                                         model_pred.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred
        return x_prev

    def euler_style_multiphase_pred(
        self,
        sample,
        model_pred,
        timestep_index,
        multiphase,
        is_target=False,
    ):
        inference_indices = np.linspace(0,
                                        len(self.euler_timesteps),
                                        num=multiphase,
                                        endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (torch.from_numpy(inference_indices).long().to(
            self.euler_timesteps.device))
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0))
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(
            dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index_end = inference_indices[last_valid_index]

        if is_target:
            sigma = extract_into_tensor(self.sigmas_prev, timestep_index,
                                        sample.shape)
        else:
            sigma = extract_into_tensor(self.sigmas, timestep_index,
                                        sample.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index_end,
                                         sample.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred

        return x_prev, timestep_index_end
