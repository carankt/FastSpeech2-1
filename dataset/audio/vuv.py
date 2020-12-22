"""F0 extractor using DIO + Stonemask algorithm."""

import logging

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import numpy as np
import torch
import pycwt as wavelet
import webrtcvad
from scipy.interpolate import interp1d
from typeguard import check_argument_types


class VuV():

    def __init__(
            self,
            J: int = 10,
    ):
        assert check_argument_types()
        super().__init__()
        self.J = J

    def forward(
            self,
            input: torch.Tensor,
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        # If not provide, we assume that the inputs have the same length
        # F0 extraction

        # input shape = [T,]
        energy = input.detach().numpy()
        energy_log = np.log(input.detach().numpy())
        # (Optional): Adjust length to match with the mel-spectrogram
        #pitch, pitch_log = self._convert_to_continuous_f0(pitch)
        energy_log_norm, mean, std = self._normalize(energy_log)
        coefs, scales = self._cwt(energy_log_norm)
        # (Optional): Average by duration to calculate token-wise f0
        # Return with the shape (B, T, 1)
        return energy, mean, std, coefs

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor :

        norm_pitch = (x - x.mean())/x.std()
        return norm_pitch, x.mean(), x.std()

    def _cwt(self, x: torch.Tensor) -> np.array:
        mother = wavelet.MexicanHat()
        dt = 0.005
        dj = 2
        s0 = dt*2
        J = self.J - 1
        Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(x, dt, dj, s0, J, mother)
        Wavelet_lf0 = np.real(Wavelet_lf0).T

        return Wavelet_lf0, scales
