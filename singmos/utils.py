import torch
import numpy as np

from scipy.interpolate import interp1d

import logging

def _convert_to_continuous_f0(f0: np.array) -> np.array:
    if (f0 == 0).all():
        logging.warning("All frames seems to be unvoiced.")
        return f0

    # padding start and end of f0 sequence
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nonzero_idxs = np.where(f0 != 0)[0]

    # perform linear interpolation
    interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
    f0 = interp_fn(np.arange(0, f0.shape[0]))

    return f0


def f0_dio(
    audio,
    sampling_rate,
    hop_size=320,
    pitch_min=40,
    pitch_max=800,
    use_log_f0=False,
    use_continuous_f0=False,
    use_discrete_f0=True,
):
    """Compute F0 with pyworld.dio

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        hop_size (int): Hop size.
        pitch_min (int): Minimum pitch in pitch extraction.
        pitch_max (int): Maximum pitch in pitch extraction.

    Returns:
        ndarray: f0 feature (#frames, ).

    Note:
        Unvoiced frame has value = 0.

    """
    if torch.is_tensor(audio):
        x = audio.cpu().numpy().astype(np.double)
    else:
        x = audio.astype(np.double)
    frame_period = 1000 * hop_size / sampling_rate
    import pyworld
    f0, timeaxis = pyworld.dio(
        x,
        sampling_rate,
        f0_floor=pitch_min,
        f0_ceil=pitch_max,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(x, f0, timeaxis, sampling_rate)
    if use_continuous_f0:
        f0 = _convert_to_continuous_f0(f0)
    if use_log_f0:
        nonzero_idxs = np.where(f0 != 0)[0]
        f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
    if use_discrete_f0:
        f0 = np.round(f0)
        f0 = f0.astype(int)
    f0[f0 > pitch_max] = pitch_max
    return f0


def calc_f0_variation(
    audio,
    sampling_rate,
    hop_size=320,
    pitch_min=40,
    pitch_max=800,
    use_log_f0=False,
    use_continuous_f0=False,
    use_discrete_f0=True,
    bias=800,
    return_tensor=True,
):
    """Compute f0 variation

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        hop_size (int): Hop size.
        pitch_min (int): Minimum pitch in pitch extraction.
        pitch_max (int): Maximum pitch in pitch extraction.
        bias: Bias to ensure positive value in pitch variation

    Returns:
        np.int: start of f0 feature
        ndarray: f0 variation feature (#frames, ).
    """
    f0 = f0_dio(
        audio,
        sampling_rate,
        hop_size=hop_size,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        use_log_f0=use_log_f0,
        use_continuous_f0=use_continuous_f0,
        use_discrete_f0=use_discrete_f0,
    )
    f0_start = f0[0]
    f0_variation = f0[1:] - f0[:-1]
    f0_variation = f0_variation + bias
    
    if return_tensor is True:
        f0_start = torch.tensor(f0_start)
        f0_variation = torch.tensor(f0_variation)
        f0 = torch.tensor(f0)
    # logging.info(f'f0: {f0.max()}, f0_var: {f0_variation.max()}')
    return f0_start, f0_variation, f0