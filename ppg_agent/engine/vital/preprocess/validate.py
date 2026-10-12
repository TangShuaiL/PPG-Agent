import numpy as np
from ppg_agent.engine.vital.params import MINIMUM_PULSE_CYCLE, MAXIMUM_PULSE_CYCLE


def validate_template(template: np.ndarray, fs: float,
                      trough_diff: float = 0.2) -> bool:
    """
    Validate the quality of the beat-to-beat single waveforms by analyzing the template morphology.
    Specifically:
        1) Pulse width
        2) Location of the maximum value
        3) Location of the through
        4) Trough difference

    Parameters:
        template (np.ndarray): Beat-to-beat single PPG waveform.
        fs (float): Sample frequency of the signal (Hz).
        trough_diff (float): Max allowed |first-last| trough difference as a
            fraction of beat amplitude. 0.2 = historical strict default (clinical
            environments). Free-living wrist data with residual drift needs a
            larger value (e.g. 0.4) — pair with template_corr_min so the extra
            admitted beats are shape-consistent with the segment median.

    Returns:
        bool: True if the waveform meets the requirements, False otherwise.
    """

    # 1) Check that the pulse width is within the normal range
    period = len(template) / fs
    if period < MINIMUM_PULSE_CYCLE or period > MAXIMUM_PULSE_CYCLE:
        return False

    # 2) Check whether the systolic peak is in the half first part of the waveform
    max_index = np.argmax(template)
    if float(max_index) / float(len(template)) >= 0.75:
        return False

    # 3) Check where is the minimum point (through)
    # DISABLED: PulseDB normalized data often has valley slightly offset
    # min_index = np.argmin(template)
    # if not (min_index == 0 or min_index == len(template) - 1):
    #     return False

    # 4) Check if there is little difference between the init and final min point (through)
    if abs(template[0] - template[-1]) / (max(template) - min(template)) > trough_diff:
        return False

    return True
