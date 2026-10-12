import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from ppg_agent.engine.vital.params import PPG_SAMPLE_RATE


def n_arg_fun(function) -> int:
    # Compute number of compulsory (not default) arguments
    all_arg = function.__code__.co_argcount
    default_arg = function.__defaults__

    if default_arg is None:
        return int(all_arg)
    else:
        return int(all_arg - len(default_arg))


def timeline(waveform, fs: float = PPG_SAMPLE_RATE) -> np.ndarray:
    """
    Description:
        Create numpy vector containing the points of time of the waveform signal, starting in zero seconds.

    Parameters:
        waveform
        fs: sample frequency of the signal (Hz)

    Return:
        timeline (np.ndarray): Vector with the time points of the waveform starting in zero
    """

    return np.linspace(0, len(waveform) / fs, len(waveform))


def derivative(waveform: np.ndarray, fs: float = PPG_SAMPLE_RATE) -> np.ndarray:
    """
    Description:
        Calculate the derivative of the waveform

    Parameters:
        waveform (np.ndarray): numpy array with the signal data points
        fs: sample frequency of the signal (Hz)

    Return:
        derivative (np.ndarray): Vector containing the derivative signal points
    """

    return np.diff(waveform, n=1) * float(fs)


def get_intersection_point(m1: float, b1: float, m2: float, b2: float):
    """
    Description:
        Calculate the intersection point of two lines defined by its slope (m) and intercept point (b), such as:
        y= m*x+b

    Parameters:
        m1 (float): slope of line 1
        b1 (float): interecept point of line 1
        m2 (float): slope of line 2
        b2 (float): interecept point of line 2

    Return:
        xi (float), yi (float): intersection point
    """

    xi = (b1 - b2) / (m2 - m1)
    yi = m1 * xi + b1
    return xi, yi


def zero_cross(waveform: np.ndarray) -> int:
    """
    Description:
        Calculate the number of times that the waveform crosses zero

    Parameters:
        waveform (np.ndarray): numpy array with the PPG signal data points
    Return:
        Number of zero crossing (int): Vector containing the derivative signal points
    """
    return len(np.where(np.diff(np.sign(waveform)))[0])


def plot_key_points(ax: Axes, template: np.ndarray, timeline: np.ndarray, points: dict, fs: float = PPG_SAMPLE_RATE,
                    legend: bool = True) -> None:
    """
    Description:
        Plot the PPG waveform and its key-points at the given axes

    Parameters:
        ax (matplotlib.axes.Axes): specific axes for plotting
        template (pd.Series): beat-to-beat PPG signal data points
        timeline (np.ndarray): Vector with the time points of the waveform starting in zero
        points (dict): dictionary with the indexes of the PPG signals key-points
        fs (float): sample frequency of the signal (Hz)
        legend (bool): whether to show the legend of the plot

    Return:
        None
    """

    ax.plot(timeline, template)
    ax.plot(points['o'] / fs, template[points['o']], 'bo', label='O')
    ax.plot(points['a'] / fs, template[points['a']], 'yo', label="a")
    ax.plot(points['md'] / fs, template[points['md']], 'go', label='MD')
    ax.plot(points['b'] / fs, template[points['b']], 'co', label="b")
    ax.plot(points['s'] / fs, template[points['s']], 'ro', label='S')
    ax.plot(points['dn'] / fs, template[points['dn']], 'mo', label="DN")
    if points['ip'] == points['d']:
        ax.plot(points['ip'] / fs, template[points['ip']], 'co', label="IP")
    else:
        ax.plot(points['ip'] / fs, template[points['ip']], 'co', label="IP")
        ax.plot(points['d'] / fs, template[points['d']], marker='o', color='orange', linestyle="None", label="D")
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_key_points_dev(template: pd.Series, points: dict, fs: float = PPG_SAMPLE_RATE) -> None:
    """
     Description:
         Plot the PPG waveform and its derivatives (first and second) and the corresponding key-points

     Parameters:
         template (pd.Series): beat-to-beat PPG signal data points
         points (dict): dictionary with the indexes of the PPG signals key-points
         fs (float): sample frequency of the signal (Hz)

     Return:
         None
     """

    template_np = template.to_numpy()
    t = timeline(template_np, fs)
    derivative1 = derivative(template_np, fs)
    derivative2 = derivative(derivative1, fs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    plt.suptitle('PPG key-points')
    plot_key_points(ax1, template_np, t, points, fs, legend=False)
    ax1.set_title('PPG template', fontsize='small')

    plot_key_points(ax2, derivative1, t[0:-1], points, fs, legend=False)
    ax2.set_title('First derivative template', fontsize='small')

    plot_key_points(ax3, derivative2, t[0:-2], points, fs)
    ax3.set_title('Second derivative template', fontsize='small')

    fig.tight_layout()
    plt.show()
