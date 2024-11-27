import matplotlib.pyplot as plt
import numpy as np

from guidance_utils import plot_endpoints


def setup_plot(window_size, data, lim=True):
    """Limit the plot to a square window of size [-window_size, window_size]"""
    plt.scatter(data[:, 0], [data[:, 1]], label='data', marker='+', s=20, color='orangered')  # firebrick
    plt.xlim(-window_size, window_size)
    plt.ylim([-window_size, window_size])
    plt.grid(alpha=0.2)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)


def plot_traject(trajec_list, t_steps, alpha=None, scale=1, width=0.005):
    n = len(t_steps) - 2

    # Linear steps from a threshold onwards, from 1 to 0
    color_steps = np.clip(np.arange(n, 0, -1), 0, n // 2)
    c_min, c_max = color_steps.min(), color_steps.max()
    color_steps = (color_steps - c_min) / (c_max - c_min)

    # Linear steps from a threshold onwards, from 0 to 1
    alpha_steps = np.clip(np.arange(n), n // 4, n)
    c_min, c_max = alpha_steps.min(), alpha_steps.max()
    alpha_steps = (alpha_steps - c_min) / (c_max - c_min)
    for c, trajec in enumerate(trajec_list):
        plt.quiver(trajec[:-1, 0], trajec[:-1, 1],
                   trajec[1:, 0] - trajec[:-1, 0],
                   trajec[1:, 1] - trajec[:-1, 1],
                   color_steps,
                   cmap='viridis',
                   scale_units='xy',
                   angles='xy',
                   scale=scale,
                   width=width,
                   alpha=alpha_steps if alpha is None else alpha,
                   headlength=0,
                   headaxislength=0)


def gen_plot1(model, x_labels, n, m, guid_weight, interval, ylabel, titles, window_size):
    """
    Args:
        n: Number of trajectories
        m: Number of trajectories to plot
        trajects: Trajectories x(t), [n, n_steps, 2]
        preds: Predictions y(t), [n, n_steps, 2]
        t_steps: Noise levels
    """

    trajects, preds, weight_hists = model.run(x_labels=x_labels,
                                              n=n,
                                              guid_weight=guid_weight,
                                              opt=False,
                                              interval=interval)

    plt.figure(figsize=(12,4))
    plt.subplots_adjust(wspace=0)
    # Trajectories
    plt.subplot(1, 3, 1)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    if titles:
        plt.title('$x(t)$')

    plot_traject(trajects[::n // m], model.t_steps)
    setup_plot(window_size, model.data)

    # Predictions
    plt.subplot(1, 3, 2)
    plot_traject(preds[::n // m], model.t_steps)
    setup_plot(window_size, model.data)
    if titles:
        plt.title('$y(t)$')

    # Endpoints
    plt.subplot(1, 3, 3)
    plot_endpoints(model.data, trajects)
    setup_plot(window_size, model.data)
    if titles:
        plt.title('$x(0)$')

    total_error = (((trajects[None, :, -1] - model.data[:, None]) ** 2).min(axis=0).sum(axis=-1) ** 0.5).mean()
    plt.xlabel(f"{total_error:.1e}")

    return total_error
