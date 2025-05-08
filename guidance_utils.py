import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm


def plot_traject(trajec_list, t_steps, alpha=None, scale=1, width=0.005):
    n = len(t_steps) - 2

    # Linear steps from a threshold onwards, from 1 to 0
    color_steps = np.clip(np.arange(n, 0, -1), 0, n // 2)
    c_min, c_max = color_steps.min(), color_steps.max()
    color_steps = (color_steps - c_min) / (c_max - c_min)

    # Linear steps from a threshold onwards, from 0 to 1
    alpha_steps = np.clip(np.arange(n), n // 2, n)
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


def plot_endpoints(data, trajects):
    for c, trajec in enumerate(trajects):
        dist = np.power(trajec[-1] - data, 2).sum(axis=1)
        plt.scatter(trajec[-1, 0], trajec[-1, 1], s=5, c=dist.min() * 10)


def setup_plot(window_size, data):
    """Limit the plot to a square window of size [-window_size, window_size]"""
    plt.scatter(data[:, 0], [data[:, 1]], label='data', marker='+', s=60, color='orangered')  # firebrick
    plt.xlim(-window_size, window_size)
    plt.ylim([-window_size, window_size])
    plt.grid(alpha=0.2)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)


def plot_row(n, m, i, trajects, preds, weight_hists, opt_trajects, opt_preds, opt_weight_hists,
             ylabel, error_ylim, titles, window_size, data, t_steps):
    """
    Args:
        n: Number of trajectories
        m: Number of trajectories to plot
        i: Row index
        trajects: Trajectories x(t), [n, n_steps, 2]
        preds: Predictions y(t), [n, n_steps, 2]
        weight_hists: Guidance weights, only for compatibility
        opt_trajects: Optimal trajectories x(t), [n, n_steps, 2)]
        opt_preds: Optimal predictions y(t), [n, n_steps, 2]
        opt_weight_hists: Optimal guidance weights, only for compatibility
        t_steps: Noise levels
    """
    rows = 4
    columns = 4
    num_steps = len(t_steps) - 1

    # Trajectories
    plt.subplot(rows, columns, columns * i + 1)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    if titles:
        plt.title('$x(t)$')

    plot_traject(trajects[::n // m], t_steps)
    setup_plot(window_size, data)

    traject_l2_error = (((trajects - opt_trajects) ** 2).sum(axis=-1) ** 0.5).mean(axis=0)  # [n, n_steps]
    plt.xlabel(f"{traject_l2_error.mean():.1e}")

    # Predictions
    plt.subplot(rows, columns, columns * i + 2)
    plot_traject(preds[::n // m], t_steps)
    setup_plot(window_size, data)
    if titles:
        plt.title('$y(t)$')
    pred_l2_error = (((preds - opt_preds) ** 2).sum(axis=-1) ** 0.5).mean()  # [n, n_steps]
    plt.xlabel(f"{pred_l2_error:.1e}")

    # Endpoints
    plt.subplot(rows, columns, columns * i + 3)
    plot_endpoints(data, trajects)
    setup_plot(window_size, data)
    if titles:
        plt.title('$x(0)$')

    total_error = (((trajects[None, :, -1] - data[:, None]) ** 2).min(axis=0).sum(axis=-1) ** 0.5).mean()
    plt.xlabel(f"{total_error:.1e}")

    # Trajectory error accumulation
    plt.subplot(rows, columns, columns * i + 4)
    plt.plot(np.arange(num_steps), traject_l2_error)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.grid(alpha=0.2)
    if error_ylim is not None:
        plt.ylim(error_ylim)
    if titles:
        plt.title('L2 Error')

    return total_error


def full_grid(model, x_labels, n, m, guid_weight, interval, init, error_ylim=None, save_as=None):
    """
    Args:
        model: ToyModel instance to use for sampling
        x_labels: Array of labels for each trajectory, shape (n,)
        n: Number of trajectories
        m: Number of trajectories to plot
        guid_weight: Guidance weight
        interval: Interval where the guidance weight is used
        save_as: Path to save the plot in
    """

    total_error = []
    returns_hist = []

    plt.figure(figsize=(12, 12), facecolor='white')
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    window_size = 2.5
    model_kwargs = {'interval': interval, 'x_labels': x_labels, 'n': n, 'init': init}
    counter = 0  # Row counter

    # Optimal model trajectories as reference
    opt_returns = model.run(guid_weight=guid_weight,
                            opt=True,
                            **model_kwargs)

    # Run: Optimal, WMG, Positive, Negative
    guid_weights = [guid_weight, guid_weight, 0, -1]
    opts = [True, False, False, False]
    labels = [f'$\\epsilon^*(x, t)$',
              f'$\\tilde \\epsilon(x, t)$',
              f'$\\epsilon_{{pos}}(x, t)$',
              f'$\\epsilon_{{neg}}(x, t)$']

    for (guid_weight, opt, label) in zip(guid_weights, opts, labels):
        returns = model.run(guid_weight=guid_weight, opt=opt, **model_kwargs) if not opt else opt_returns
        total_error.append(plot_row(n,              # Number of trajectories
                                    m,              # Number of trajectories to plot
                                    counter,        # Row index
                                    *returns,       # Trajectories, predictions, weights
                                    *opt_returns,   # Optimal trajectories, predictions, weights
                                    label,          # ylabels
                                    error_ylim,     # ylim
                                    opt,            # Titles
                                    window_size,    # Window size
                                    model.data,     # Data
                                    model.t_steps)) # Noise levels
        counter += 1
        returns_hist.append(returns)

    if save_as is not None:
        plt.savefig(save_as, dpi=150, bbox_inches='tight')

    plt.show()

    return returns_hist


def geom_analysis(model, x_labels, guid_weight, interval, init, steps, legend=False):
    # Compute trajectory as a baseline
    model_kwargs = {'guid_weight': guid_weight, 'interval': interval, 'x_labels': x_labels, 'n': 1, 'init': init}
    x_opt, y_opt, weights_opt = model.run(opt=False, **model_kwargs)

    for step in steps:
        # Select one trajectory and plot it
        plot_traject([x_opt[0]], model.t_steps, scale=0.9, width=0.008, alpha=1)
        x_start = x_opt[0][step - 1]
        t_cur, t_next = model.t_steps[step], model.t_steps[step + 1]

        # Predictions
        if x_labels is None:
            x_labels = [0]
        y_opt = model.y_toy(x_start[None], t_cur, x_labels, **{'delta': 0, 'cond': model.pos_kwargs['cond']}).squeeze()
        y_pos = model.y_toy(x_start[None], t_cur, x_labels, **model.pos_kwargs).squeeze()
        y_neg = model.y_toy(x_start[None], t_cur, x_labels, **model.neg_kwargs).squeeze()

        # Optimal weight
        opt_weight = norm(y_opt - y_pos) / norm(y_pos - y_neg)
        guid_weight = guid_weight if guid_weight != 'opt_weight' else opt_weight
        y_pred = y_pos + guid_weight * (y_pos - y_neg)

        # Colors
        pos_color = 'dodgerblue'
        neg_color = 'purple'
        guid_color = 'black'

        # Positive step
        plt.quiver(x_start[0],
                   x_start[1],
                   y_pos[0] - x_start[0],
                   y_pos[1] - x_start[1],
                   color=pos_color,
                   scale_units='xy',
                   angles='xy',
                   scale=1,
                   label='Pos')

        # Negative step
        plt.quiver(x_start[0],
                   x_start[1],
                   y_neg[0] - x_start[0],
                   y_neg[1] - x_start[1],
                   color=neg_color,
                   scale_units='xy',
                   angles='xy',
                   scale=1,
                   label='Neg')

        # Guidance step
        plt.plot([y_neg[0], y_pos[0]], [y_neg[1], y_pos[1]], color='black', linestyle=':')
        plt.quiver(y_pos[0],
                   y_pos[1],
                   y_pred[0] - y_pos[0],
                   y_pred[1] - y_pos[1],
                   color=guid_color,
                   scale_units='xy',
                   angles='xy',
                   scale=1,
                   label='Guidance')

        if step == steps[0] and legend:
            plt.legend(prop={'size': 11})

    # Setup
    plt.scatter(model.data[:, 0], [model.data[:, 1]], label='data', marker='+', s=100 * 4, color='orangered', alpha=1)
    print(f'Opt weight = {opt_weight:.2f}')


def optimal_weights(model, x_labels, guid_weight, interval, init, n=100):
    """Plot the optimal guidance weight w^*(x,t)"""
    model_kwargs = {'guid_weight': guid_weight, 'interval': interval, 'x_labels': x_labels, 'n': n, 'init': init}
    weight_hists = model.run(opt=False, **model_kwargs)[2]

    for hist in weight_hists.mean(axis=0, keepdims=True):  # Average over trajectories
        plt.plot(hist, label=f"$\\delta_{{neg}} = {model.neg_kwargs['delta']}$")
