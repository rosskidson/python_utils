import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# For fitting a gaussian curve.
from scipy.stats import norm


def plot(data,
         fig_w=15,
         fig_h=9):
    plt.figure(figsize=(fig_w, fig_h))
    plt.grid()
    _ = plt.plot(data, 'x', linestyle='-')


def plot_multiple(data,
         fig_w=15,
         fig_h=9,
         x_range=None,
         heading_line=None,
         heading_size=16,
         x_axis_label=None,
         y_axis_label=None,
         legend_list=[],
         axis_color='black'):
    plt.figure(figsize=(fig_w, fig_h))
    if x_range is not None:
        data = [list(n) for n in data]
        data = [n[x_range[0]:x_range[1]] for n in data]
    [plt.plot(n, 'x', linestyle='-') for n in data]
    plt.axes().legend(legend_list)
    plt.title(heading_line, size=heading_size, color=axis_color)
    plt.setp(plt.gca().get_xticklabels(), color=axis_color)
    plt.setp(plt.gca().get_yticklabels(), color=axis_color)
    plt.xlabel(x_axis_label, size=heading_size - 2, color=axis_color)
    plt.ylabel(y_axis_label, size=heading_size - 2, color=axis_color)
    plt.grid()


def plot_xy(x, y):
    plt.figure(figsize=(15, 9))
    plt.grid()
    _ = plt.plot(x, y, 'x')


def plot_xy_multiple(x_array, y_array, marker_array=None):
    if not marker_array:
        marker_array = ['-'] * len(x_array)
    plt.figure(figsize=(15, 9))
    plt.grid()
    for i in range(len(x_array)):
        _ = plt.plot(x_array[i], y_array[i], marker_array[i])


def remove_outliers(data, percent):
    data = sorted(data)
    start_idx = int(len(data) * (percent / 2.0))
    return data[start_idx:(len(data) - start_idx)]


def fit_gaussian(data, plt, num_bins):
    (mu, sigma) = norm.fit(data)
    #y = mlab.normpdf(num_bins, mu, sigma)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, num_bins)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, norm.pdf(x, mu, sigma))
    plt.axvline(x=mu, color='r')
    return ' mu={:.3f}'.format(mu) + '  sigma={:.3f}'.format(sigma)


def plot_hist(data,
              outlier_removal_percentage=0.00,
              num_bins=100,
              add_gaussian_fit=True,
              heading_line=None,
              heading_size=14,
              fig_w=15,
              fig_h=9,
              axis_color='black'):
    data = remove_outliers(data, outlier_removal_percentage)
    plt.figure(figsize=(fig_w, fig_h))
    plt.grid()
    ret = plt.hist(data, normed=True, bins=num_bins)
    if add_gaussian_fit:
        fit_gaussian(data, plt, num_bins)
        gaussian_text = fit_gaussian(data, plt,
                                     num_bins) if add_gaussian_fit else ''
        if heading_line is not None:
            _ = plt.title(heading_line + '\n' + gaussian_text, size=heading_size, color=axis_color)
        else:
            _ = plt.title(gaussian_text, size=heading_size, color=axis_color)
    plt.setp(plt.gca().get_xticklabels(), color=axis_color)
    plt.setp(plt.gca().get_yticklabels(), color=axis_color)


def scale_data(convert_rpy_to_degrees, data, key):
    if convert_rpy_to_degrees and any(rpy in key
                                      for rpy in ['roll', 'pitch', 'yaw']):
        return data * 180 / np.pi
    else:
        return data


def plot_6dof(data,
              keys,
              outlier_removal_percentage=0.00,
              num_bins=100,
              x_scales=None,
              convert_rpy_to_degrees=True,
              add_gaussian_fit=True,
              apply_xrange_to_gaussian=False,
              do_plot=True):
    # Input parameter validation.
    apply_xrange_to_gaussian = False if x_scales is None else apply_xrange_to_gaussian

    if do_plot:
        plt.figure(figsize=(15, 9))
        plt.tight_layout()

    fit_values = dict()
    for idx, key in enumerate(keys):
        if x_scales is not None:
            x_min = x_scales[idx][0]
            x_max = x_scales[idx][1]

        # Convert to degrees and remove outliers
        processed_data = scale_data(convert_rpy_to_degrees, data[key], key)
        processed_data = remove_outliers(processed_data,
                                         outlier_removal_percentage)
        if do_plot:
            plt.subplot(2, 3, idx + 1)
            plt.grid()

            if x_scales is not None:
                # Apply x range to both plot area and to histogram calculation.
                plt.xlim(xmin=x_min, xmax=x_max)
                _ = plt.hist(processed_data,
                             range=[x_min, x_max],
                             normed=True,
                             bins=num_bins)
            else:
                _ = plt.hist(processed_data, normed=True, bins=num_bins)

        # Also apply the x range to gaussian fit, if requested.
        gaussian_input_data = [
            x for x in processed_data if x >= x_min and x <= x_max
        ] if apply_xrange_to_gaussian else processed_data

        # Fit the data
        (mu, sigma) = norm.fit(gaussian_input_data)

        # Plot the Gaussian fit.
        if do_plot:
            plot_gaussian(plt, num_bins, mu, sigma)
            gaussian_text = (
                '\n' +
                format_fit_parameters(mu, sigma)) if add_gaussian_fit else ''

            # Add title.
            _ = plt.title(keys[idx] + gaussian_text)

        fit_values[key] = (mu, sigma)

    return fit_values
