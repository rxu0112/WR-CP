from torch import nn
from utils import *
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

def whiten(x, uni):
    with torch.no_grad():
        x -= uni.mean(dim=0)
        x /= uni.std(dim=0)
    return torch.nan_to_num(x)


def mean_l1(pred, y):
    return nn.functional.l1_loss(pred, y)


def mean_residual(pred, y):
    return nn.functional.l1_loss(pred, y)


def Wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    """
    A modified version of scipy.stats.wasserstein_distance

    """
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)
    all_values = torch.concatenate((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    deltas = torch.diff(all_values)
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], side='right')
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], side='right')

    zero = torch.tensor([0]).to('cuda:0')
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.shape[0]
    else:
        u_sorted_cumweights = torch.concatenate((zero, torch.cumsum(u_weights[u_sorter], dim=0)))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.shape[0]
    else:
        v_sorted_cumweights = torch.concatenate((zero, torch.cumsum(v_weights[v_sorter], dim=0)))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    pos = torch.searchsorted(u_cdf, torch.tensor([1.0]).to('cuda:0')).item()

    u_cdf_scaled = u_cdf[0:pos]
    v_cdf_scaled = v_cdf[0:pos]
    deltas_scaled = deltas[0:pos]
    W = torch.sum(torch.multiply(torch.abs(u_cdf_scaled - v_cdf_scaled), deltas_scaled))
    return W


def get_wr_penalty(train_envs, cal_envs_uni):
    # require (n,) format, so flatten()
    cwn_losses = torch.stack(
        [Wasserstein_distance(cal_envs_uni['res_set'].flatten(), e['res_set'].flatten(),
                                         u_weights=e['weights'], v_weights=None) for e
         in train_envs])
    penalty = torch.mean(cwn_losses)  # must include mean as a loss, only mean is better
    return penalty



def weight_calculation(Input_CS, Input, kde, bandwidth):
    kde_CS = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Input_CS)
    weight = []
    for i in range(Input.shape[0]):
        log_prob = kde.score_samples([Input[i]])
        log_prob_CS = kde_CS.score_samples([Input[i]])
        prob = np.exp(log_prob)
        prob_CS = np.exp(log_prob_CS)
        weight.append(prob_CS / prob)

    weight = np.array(weight).reshape(1, -1).flatten()
    return weight, kde_CS


def weighted_quantile(values, quantiles, weight, values_sorted=False):
    """ Very close to np.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]! """
    values = np.array(values)
    quantiles = np.array(quantiles)
    weight = np.array(weight)
    sorter = np.argsort(values)
    if not values_sorted:
        values = values[sorter]
        weight = weight[sorter]
    weighted_quantiles = np.cumsum(weight) - 0.5 * weight
    weighted_quantiles /= np.sum(weight)
    return np.interp(quantiles, weighted_quantiles, values)


def Testing_Coverage(true_test, pred_test, R_cal, alpha, weight):
    if np.any(weight):
        Size = 2 * weighted_quantile(R_cal, 1 - alpha, weight)
        up_test_CPCS = pred_test + weighted_quantile(R_cal, 1 - alpha, weight)
        low_test_CPCS = pred_test - weighted_quantile(R_cal, 1 - alpha, weight)
        Covered = np.sum((true_test[:, np.newaxis] >= low_test_CPCS[:, np.newaxis]) &
                         (true_test[:, np.newaxis] <= up_test_CPCS[:, np.newaxis]))
    else:
        Covered = 0
        Size = 0

    return Covered, Size


def pinball_loss(pred, y, alpha=0.5):
    # Compute the difference between prediction and true value
    diff = y - pred

    # Implement the pinball loss formula
    loss = torch.maximum(alpha * diff, (alpha - 1) * diff)

    # Return the mean loss
    return torch.mean(loss)


def TV_Distance(data1, data2, grid_points=100):

    # Create KDEs for each dataset
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    # Determine the range for integration
    all_data = np.concatenate([data1, data2])
    lower_bound, upper_bound = np.min(all_data), np.max(all_data)

    # Create a grid of points where the KDEs will be evaluated
    grid = np.linspace(lower_bound, upper_bound, grid_points)

    # Evaluate each KDE on the grid
    pdf1 = kde1(grid)
    pdf2 = kde2(grid)

    # Estimate the TV distance using numerical integration
    tv_distance = 0.5 * np.trapz(np.abs(pdf1 - pdf2), grid)

    return tv_distance


def KL_Divergence(data1, data2, grid_points=100):

    # Create KDEs for each dataset
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    all_data = np.concatenate([data1, data2])
    lower_bound, upper_bound = np.min(all_data), np.max(all_data)

    # Create a grid of points where the KDEs will be evaluated
    grid = np.linspace(lower_bound, upper_bound, grid_points)

    # Evaluate each KDE on the grid
    pdf1 = kde1(grid)
    pdf2 = kde2(grid)

    # Ensure positive values to avoid log of zero or negative values
    pdf1 = np.where(pdf1 <= 0, np.finfo(float).eps, pdf1)
    pdf2 = np.where(pdf2 <= 0, np.finfo(float).eps, pdf2)

    # Calculate the KL divergence using numerical integration
    kl_divergence = np.trapz(pdf1 * np.log(pdf1 / pdf2), grid)

    return kl_divergence