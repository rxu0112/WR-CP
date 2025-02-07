import argparse
from torch import optim
import pickle
from module import *
import os

parser = argparse.ArgumentParser(description='CDPI')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--steps', type=int, default=3000)
parser.add_argument('--dataset', type=str, default='airfoil')  # argument for specifying the dataset
parser.add_argument('--version', type=str, default='v1')  # argument for specifying the trial
flags = parser.parse_args()


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        lin1 = nn.Linear(input_size, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            lin1, nn.Tanh(),  # nn.ReLU(True),
            nn.Dropout(),
            lin2, nn.Tanh(),  # nn.ReLU(True),
            nn.Dropout(),
            lin3)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_size)
        out = self._main(x)
        return out


if __name__ == "__main__":

    print('Flags:')
    for k, v in sorted(vars(flags).items()):
        print("    {}: {}".format(k, v))

    # drop data
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_envs = pickle.load(open(project_path + "/data/" + flags.dataset + "/processed/train_" + flags.version, "rb"))
    test_envs = pickle.load(open(project_path + "/data/" + flags.dataset + "/processed/test_" + flags.version, "rb"))
    cal_envs_uni = np.load(project_path + "/data/" + flags.dataset + "/processed/cal_uni_" + flags.version + '.npy',
                           allow_pickle='TRUE').item()

    # init
    RESULT = []
    print('erm')

    logger = Logger()
    alpha_array = np.linspace(0.1, 0.9, 9)
    coverage_hour_target = 1 - alpha_array
    coverage_summary_target = [coverage_hour_target[:] for _ in range(len(test_envs))]

    mlp = MLP(train_envs[0]['images'].shape[1]).cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train l1')

    for step in range(flags.steps):
        cal_envs_uni['pred'] = mlp(cal_envs_uni['images'])
        cal_envs_uni['res_set'] = torch.abs(torch.subtract(cal_envs_uni['pred'], cal_envs_uni['labels']))

        for env in train_envs:
            env['pred'] = mlp(env['images'])
            env['l1'] = mean_l1(env['pred'], env['labels'])
            env['res'] = mean_residual(env['pred'], env['labels'])
            env['res_set'] = torch.abs(torch.subtract(env['pred'], env['labels']))

        train_l1 = torch.stack([e['l1'] for e in train_envs]).mean()
        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_l1.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log('train_l1', train_l1)
        if step % 100 == 0:
            print_stats(step, logger)

    # Calculate coverage gap and metrics
    coverage_summary = []
    Total_variation_summary = []
    KL_summary = []
    Wasserstein_summary = []
    Expectation_diff_summary = []

    cal_envs_uni['pred'] = mlp(cal_envs_uni['images'])
    cal_envs_uni['res_set'] = torch.abs(torch.subtract(cal_envs_uni['pred'], cal_envs_uni['labels']))
    R_cal = torch.abs(cal_envs_uni['pred'] - cal_envs_uni['labels']).cpu().detach().numpy().flatten()

    for env in test_envs:
        # calculate coverage
        env['pred'] = mlp(env['images'])
        env['res_set'] = torch.abs(torch.subtract(env['pred'], env['labels']))

        pred = env['pred'].cpu().detach().numpy().flatten()
        label = env['labels'].cpu().detach().numpy().flatten()
        weights = np.ones(len(R_cal))
        coverage_hour = []
        for alpha in alpha_array:
            Covered, Size = Testing_Coverage(label, pred, R_cal, alpha, weights)  # correct func
            coverage_hour.append(Covered / len(label))

        coverage_summary.append(coverage_hour)

        # Total variation distance
        R_test = np.abs(pred - label)
        Total_variation_summary.append(TV_Distance(R_test, R_cal, grid_points=100))
        KL_summary.append(KL_Divergence(R_test, R_cal, grid_points=100))
        Expectation_diff_summary.append(abs(np.mean(R_test) - np.mean(R_cal)))
        Wasserstein_summary.append(Wasserstein_distance(cal_envs_uni['res_set'].flatten(), env['res_set'].flatten(),
                                                        u_weights=None, v_weights=None).cpu().detach().numpy())

    coverage_summary = np.stack(coverage_summary)
    gap_summary = np.abs(coverage_summary - coverage_summary_target)
    gap_overall = np.mean(gap_summary, axis=1)
    gap_worst = np.max(gap_summary, axis=1)

    # print results
    result = {}
    result['test_overall_gap'] = gap_overall
    result['test_worst_gap'] = gap_worst
    result['wasserstein_distance'] = Wasserstein_summary
    result['total_variation_distance'] = Total_variation_summary
    result['KL_divergence'] = KL_summary
    result['expectation_difference'] = Expectation_diff_summary

    RESULT.append(result)
    pickle.dump(RESULT,
                open(project_path + "/correlation/correlation_result/" + flags.dataset + "/" + flags.version, "wb"))
