import argparse
from torch import optim
import pickle
from module import *
import os

parser = argparse.ArgumentParser(description='WCCP with WRCP guarantee')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--penalty_anneal_iters', type=int,
                    default=1000)  # 1000 for airfoil seattle pemsd4 pemsd8, 1500 for japan states
parser.add_argument('--steps', type=int,
                    default=3000)  # 3000 for airfoil seattle pemsd4 pemsd8, 3500 for japan states
parser.add_argument('--dataset', type=str, default='airfoil')  # argument for specifying the dataset
parser.add_argument('--version', type=str, default='v1')  # argument for specifying the dataset
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
            lin1, nn.Tanh(),
            nn.Dropout(),
            lin2, nn.Tanh(),
            nn.Dropout(), # remove the last dropout layer for japan and states
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
    RESULT_BOUND = pickle.load(
            open(project_path + "/guaranteed/guaranteed_result/" + flags.dataset + "/" + flags.version, "rb"))

    for w_wr in [0]:  # set w_wr = 0 since WC-CP is optimized by empirical risk minimization (erm).
        if w_wr:
            print('w_wr:' + str(w_wr))
        else:
            print('erm')

        logger = Logger()
        alpha_array = np.round(np.linspace(0.1, 0.9, 9), 2)
        coverage_hour_target = 1 - alpha_array
        coverage_summary_target = [coverage_hour_target[:] for _ in range(len(test_envs))]

        mlp = MLP(train_envs[0]['images'].shape[1]).cuda()
        optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)


        pretty_print('step', 'train l1', 'wr penalty')

        for step in range(flags.steps):
            cal_envs_uni['pred'] = mlp(cal_envs_uni['images'])
            cal_envs_uni['res_set'] = torch.abs(torch.subtract(cal_envs_uni['pred'], cal_envs_uni['labels']))

            for env in train_envs:
                env['pred'] = mlp(env['images'])
                env['l1'] = mean_l1(env['pred'], env['labels'])
                env['res'] = mean_residual(env['pred'], env['labels'])
                env['res_set'] = torch.abs(torch.subtract(env['pred'], env['labels']))

            train_l1 = torch.stack([e['l1'] for e in train_envs]).mean()
            train_res = torch.stack([e['res'] for e in train_envs]).mean()

            wr_penalty = get_wr_penalty(train_envs, cal_envs_uni)

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_l1.clone()

            loss += flags.l2_regularizer_weight * weight_norm

            if w_wr:
                if step >= flags.penalty_anneal_iters:
                    loss /= w_wr
                loss += wr_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log('train_l1', train_l1)
            logger.log('train_res', [e['res'] for e in train_envs])
            logger.log('wr_penalty', wr_penalty)

            if step % 100 == 0:
                print_stats(step, logger)

        # check coverage on test
        test_overall_gap_list = []
        test_worst_gap_list = []
        test_res_list = []
        test_detailed_coverage_list = []
        test_detailed_size_list = []
        test_W_list = []

        for env in test_envs:
            env['pred'] = mlp(env['images'])
            env['res'] = mean_residual(env['pred'], env['labels'])
            env['res_set'] = torch.abs(torch.subtract(env['pred'], env['labels']))

        for i in range(3):
            coverage_summary_test = []
            size_summary_test = []
            W_summary_test = []

            largest_list_at_diff_alpha = []
            for a, alpha in enumerate(alpha_array):
                desired_coverage = 1-alpha-RESULT_BOUND[0]['gap_bound_list'][a]
                largest_list_at_one_alpha = []
                for d, _ in enumerate(desired_coverage):
                    largest = 0
                    for env in test_envs:
                        R_test = torch.abs(env['pred'] - env['labels']).cpu().detach().numpy().flatten()
                        weights = np.ones(len(R_test))
                        quantile = weighted_quantile(R_test, desired_coverage[d], weights)
                        if quantile >= largest:
                            largest = quantile
                    largest_list_at_one_alpha.append(largest)
                largest_list_at_diff_alpha.append(largest_list_at_one_alpha)
                largest_quantile_array = np.array(largest_list_at_diff_alpha)

            for idx, env in enumerate(test_envs):
                pred = env['pred'].cpu().detach().numpy().flatten()
                label = env['labels'].cpu().detach().numpy().flatten()
                coverage_hour = []
                size_hour = []

                for quantile in largest_quantile_array[:,idx]:
                    Size = 2 * quantile
                    up = pred + quantile
                    low = pred - quantile
                    Covered = np.sum((label[:, np.newaxis] >= low[:, np.newaxis]) &
                                     (label[:, np.newaxis] <= up[:, np.newaxis]))
                    coverage_hour.append(Covered / len(label))
                    size_hour.append(Size)

                coverage_summary_test.append(coverage_hour)
                size_summary_test.append(size_hour)

                W_env = Wasserstein_distance(cal_envs_uni['res_set'].flatten(), env['res_set'].flatten(),
                                         u_weights=None, v_weights=None).cpu().detach().numpy()
                W_summary_test.append(W_env)

            test_res_trial = torch.stack([e['res'] for e in test_envs]).mean()
            coverage_summary_test = np.stack(coverage_summary_test)
            size_summary_test = np.stack(size_summary_test)
            test_overall_gap_trial = np.mean(np.abs(coverage_summary_test - coverage_summary_target))
            test_worst_gap_trial = np.max(np.mean(np.abs(coverage_summary_test - coverage_summary_target), axis=1))

            test_res_list.append(test_res_trial.cpu().detach().numpy())
            test_detailed_coverage_list.append(coverage_summary_test)
            test_detailed_size_list.append(size_summary_test)
            test_overall_gap_list.append(test_overall_gap_trial)
            test_worst_gap_list.append(test_worst_gap_trial)
            test_W_list.append(np.mean(W_summary_test))

        test_res = np.mean(test_res_list)
        test_detailed_coverage = np.mean(test_detailed_coverage_list,
                                         axis=0)  # get (envs, alphas) array after average over i
        test_detailed_size = np.mean(test_detailed_size_list,
                                     axis=0)  # get (envs, alphas) array after average over i
        test_overall_size = np.mean(test_detailed_size)  # size float after average over env, alpha, i
        test_overall_gap = np.mean(test_overall_gap_list)  # gap float after average over env, alpha, i
        test_worst_gap = np.mean(test_worst_gap_list)
        test_W = np.mean(test_W_list)

        result = {'test_res': test_res, 'test_overall_gap': test_overall_gap, 'test_overall_size': test_overall_size,
                  'test_worst_gap': test_worst_gap, 'test_detailed_coverage': test_detailed_coverage,
                  'test_detailed_size': test_detailed_size, 'test_W': test_W, 'opt': 'erm', 'weight': 0, 'cp': 'wc-scp'}

        RESULT.append(result)

        pickle.dump(RESULT, open(project_path + "/guaranteed/guaranteed_result/" + flags.dataset + "/wc_" + flags.version, "wb"))
