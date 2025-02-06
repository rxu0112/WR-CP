import argparse
from torch import optim
import pickle
from module import *
import os

parser = argparse.ArgumentParser(description='WRCP guarantee')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=1e-3)  # 0.001
parser.add_argument('--penalty_anneal_iters', type=int, default=1000)
# airfoil seattle pemsd4 pemsd8 1000, japan states 1500
parser.add_argument('--steps', type=int, default=1000)
# airfoil seattle pemsd4 pemsd8 3000, japan states 3500
parser.add_argument('--dataset', type=str, default='pemsd8')
parser.add_argument('--version', type=str, default='v3')
flags = parser.parse_args()

# this is the code for obtaining coverage guarantee 1-alpha-alpha_D of wasserstein regularization

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
            nn.Dropout(),  # not for japan, states
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

    Recalibration_RESULT = []
    for w_wr in [1]:
        if w_wr:
            print('w_wr:' + str(w_wr))
        else:
            print('erm')

        logger = Logger()
        alpha_array = np.linspace(0.1, 0.9, 9)
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

        cal_envs_uni['pred'] = mlp(cal_envs_uni['images'])
        cal_envs_uni['res_set'] = torch.abs(torch.subtract(cal_envs_uni['pred'], cal_envs_uni['labels']))
        R_cal = torch.abs(cal_envs_uni['pred'] - cal_envs_uni['labels']).cpu().detach().numpy().flatten()
        gap_bound_list = []

        for alpha in alpha_array:
            max_gap = []

            for env_te in test_envs:
                gap_env = 0
                env_te['pred'] = mlp(env_te['images'])
                env_te['res'] = mean_residual(env_te['pred'], env_te['labels'])
                env_te['res_set'] = torch.abs(torch.subtract(env_te['pred'], env_te['labels']))
                pred = env_te['pred'].cpu().detach().numpy().flatten()
                label = env_te['labels'].cpu().detach().numpy().flatten()
                weights = env_te['weights'].cpu().detach().numpy()
                tau_test = weighted_quantile(R_cal, 1 - alpha, weights)

                for env_tr in train_envs:
                    coverage_tr = (env_tr['res_set'] < tau_test).sum().item() / env_tr['res_set'].numel()
                    mask = cal_envs_uni['res_set'] < tau_test
                    weights_normalized = (env_tr['weights'] / env_tr['weights'].sum()).reshape(-1, 1)
                    coverage_cal = ((weights_normalized * mask).sum() / weights_normalized.sum()).item()
                    gap_tr = abs(coverage_tr-coverage_cal)
                    if gap_tr > gap_env:
                        gap_env = gap_tr

                max_gap.append(gap_env)

            gap_bound_list.append(max_gap)

        result = {}
        result['weight'] = np.array(w_wr)
        result['gap_bound_list'] = gap_bound_list
        Recalibration_RESULT.append(result)

    pickle.dump(Recalibration_RESULT,
                open(project_path + "/guaranteed/guaranteed_result/" + flags.dataset + "/" + flags.version, "wb"))
