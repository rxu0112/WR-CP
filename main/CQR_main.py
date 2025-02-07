import argparse
from torch import optim
import pickle
from module import *
import os

parser = argparse.ArgumentParser(description='CQR')
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

    try:
        RESULT = pickle.load(open(project_path + "/main/main_result/" + flags.dataset + "/" + flags.version, "rb"))
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        RESULT = []
        print('No previous result')

    alpha_array = np.linspace(0.1, 0.9, 9)
    test_detailed_coverage = [] #env,alpha
    test_detailed_size = [] #env,alpha
    test_overall_gap = []
    test_overall_size = []

    for alpha in alpha_array:
        print(alpha)
        mlp_lo = MLP(train_envs[0]['images'].shape[1]).cuda()
        optimizer_lo = optim.Adam(mlp_lo.parameters(), lr=flags.lr)
        mlp_hi = MLP(train_envs[0]['images'].shape[1]).cuda()
        optimizer_hi = optim.Adam(mlp_hi.parameters(), lr=flags.lr)

        for step in range(flags.steps):
            cal_envs_uni['pred_lo'] = mlp_lo(cal_envs_uni['images'])
            for env in train_envs:
                env['pred_lo'] = mlp_lo(env['images'])
                env['pinball_lo'] = pinball_loss(env['pred_lo'], env['labels'], alpha=alpha/2)

            train_pinball = torch.stack([e['pinball_lo'] for e in train_envs]).mean()
            weight_norm = torch.tensor(0.).cuda()
            for w in mlp_lo.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_pinball.clone()
            loss += flags.l2_regularizer_weight * weight_norm

            optimizer_lo.zero_grad()
            loss.backward()
            optimizer_lo.step()

            if step % 100 == 0:
                print(step)
                print(train_pinball.cpu().detach().numpy())

        for step in range(flags.steps):
            cal_envs_uni['pred_hi'] = mlp_hi(cal_envs_uni['images'])
            for env in train_envs:
                env['pred_hi'] = mlp_hi(env['images'])
                env['pinball_hi'] = pinball_loss(env['pred_hi'], env['labels'], alpha=1-alpha/2)

            train_pinball = torch.stack([e['pinball_hi'] for e in train_envs]).mean()
            weight_norm = torch.tensor(0.).cuda()
            for w in mlp_hi.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_pinball.clone()
            loss += flags.l2_regularizer_weight * weight_norm

            optimizer_hi.zero_grad()
            loss.backward()
            optimizer_hi.step()

            if step % 100 == 0:
                print(step)
                print(train_pinball.cpu().detach().numpy())

        # check coverage on test
        R_cal = []
        for i in range(len(cal_envs_uni['labels'])):
            R_cal.append(max(cal_envs_uni['pred_lo'][i] - cal_envs_uni['labels'][i], cal_envs_uni['labels'][i] -
                             cal_envs_uni['pred_hi'][i]).cpu().detach().numpy().flatten())

        R_cal = np.array(R_cal).flatten()
        weights = np.ones(len(R_cal))
        quantile = weighted_quantile(R_cal, 1 - alpha, weights)

        coverage_env = []
        size_env = []
        for env in test_envs:
            env['pred_lo'] = mlp_lo(env['images'])
            env['pred_hi'] = mlp_hi(env['images'])
            lower_bound = env['pred_lo'].cpu().detach().numpy().flatten()-quantile
            upper_bound = env['pred_hi'].cpu().detach().numpy().flatten()+quantile
            label = env['labels'].cpu().detach().numpy().flatten()
            Covered = np.sum((label[:, np.newaxis] >= lower_bound[:, np.newaxis]) &
                   (label[:, np.newaxis] <= upper_bound[:, np.newaxis]))
            Size = np.mean(upper_bound - lower_bound)
            Coverage = Covered/len(label)
            coverage_env.append(Coverage)
            size_env.append(Size)

        overall_gap = np.mean(np.abs(np.array(coverage_env)-(1-alpha)))
        overall_size = np.mean(size_env)

        test_detailed_coverage.append(coverage_env)
        test_detailed_size.append(size_env)
        test_overall_gap.append(overall_gap)
        test_overall_size.append(overall_size)

    result = {}
    result['test_overall_gap'] = np.mean(test_overall_gap)
    result['test_overall_size'] = np.mean(test_overall_size)
    result['test_detailed_coverage'] = np.array(test_detailed_coverage).T
    result['test_detailed_size'] = np.array(test_detailed_size).T
    result['opt'] = 'qr'
    result['cp'] = 'cqr'
    RESULT.append(result)

    pickle.dump(RESULT, open(project_path + "/main/main_result/" + flags.dataset + "/raw/" + flags.version, "wb"))
