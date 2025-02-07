import numpy as np
import pandas as pd
import torch
from module import *
from sklearn.model_selection import GridSearchCV
import pickle
import random
import os

def sampling(sample_size, env):
    combined_tensor = torch.cat((env['images'], env['labels']), dim=1)
    shuffled_tensor = combined_tensor[torch.randperm(combined_tensor.size(0))]
    image_samples, label_samples = shuffled_tensor[:, :2], shuffled_tensor[:, 2:]
    return image_samples[:sample_size], label_samples[:sample_size]


def generate_triplets(n):
    triplets = []
    for _ in range(n):
        ai = random.uniform(0, 1)  # give more weight to one domain to be more distinct with P_{XY}
        bi = random.uniform(0, 1 - ai)  # Ensure ai + bi <= 1
        ci = 1 - ai - bi
        triplets.append([ai, bi, ci])
    return triplets


def preprocess(data, endpoint, node):
    I_dt_stack, I_st_stack, I_sum_stack = [], [], []

    for i in range(endpoint.shape[0]):
        I = data.loc[:, node].values.flatten()[endpoint[i, 0]:endpoint[i, 1]]  # data is (record,location)
        I_sum = np.zeros(len(I) - 1)
        for j in range(len(I) - 1):
            I_sum[j] = np.sum(I[0:j])
        I_sum_ratio = np.zeros(len(I) - 1)
        for j in range(len(I) - 1):
            I_sum_ratio[j] = I_sum[j] / I_sum[-1]
        I_dt = np.diff(I)
        I_st = I[0:-1]
        I_dt_stack.append(I_dt)
        I_st_stack.append(I_st)
        I_sum_stack.append(I_sum)

    I_st_stack = np.concatenate(I_st_stack)
    I_dt_stack = np.concatenate(I_dt_stack)
    I_sum_stack = np.concatenate(I_sum_stack)

    Input_stack = np.array([I_st_stack, I_dt_stack, I_sum_stack]).T

    return Input_stack


ver = 1 # modify ver from 1 to 10 for 10 sampling trials
locations = [10, 32, 42, 45, 20, 24, 16, 47, 48, 31]
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(project_path, "data\\states\\raw\\state360.txt")
raw_data = pd.read_csv(file_path, sep="\t", header=None)

endpoint = np.array([[0, 43], [43, 89], [89, 143], [143, 195], [195, 247], [249, 301], [301, 352]])

train_envs, cal_envs, test_envs_d = [], [], []

for index, loc in enumerate(locations):
    Input_stack = preprocess(raw_data, endpoint, loc)
    np.random.shuffle(Input_stack)
    split_size1 = int(len(Input_stack) * 2 / 5)
    split_size2 = int(len(Input_stack) * 1 / 5)
    train_data = Input_stack[:split_size1]
    cal_data = Input_stack[split_size1:split_size1 + split_size2]
    test_data = Input_stack[split_size1 + split_size2:]

    Image_train = np.array((train_data[:, 0], train_data[:, 2])).T
    Label_train = train_data[:, 1].reshape(-1, 1)
    train_envs.append({})
    train_envs[index]['images'] = torch.tensor(Image_train, dtype=torch.float32).cuda()
    train_envs[index]['labels'] = torch.tensor(Label_train, dtype=torch.float32).cuda()

    Image_cal = np.array((cal_data[:, 0], cal_data[:, 2])).T
    Label_cal = cal_data[:, 1].reshape(-1, 1)
    cal_envs.append({})
    cal_envs[index]['images'] = torch.tensor(Image_cal, dtype=torch.float32).cuda()
    cal_envs[index]['labels'] = torch.tensor(Label_cal, dtype=torch.float32).cuda()

    Image_test = np.array((test_data[:, 0], test_data[:, 2])).T
    Label_test = test_data[:, 1].reshape(-1, 1)
    test_envs_d.append({})
    test_envs_d[index]['images'] = torch.tensor(Image_test, dtype=torch.float32).cuda()
    test_envs_d[index]['labels'] = torch.tensor(Label_test, dtype=torch.float32).cuda()

universal_image = []
for e in train_envs + test_envs_d + cal_envs:
    universal_image.append(e['images'])
universal_image = torch.concatenate(universal_image)

universal_label = []
for e in train_envs + test_envs_d + cal_envs:
    universal_label.append(e['labels'])
universal_label = torch.concatenate(universal_label)

print('whiten environments')
for e in train_envs + test_envs_d + cal_envs:
    e['images'] = whiten(e['images'], universal_image)
    e['labels'] = whiten(e['labels'], universal_label)

print('generate a universal environment in calibration set')
universal_image_cal = []
universal_label_cal = []
for e in cal_envs:
    universal_image_cal.append(e['images'])
    universal_label_cal.append(e['labels'])
universal_image_cal = torch.concatenate(universal_image_cal)
universal_label_cal = torch.concatenate(universal_label_cal)
cal_envs_uni = ({'images': universal_image_cal, 'labels': universal_label_cal, 'info': 'universal'})
cal_envs_uni['locations'] = locations

print('generate test envs q')
test_envs_q_num = 100
test_from = 3
test_q_size = 135
test_envs_q = []
portions = generate_triplets(test_envs_q_num)
for i in range(test_envs_q_num):
    picked_values = random.sample(range(10), test_from)
    normalized_numbers = portions[i]
    print(picked_values)
    print(normalized_numbers)
    images_i = []
    labels_i = []
    for (index, j) in enumerate(picked_values):
        images_j, labels_j = sampling(int(test_q_size * normalized_numbers[index]), test_envs_d[j])
        images_i.append(images_j)
        labels_i.append(labels_j)
    images_i = torch.concatenate(images_i)
    labels_i = torch.concatenate(labels_i)
    test_envs_q.append({'images': images_i, 'labels': labels_i})

print('Bandwidth selection for calibration')
kde_params = {'bandwidth': np.logspace(-1, 0.7, 20)}
kde_grid_cal = GridSearchCV(KernelDensity(kernel='gaussian'), kde_params)
kde_grid_cal.fit(cal_envs_uni['images'].cpu())
bandwidth_cal_uni = kde_grid_cal.best_estimator_.bandwidth
kde_cal_uni = KernelDensity(kernel='gaussian', bandwidth=bandwidth_cal_uni).fit(
    cal_envs_uni['images'].cpu())

print('calculate weights for each test env to universal cal env')
for e in test_envs_q:
    weights, kde = weight_calculation(e['images'].cpu().numpy(), cal_envs_uni['images'].cpu().numpy(),
                                      kde_cal_uni, bandwidth_cal_uni)
    e['weights'] = torch.from_numpy(weights).to('cuda:0')  # cal_uni's weights
    e['kde'] = kde

print('calculate weights for each training env to universal cal env')
for e in train_envs:
    weights, kde = weight_calculation(e['images'].cpu().numpy(), cal_envs_uni['images'].cpu().numpy(),
                                      kde_cal_uni, bandwidth_cal_uni)
    e['weights'] = torch.from_numpy(weights).to('cuda:0')  # cal_uni’s weights
    e['kde'] = kde

pickle.dump(train_envs, open(project_path + "/data/states/processed/train_v" + str(ver), "wb"))
pickle.dump(test_envs_q, open(project_path + "/data/states/processed/test_v" + str(ver), "wb"))
np.save(project_path + '/data/states/processed/cal_uni_v' + str(ver) + '.npy', cal_envs_uni)
