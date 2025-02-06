import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from module import *
from utils import *
from sklearn.model_selection import GridSearchCV
import pickle
import random
from sklearn.preprocessing import PowerTransformer
import os
def generate_triplets(n):
    triplets = []
    for _ in range(n):
        ai = random.uniform(0, 1)
        bi = random.uniform(0, 1 - ai)  # Ensure ai + bi <= 1
        ci = 1 - ai - bi
        triplets.append([ai, bi, ci])
    return triplets

ver = 1

column_names = ['frequency', 'angle', 'length', 'velocity', 'thickness', 'pressure']

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(project_path, "data\\airfoil\\raw\\airfoil_self_noise.dat")
df = pd.read_csv(file_path, delimiter='\t', names=column_names)

A = df['frequency'].quantile(0.33)
B = df['frequency'].quantile(0.66)


subset1 = df[df['frequency'] < A]
subset2 = df[(df['frequency'] >= A) & (df['frequency'] < B)]
subset3 = df[df['frequency'] >= B]


subset1B, subset1A = train_test_split(subset1, test_size=0.7)
subset1C, subset1B = train_test_split(subset1B, test_size=0.66)


subset2B, subset2A = train_test_split(subset2, test_size=0.7)
subset2C, subset2B = train_test_split(subset2B, test_size=0.66)


subset3B, subset3A = train_test_split(subset3, test_size=0.7)
subset3C, subset3B = train_test_split(subset3B, test_size=0.66)


df_1 = pd.concat([subset1A, subset2B, subset3C])
df_2 = pd.concat([subset1B, subset2C, subset3A])
df_3 = pd.concat([subset1C, subset2A, subset3B])


df_1['pressure'] = df_1['pressure']+df_1['pressure']/1000*np.random.normal(0, 10, df_1.shape[0])
df_2['pressure'] = df_2['pressure']+df_2['pressure']/np.random.normal(0, 10, df_2.shape[0])
df_3['pressure'] = df_3['pressure']+np.random.normal(0, 10, df_3.shape[0])

df_1_train = df_1.iloc[::3]
df_1_cal = df_1.iloc[1::3]
df_1_test = df_1.iloc[2::3]

df_2_train = df_2.iloc[::3]
df_2_cal = df_2.iloc[1::3]
df_2_test = df_2.iloc[2::3]

df_3_train = df_3.iloc[::3]
df_3_cal = df_3.iloc[1::3]
df_3_test = df_3.iloc[2::3]

test_size = 170
test_envs_num = 30 # more test envs will make scp,iw scpe more diverse(higher std)
triplets_list = generate_triplets(test_envs_num)
df_test = []
for portion in triplets_list:
    sampled_df1 = df_1_test.sample(n=int(portion[0] * test_size), replace=True)
    sampled_df2 = df_2_test.sample(n=int(portion[1] * test_size), replace=True)
    sampled_df3 = df_3_test.sample(n=int(portion[2] * test_size), replace=True)
    new_df = pd.concat([sampled_df1, sampled_df2, sampled_df3], ignore_index=True)
    df_test.append(new_df)


##
train_envs, cal_envs, test_envs = [], [], []
df_train = [df_1_train, df_2_train, df_3_train]
df_cal = [df_1_cal, df_2_cal, df_3_cal] #manually change cal will further change performance of scp and cpdi


for idx, env in enumerate(df_train):
    train_envs.append({})
    image = torch.tensor(env.iloc[:, :-1].values, dtype=torch.float32).cuda()
    label = torch.tensor(env.iloc[:, -1].values, dtype=torch.float32).view(-1, 1).cuda()
    train_envs[idx]['images'] = image
    train_envs[idx]['labels'] = label
    train_envs[idx]['info'] = idx

for idx, env in enumerate(df_cal):
    cal_envs.append({})
    image = torch.tensor(env.iloc[:, :-1].values, dtype=torch.float32).cuda()
    label = torch.tensor(env.iloc[:, -1].values, dtype=torch.float32).view(-1, 1).cuda()
    cal_envs[idx]['images'] = image
    cal_envs[idx]['labels'] = label
    cal_envs[idx]['info'] = idx

for idx, env in enumerate(df_test):
    test_envs.append({})
    image = torch.tensor(env.iloc[:, :-1].values, dtype=torch.float32).cuda()
    label = torch.tensor(env.iloc[:, -1].values, dtype=torch.float32).view(-1, 1).cuda()
    test_envs[idx]['images'] = image
    test_envs[idx]['labels'] = label
    test_envs[idx]['info'] = idx



print('target transformation')
label_sizes = []
universal_label = []
for e in train_envs + test_envs + cal_envs:
    universal_label.append(e['labels'])
    label_sizes.append(len(e['labels']))
universal_label = torch.concatenate(universal_label)
split_indices = np.cumsum(label_sizes)[:-1]
# pt = PowerTransformer(method='yeo-johnson')
# transformed_label = pt.fit_transform(universal_label.cpu())
sqrt_label = np.sqrt(np.abs(universal_label.cpu()))
transformed_label = np.where(universal_label.cpu() < 0, -sqrt_label, sqrt_label)
transformed_label_list = np.split(transformed_label, split_indices)
for (j, e) in enumerate(train_envs + test_envs + cal_envs):
    e['labels'] = torch.tensor(transformed_label_list[j], dtype=torch.float32).cuda()

print('whiten environments')
universal_image = []
for e in train_envs + test_envs + cal_envs:
    universal_image.append(e['images'])
universal_image = torch.concatenate(universal_image)

universal_label = []
for e in train_envs + test_envs + cal_envs:
    universal_label.append(e['labels'])
universal_label = torch.concatenate(universal_label)

for e in train_envs + test_envs + cal_envs:
    e['images'] = whiten(e['images'], universal_image)
    e['labels'] = whiten(e['labels'], universal_label)


print_env_info(train_envs, test_envs, cal_envs)

print('generate a universal environment in calibration set')
universal_image_cal = []
universal_label_cal = []
for e in cal_envs:
    universal_image_cal.append(e['images'])
    universal_label_cal.append(e['labels'])
universal_image_cal = torch.concatenate(universal_image_cal)
universal_label_cal = torch.concatenate(universal_label_cal)
cal_envs_uni = ({'images': universal_image_cal, 'labels': universal_label_cal, 'info': 'universal'})

print('Bandwidth selection for calibration')
kde_params = {'bandwidth': np.logspace(-1, 0.7, 20)}
kde_grid_cal = GridSearchCV(KernelDensity(kernel='gaussian'), kde_params)
kde_grid_cal.fit(cal_envs_uni['images'].cpu())
bandwidth_cal_uni = kde_grid_cal.best_estimator_.bandwidth
kde_cal_uni = KernelDensity(kernel='gaussian', bandwidth=bandwidth_cal_uni).fit(
    cal_envs_uni['images'].cpu())

print('calculate the weights for each test env to universal cal env')
for e in test_envs:
    weights, kde = weight_calculation(e['images'].cpu().numpy(), cal_envs_uni['images'].cpu().numpy(),
                                      kde_cal_uni, bandwidth_cal_uni)
    e['weights'] = torch.from_numpy(weights).to('cuda:0')  # cal_uni's weights
    e['kde'] = kde

print('calculate weights for ech training env to universal cal env')
for e in train_envs:
    weights, kde = weight_calculation(e['images'].cpu().numpy(), cal_envs_uni['images'].cpu().numpy(),
                                      kde_cal_uni, bandwidth_cal_uni)
    e['weights'] = torch.from_numpy(weights).to('cuda:0')  # cal_uni’s weights
    e['kde'] = kde

# save env data as
pickle.dump(train_envs, open(project_path + "/data/airfoil/processed/train_v" + str(ver), "wb"))
pickle.dump(test_envs, open(project_path + "/data/airfoil/processed/test_v" + str(ver), "wb"))
np.save(project_path + '/data/airfoil/processed/cal_uni_v' + str(ver) + '.npy', cal_envs_uni)