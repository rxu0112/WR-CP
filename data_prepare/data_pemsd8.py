import numpy as np
import pandas as pd
import torch
from module import *
from sklearn.model_selection import GridSearchCV
import pickle
import random
import os

def get_hours(date, roads):
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fil_path = os.path.join(project_path, "data\\pemsd8\\raw\\pemsd8.npz")
    raw_data = np.load(fil_path)['data']  # (time*node*(volume,occupy,speed)
    date_range = pd.date_range(start='2016-07-01 00:00:00', periods=17856, freq='5T')
    speed_dataset = pd.DataFrame(data=raw_data[:, :, 2], index=date_range.strftime('%Y-%m-%d %H:%M:%S'))
    volume_dataset = pd.DataFrame(data=raw_data[:, :, 0], index=date_range.strftime('%Y-%m-%d %H:%M:%S'))
    date_list = list(date['Date'])
    x = []
    y = []
    for road in roads:
        hour = [0, 24]
        print(road)
        time_list = get_time_list(hour[0], hour[1])

        speed_site = speed_dataset.iloc[:, road[0]:road[1]]
        volume_site = volume_dataset.iloc[:, road[0]:road[1]]

        date_index = []
        for i in range(len(date_list)):
            for j in range(len(speed_site.index)):
                if date_list[i] in speed_site.index[j]:
                    date_index.append(j)
        speed_site_date = speed_site.iloc[date_index, :]
        volume_site_date = volume_site.iloc[date_index, :]

        time_index = []
        for i in range(len(time_list)):
            for j in range(len(speed_site_date.index)):
                if time_list[i] in speed_site_date.index[j]:
                    time_index.append(j)
        speed_site_date_time = speed_site_date.iloc[time_index, :]
        volume_site_date_time = volume_site_date.iloc[time_index, :]

        speed_site_center = speed_site_date_time.iloc[:, 1]
        speed_diff = []
        for i in range(len(time_list) - 1):
            speed_diff.append(speed_site_center.values[(i + 1) * len(date_list):(i + 2) * len(date_list)]
                              - speed_site_center.values[i * len(date_list):(i + 1) * len(date_list)])
        speed_diff = np.concatenate(speed_diff)
        speed_array = speed_site_date_time.iloc[0:len(speed_diff), :].values
        volume_array = volume_site_date_time.iloc[0:len(speed_diff), :].values
        feature = np.concatenate((speed_array, volume_array), axis=1)
        x.append(feature)
        y.append(speed_diff)

    return [{
        'images': torch.tensor(x_, dtype=torch.float32).cuda(),
        'labels': torch.tensor(y_.reshape(-1, 1), dtype=torch.float32).cuda(),
        'info': rd_,
    } for x_, y_, rd_ in zip(x, y, roads)]


def get_time_list(start, end):
    time_list = []
    if start > end:
        if end == 0:
            for hour in range(start, 24):
                for minute in range(0, 60, 5):
                    time_list.append(f'{hour:02d}:{minute:02d}:00')
        else:
            for hour in range(start, 24):
                for minute in range(0, 60, 5):
                    time_list.append(f'{hour:02d}:{minute:02d}:00')
            for hour in range(0, end):
                for minute in range(0, 60, 5):
                    time_list.append(f'{hour:02d}:{minute:02d}:00')
    else:
        for hour in range(start, end):
            for minute in range(0, 60, 5):
                time_list.append(f'{hour:02d}:{minute:02d}:00')
    return time_list


def sampling(sample_size, env):
    combined_tensor = torch.cat((env['images'], env['labels']), dim=1)
    shuffled_tensor = combined_tensor[torch.randperm(combined_tensor.size(0))]
    image_samples, label_samples = shuffled_tensor[:, :6], shuffled_tensor[:, 6:]
    return image_samples[:sample_size], label_samples[:sample_size]


def generate_pairs(num_pairs, step, end):
    pairs = []
    for _ in range(num_pairs):
        a1 = random.randint(0, end)  # a1 can be from 0 to 23
        a2 = a1 + step  # Ensure a2 - a1 = 2
        pairs.append([a1, a2])
    return pairs


def generate_triplets(n):
    triplets = []
    for _ in range(n):
        ai = random.uniform(0, 1)  # give more weight to one domain to be more distinct with P_{XY}
        bi = random.uniform(0, 1 - ai)  # Ensure ai + bi <= 1
        ci = 1 - ai - bi
        triplets.append([ai, bi, ci])
    return triplets


ver = 1
roads = [[114, 117], [132, 135], [31, 34], [64, 67], [165, 168], [138, 141], [112, 115], [123, 126], [68, 71], [133, 136]]
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workday_path = os.path.join(project_path, "data\\pemsd8\\raw\\Workday.txt")
workday = pd.read_table(workday_path)
workday_test = workday.sample(frac=0.333)
workday_cal = workday.drop(workday_test.index).sample(frac=0.5)
workday_train = workday.drop(workday_test.index).drop(workday_cal.index)

print('generate train envs')
train_envs = get_hours(date=workday_train, roads=roads)
print('generate test envs d')
test_envs_d = get_hours(date=workday_test, roads=roads)
print('generate cal envs')
cal_envs = get_hours(date=workday_cal, roads=roads)

universal_image = []
for e in train_envs + test_envs_d + cal_envs:
    universal_image.append(e['images'])
universal_image = torch.concatenate(universal_image)

print('whiten environments')
for e in train_envs + test_envs_d + cal_envs:
    e['images'] = whiten(e['images'], universal_image)

print('generate a universal environment in calibration set')
universal_image_cal = []
universal_label_cal = []
for e in cal_envs:
    universal_image_cal.append(e['images'])
    universal_label_cal.append(e['labels'])
universal_image_cal = torch.concatenate(universal_image_cal)
universal_label_cal = torch.concatenate(universal_label_cal)
cal_envs_uni = ({'images': universal_image_cal, 'labels': universal_label_cal, 'info': 'universal'})

print('generated sampled calibration set')
cal_size = 10000
cal_envs_uni['images'], cal_envs_uni['labels'] = sampling(cal_size, cal_envs_uni)
cal_envs_uni['roads'] = roads

print('generate test envs q')
test_envs_q_num = 100
test_from = 3
test_q_size = 2000
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

pickle.dump(train_envs, open(project_path + "/data/pemsd8/processed/train_v" + str(ver), "wb"))
pickle.dump(test_envs_q, open(project_path + "/data/pemsd8/processed/test_v" + str(ver), "wb"))
np.save(project_path + '/data/pemsd8/processed/cal_uni_v' + str(ver) + '.npy', cal_envs_uni)
