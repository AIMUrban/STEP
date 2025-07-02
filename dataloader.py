import datetime
import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def load_npy_file(save_path):
    loaded_data = np.load(save_path, allow_pickle=True)
    return loaded_data


class MyDataset(Dataset):
    def __init__(self, dataset_path, device, load_mode, args):
        self.seq_len = 20
        self.device = device
        self.load_mode = load_mode
        self.dataset_path = dataset_path
        self.args = args
        self.num_timeslots = args.num_timeslots
        self.dataset ='TC' in dataset_path
        self.user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
        self.location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

        if load_mode == 'test':
            self.data = load_npy_file(os.path.join(dataset_path, f'{load_mode}_{self.num_timeslots}.npy'))
        else:
            if not os.path.exists(os.path.join(dataset_path, f'{load_mode}_{self.num_timeslots}.npy')):
                self.generate_data(load_mode='train', num_timeslots=self.num_timeslots)
                self.generate_data(load_mode='test', num_timeslots=self.num_timeslots)
            self.data = load_npy_file(os.path.join(dataset_path, f'{load_mode}_{self.num_timeslots}.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return data


    def generate_data(self, load_mode, num_timeslots):
        res = []
        with open(os.path.join(self.dataset_path, f'{load_mode}.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line_i, line in enumerate(tqdm(lines, desc=f'Initial {load_mode} data for {self.num_timeslots}')):
                user = line.strip().split(',')[0]
                stay_points = line.strip().split(',')[1:]
 
                sequence_count, left = divmod(len(stay_points), self.seq_len)
                assert sequence_count > 0, f"{user}'s does not have enough data."
                sequence_count -= 1 if left == 0 else 0
                for i in range(sequence_count):
                    split_start = i * self.seq_len
                    split_end = (i + 1) * self.seq_len
                    location_x = [self.location2id[item.split('@')[0]] for item in stay_points[split_start:split_end]]
                    timestamp_x = [item.split('@')[1] for item in stay_points[split_start:split_end]]
                    location_y = [self.location2id[item.split('@')[0]] for item in
                                  stay_points[split_start + 1:split_end + 1]]
                    timestamp_y = [item.split('@')[1] for item in stay_points[split_start + 1:split_end + 1]]
                    timeslot_y = []
                    timeslot_x = []
                    weekday_x = []
                    weekend_y = []
                    for item in timestamp_x:
                        weekday, hour = datetime_to_features(item)
                        if num_timeslots == 168:
                            time_x = 24 * weekday + hour
                        elif num_timeslots == 48:
                            if weekday in [0, 6]:
                                time_x = hour + 24
                            else:
                                time_x = hour
                        elif num_timeslots == 24:
                            time_x = hour
                        timeslot_x.append(time_x)
                        weekday_x.append(weekday)
                    for item in timestamp_y:
                        weekday, hour = datetime_to_features(item)
                        if num_timeslots == 168:
                            time_y = weekday * 24 +hour
                        elif num_timeslots == 48:
                            if weekday in [0, 6]:
                                time_y = hour + 24
                            else:
                                time_y = hour
                        else:
                            time_y = hour
                        timeslot_y.append(time_y)
                        weekend_y.append(weekday)
                    res.append(
                        {
                            'user': self.user2id[user],
                            'location_x': location_x,
                            'timeslot': timeslot_x,
                            'weekday': weekday_x,
                            'location_y': location_y,
                            'timeslot_y': timeslot_y,
                            'weekend_y': weekend_y,
                        }
                    )
        np.save(os.path.join(self.dataset_path, f'{load_mode}_{self.num_timeslots}.npy'), res)


def datetime_to_features(timestamp):
    dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)
    weekday = dt.weekday()
    hour = dt.hour
    return weekday, hour

def calculate_matrix(config, num_timeslots, data_path):
    """
    Calculates the location-timeslot frequency matrix from train.csv,
    row-normalizes it, and saves it to save_path.
    The matrix represents P(timeslot | location).
    """
    loc_time_mat_path = os.path.join(data_path, f'location_timeslot_matrix_{num_timeslots}.npy')
    user_time_mat_path = os.path.join(data_path, f'user_timeslot_matrix_{num_timeslots}.npy')
    loc_loc_mat_path = os.path.join(data_path, f'location_location_matrix_{num_timeslots}.npy')
    if os.path.exists(loc_time_mat_path):
        normalized_loc_time_mat = np.load(loc_time_mat_path)
        normalized_user_time_mat = np.load(user_time_mat_path)
        normalized_loc_loc_mat = np.load(loc_loc_mat_path)
        return {
            'loc_time_mat': normalized_loc_time_mat,
            'user_time_mat': normalized_user_time_mat,
            'loc_loc_mat': normalized_loc_loc_mat,
        }

    num_locations = config.Dataset.num_locations
    num_users = config.Dataset.num_users
    loc_time_matrix = np.zeros((num_locations, num_timeslots), dtype=np.float32)
    user_time_matrix = np.zeros((num_users, num_timeslots), dtype=np.float32)
    loc_loc_matrix = np.zeros((num_locations, num_locations), dtype=np.float32)
    location2id = np.load(os.path.join(data_path, 'location_mapper.npy'), allow_pickle=True).item()
    user2id = np.load(os.path.join(data_path, 'user_mapper.npy'), allow_pickle=True).item()

    # This matrix should be built from the raw training data for comprehensiveness
    raw_train_data_path = os.path.join(data_path, 'train.csv')

    print(f"Calculating frequencies from {raw_train_data_path}...")
    with open(raw_train_data_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc='Processing train.csv for loc-ts matrix'):
            parts = line.strip().split(',')
            user = parts[0]
            uid = user2id[user]
            stay_points = parts[1:]
            prev_location_id = None

            for sp_entry in stay_points:
                loc_str, ts_str = sp_entry.split('@')
                location_id = location2id[loc_str]
                weekday, hour = datetime_to_features(ts_str)
                if num_timeslots == 168:
                    timeslot_id = weekday * 24 + hour
                elif num_timeslots == 48:
                    if weekday in [0, 6]:
                        timeslot_id = hour + 24
                    else:
                        timeslot_id = hour
                else:
                    timeslot_id = hour

                loc_time_matrix[location_id, timeslot_id] += 1
                user_time_matrix[uid, timeslot_id] += 1

                if prev_location_id:
                    loc_loc_matrix[prev_location_id, location_id] += 1
                prev_location_id = location_id

    # Row normalization
    print("Normalizing frequency matrix...")
    loc_time_row_sums = loc_time_matrix.sum(axis=1, keepdims=True)
    user_time_row_sums = user_time_matrix.sum(axis=1, keepdims=True)
    loc_loc_row_sums = loc_loc_matrix.sum(axis=1, keepdims=True)

    normalized_loc_time_mat = np.divide(loc_time_matrix, loc_time_row_sums,
                                  out=np.zeros_like(loc_time_matrix, dtype=np.float32),
                                  where=loc_time_row_sums != 0)
    normalized_user_time_mat = np.divide(user_time_matrix, user_time_row_sums)
    normalized_loc_loc_mat = np.divide(loc_loc_matrix, loc_loc_row_sums)
    np.save(loc_time_mat_path, normalized_loc_time_mat)
    np.save(user_time_mat_path, normalized_user_time_mat)
    np.save(loc_loc_mat_path, normalized_loc_loc_mat)
    print(f"Matrix saved to {loc_time_mat_path}, {user_time_mat_path}, {loc_loc_mat_path}")
    return {
            'loc_time_mat': normalized_loc_time_mat,
            'user_time_mat': normalized_user_time_mat,
            'loc_loc_mat': normalized_loc_loc_mat,
        }