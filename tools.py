import os
from datetime import time

import numpy as np
import torch
import tqdm
import yaml
from easydict import EasyDict


def get_config(path, easy):
    """
    get config yml
    :param easy: is easydict mode
    :param path: yml path
    :return: EasyDict format
    """
    f = open(path, 'r', encoding='utf-8')
    res = yaml.safe_load(f)
    if easy:
        return EasyDict(res)
    else:
        return res


def get_user_input():
    while True:
        user_input = input("Configuration yml file modification completed? (y/n):").strip().lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid Options")

def custom_collate(batch, device):
    batch_dict = {
        'user': torch.tensor([item['user'] for item in batch]).to(device),
        'location_x': torch.stack([torch.tensor(item['location_x']) for item in batch]).to(device),
        'timeslot': torch.stack([torch.tensor(item['timeslot']) for item in batch]).to(device),
        'weekday': torch.stack([torch.tensor(item['weekday']) for item in batch]).to(device),
        'location_y': torch.tensor([item['location_y'] for item in batch]).to(device),
        'timeslot_y': torch.tensor([item['timeslot_y'] for item in batch]).to(device),
    }

    return batch_dict

def update_config(path, key_list, value):
    """
    update config yml
    :param key_list: yml key list
    :param path: yml path
    :param value: corresponding value
    :return:
    """
    config = get_config(path, easy=False)

    current_level = config
    outer_key = key_list[0]
    inner_key = key_list[1]
    if outer_key not in current_level:
        print(f'Update config Error: outermost key {outer_key} not exist!')
        exit()
    if inner_key not in current_level[outer_key]:
        print(f'Update config Error: inner key {inner_key} not exist in {outer_key}!')
        exit()

    current_level[outer_key][inner_key] = value

    with open(path, 'w') as f_writer:
        yaml.dump(config, f_writer, default_flow_style=False)
        f_writer.close()


def get_mapper(dataset_path):
    location_mapper_path = os.path.join(dataset_path, 'location_mapper.npy')
    user_mapper_path = os.path.join(dataset_path, 'user_mapper.npy')

    if os.path.exists(location_mapper_path) and os.path.exists(user_mapper_path):
        return

    location_set = set()
    user_set = set()

    with open(os.path.join(dataset_path, 'train.csv'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(',')
            uid = elements[0]
            item_seq = elements[1:]

            user_set.add(uid)

            for item in item_seq:
                loc = item.split('@')[0]
                location_set.add(loc)
        f.close()
    with open(os.path.join(dataset_path, 'test.csv'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(',')
            uid = elements[0]
            item_seq = elements[1:]

            user_set.add(uid)

            for item in item_seq:
                loc = item.split('@')[0]
                location_set.add(loc)
        f.close()

    location2id = {location: idx for idx, location in enumerate(location_set)}
    user2id = {user: idx for idx, user in enumerate(user_set)}

    print('\n*** Please check the corresponding dataset configuration yml file. ***')
    print('unique location num:', len(location2id))
    print('unique user num:', len(user2id))

    yml_modified = get_user_input()
    if yml_modified:
        np.save(location_mapper_path, location2id)
        np.save(user_mapper_path, user2id)
    else:
        print('Program Exit')
        exit()


def run_test(dataloader, aux_mat, model_path, model, device, epoch, args, test_only=False):
    if test_only:
        saved_model_path = os.path.join(model_path, f'checkpoint_epoch{epoch}.pth')
        if not os.path.exists(saved_model_path):
            print(f"Warning: Model checkpoint not found at {saved_model_path}.")
            exit()
        else:
            print(f"Loading model for testing from {saved_model_path}")
            model.load_state_dict(torch.load(saved_model_path, map_location=device)['model_state_dict'])

    model.to(device)
    model.eval()
    precision_loc = 0
    top_k_values = [1, 3, 5, 10]
    top_k_correct_loc = np.array([0 for _ in range(len(top_k_values))])
    total_samples = 0

    def evaluate(output, label, ks):
        topk_correct_counts = [
            torch.sum(
                (torch.topk(output, k=top_k, dim=1)[1] + 0) == label.unsqueeze(1)
            ).item()
            for top_k in ks
        ]
        return np.array(topk_correct_counts)
    
    
    def calculate_mrr(output, true_labels):
        res = 0.0
        for i, pred in enumerate(output):
            sorted_indices = torch.argsort(pred, descending=True)
            true_index = np.where(true_labels[i].cpu() == sorted_indices.cpu())[0]
            if len(true_index) > 0:
                res += 1.0 / (true_index[0] + 1)
        return res

    total_time_samples = 0
    with torch.no_grad():
        for batch_data in tqdm.tqdm(dataloader):
            location_output, _ = model(batch_data, aux_mat)
            location_y = batch_data['location_y']
            location_y = location_y.view(-1)
            time_y =batch_data['timeslot_y']
            time_y = time_y.view(-1)
            total_samples += location_y.size(0)

            top_k_correct_loc += evaluate(location_output, location_y, top_k_values)
            precision_loc += calculate_mrr(location_output, location_y)
            total_time_samples += time_y.size(0)

    top_k_accuracy_loc = [count / total_samples * 100 for count in list(top_k_correct_loc)]


    result_str = "*********************** Test ***********************\n"
    result_str += f"base_dim: {args.base_dim}\n"
    result_str += f"Epoch {epoch + 1}: Total {total_samples} predictions on Next Location:\n"
    for k, accuracy in zip(top_k_values, top_k_accuracy_loc):
        result_str += f"Acc@{k}: {accuracy:.2f}\n"
    result_str += f"MRR: {precision_loc * 100 / total_samples:.2f}\n"
    result_save = top_k_accuracy_loc
    result_save.append(precision_loc * 100 / total_samples)
    result_save = np.array(result_save)

    print(result_str)
    top_k_accuracy_loc_return = top_k_correct_loc / total_samples * 100
    mrr_return = precision_loc * 100 / total_samples
    with open(os.path.join(model_path, 'results.txt'), 'a', encoding='utf8') as res_file:
        res_file.write(result_str + '\n\n')
    return top_k_accuracy_loc_return,mrr_return    


def save_checkpoint(save_dir, model, epoch):
    save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)


def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

