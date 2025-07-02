import argparse
import os
import time

import torch
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import  MultiStepLR
from torch.utils.data import DataLoader


from dataloader import MyDataset, calculate_matrix
from framework import MyModel
from tools import get_config, run_test, get_mapper, custom_collate, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="6")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--dataset', type=str, default='TC')
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--base_dim', type=int, default=48, help='must be a multiple of 4')
parser.add_argument('--prefer_dim', type=int, default=64)
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--epoch', type=int, default=15, help='epoch num')
parser.add_argument('--test_epoch', type=int, default=None, help='test epoch num')
parser.add_argument('--min_test_epoch', type=int, default=1)
parser.add_argument('--num_prototypes', type=int, default=15)
parser.add_argument('--num_timeslots', type=int, default=24)
parser.add_argument("--test", action='store_true', default=False)
args = parser.parse_args()



gpu_list = args.gpu
set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
dataset_path = f'./data/{args.dataset}'
timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
dataset_image_path = f'./data/{args.dataset}/image'
save_path = args.dataset

save_dir = f"./saved_models/{save_path}"
config_path = f"{save_dir}/settings.yml"
device = torch.device("cuda")
test_epoch = args.test_epoch
if test_epoch is None:
    test_epoch = args.epoch

test_only = args.test

if __name__ == '__main__':
    get_mapper(dataset_path=dataset_path)
    config = get_config(config_path, easy=True)
    aux_mat = calculate_matrix(config, num_timeslots=args.num_timeslots, data_path=dataset_path)
    for mat in aux_mat.keys():
        aux_mat[mat] = torch.from_numpy(aux_mat[mat]).float().to(device)

    dataset = MyDataset(dataset_path=dataset_path, device=device, load_mode='train', args=args)
    batch_size = args.batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=lambda batch: custom_collate(batch, device))

    test_dataset = MyDataset(dataset_path=dataset_path, device=device, load_mode='test', args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 drop_last=True, collate_fn=lambda batch: custom_collate(batch, device))
    model = MyModel(config, args)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total training samples: {len(dataloader) * batch_size} | Total trainable parameters: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Dataset: {args.dataset} | Device: {device}")
    print(f"dim: {args.base_dim}")

    if test_only:
        save_dir = f'./saved_models/{save_path}'
        top_k_acc, mrr = run_test(test_dataloader, aux_mat, model_path=save_dir, model=model, device=device, epoch=args.epoch, args=args, test_only=test_only)
        exit()

    best_val_loss = float("inf")
    start_time = time.time()
    num_epochs = args.epoch
    scheduler = MultiStepLR(
        optimizer,
        milestones=[6, 8, 10, 12, 14, 16, 18],
        gamma=0.5,
    )
    epoch_losses = []
    epoch_acc1 = []
    epoch_acc10 = []
    epoch_mrr = []
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "report.txt"), "w") as report_file:
        print('Train batches:', len(dataloader))
        print('Test batches:', len(test_dataloader))
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss_epoch = 0.0
            epoch_str = f"================= Epoch [{epoch + 1}/{num_epochs}]| {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | LR: {optimizer.param_groups[0]['lr']}=================\n"
            for batch_data in dataloader:
                location_output, time_output = model(batch_data, aux_mat)
                location_y = batch_data['location_y'].view(-1)
                time_y = batch_data['timeslot_y'].view(-1)
                location_loss = loss_fn(location_output, location_y)

                balance_alpha = args.alpha
                lambda_ortho = 0
                total_loss = balance_alpha * location_loss.sum()
  
                time_loss = loss_fn(time_output, time_y)
                total_loss += (1-balance_alpha) * time_loss.sum()
                if args.num_prototypes > 0:
                    prototype_vectors = model.embedding_layer.prototypes_embedding.weight


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                total_loss_epoch += total_loss.item()

            average_loss = total_loss_epoch / len(dataloader)
            epoch_losses.append(average_loss)

            if average_loss <= best_val_loss:
                epoch_str += f"Best Loss: {best_val_loss:.6f} ---> {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"
                best_val_loss = average_loss
            else:
                epoch_str += f"Best Loss: {best_val_loss:.6f} | Epoch Loss: {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"

            report_file.write(epoch_str + '\n\n')
            report_file.flush()
            print(epoch_str)
            if (epoch+1) % test_epoch == 0 and (epoch+1) >= args.min_test_epoch:
                top_k_acc, mrr = run_test(test_dataloader, aux_mat, model_path=save_dir, model=model, device=device, epoch=epoch, args=args)
                epoch_acc1.append(top_k_acc[0])
                epoch_acc10.append(top_k_acc[3])
                epoch_mrr.append(mrr)
            scheduler.step()
        save_checkpoint(save_dir, model, epoch+1)

    end_time = time.time()
    total_time = end_time - start_time

    with open(os.path.join(save_dir, "report.txt"), "a") as report_file:
        report_file.write(f"Total Running Time: {total_time:.2f} seconds\n")
    print(f"\nModel done.\n")
