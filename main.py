import torch
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from opts import parser
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from utils import read_mapping_dict
from data.basedataset import BaseDataset
from model.tcca import TCCA
from train import train
from predict import predict_multilabel, predict_long
import gc
gc.collect()

from tabulate import tabulate

def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    # Seed fix
    seed = args.seed
    seed_experiment(seed)

    args.hidden_dim_decoder = args.hidden_dim if args.hidden_dim_decoder == -1 else args.hidden_dim_decoder

    if str(args.gpu) == "-1" or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('using cpu')
    elif str(args.gpu) == "all":
        device = torch.device('cuda')
        print('using all gpus')
    else:
        device = torch.device(f'cuda:{str(args.gpu)}')
        # empty previous cache and memory
        torch.cuda.empty_cache()
        print('using gpu =>', device)
        
    print('Runs : ', args.runs)
    print('Model type : ', args.model)
    print('Epoch : ', args.epochs)
    print("Split : ", args.split)


    dataset = args.dataset
    split = args.split
    data_path = os.path.join('datasets', dataset)

    if args.crf_init_matrix: 
        crf_init_matrix_path = os.path.join(data_path, 'SR_matrix')
        args.crf_init_matrix_path = os.path.join(crf_init_matrix_path, f"{args.crf_init_matrix}_s{args.split}.npy")

    # print a title using # for ARGUMENTS so that is in the middle of 30 # 
    print("#" * 20 + " ARGUMENTS " + "#" * 20)
    # Print all args as table
    print(tabulate(vars(args).items(), headers=['arg', 'value']))
    print("#" * 60, end='\n\n')

    mapping_file = os.path.join(data_path, 'mapping.txt')
    actions_dict = read_mapping_dict(mapping_file)
    video_file_path = os.path.join(data_path, 'splits', 'train.split'+args.split+'.bundle' )
    video_file_test_path = os.path.join(data_path, 'splits', 'test.split'+args.split+'.bundle' )

    video_file = open(video_file_path, 'r')
    video_file_test = open(video_file_test_path, 'r')

    video_list = video_file.read().split('\n')[:-1]
    video_test_list = video_file_test.read().split('\n')[:-1]

    features_path = os.path.join(data_path, 'features')
    gt_path = os.path.join(data_path, 'groundTruth')

    if not args.model_path:
        if args.dataset == 'breakfast' :
            model_path = './ckpt/bf_split'+args.split+'_cpu.ckpt'
        elif args.dataset == '50salads':
            model_path = './ckpt/50s_split'+args.split+'_cpu.ckpt'
    else:
        model_path = args.model_path

    n_class = len(actions_dict) + 1 # 48 actions + 1 NONE token for breakfast
    args.n_class = n_class
    pad_idx = n_class + 1 # pad index. For Breakfast, 50

    print(f"Number of classes : {n_class} | {n_class-1} actions + 1 NONE token")
    print(f"None index : {n_class}")
    print(f"Pad index : {pad_idx}")

    # Model specification
    model = TCCA(n_class, args.hidden_dim, args.hidden_dim_decoder, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer)

    model_save_path = os.path.join('save_dir', args.dataset, 'runs'+str(args.runs), f'split{split}', 'models')
    results_save_path = os.path.join('save_dir', args.dataset, 'runs'+str(args.runs), f'split{split}', 'results')
    os.makedirs(results_save_path, exist_ok=True)

    if str(args.gpu) == "all":
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    if args.dataset == 'breakfast':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs, eta_min=args.lr/10, warmup_start_lr=10e-6)
    else:
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs, eta_min=args.lr/100, warmup_start_lr=1e-7)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, verbose=True)
    criterion = nn.MSELoss(reduction = 'none')
    predict = None
    if args.task == 'long':
        predict = predict_long
    elif args.task == 'multi-label':
        predict = predict_multilabel

    if args.predict :
        if args.task == 'long':
            obs_perc = [0.2, 0.3]
        elif args.task == 'multi-label':
            obs_perc = [0.25, 0.5, 0.75]

        print("Predict with ", model_path)


        model.load_state_dict(torch.load(model_path, map_location=device))
        if str(args.gpu) == "all":
            model = nn.DataParallel(model)
        model.to(device)
        
        # Save model in cpu
        if str(args.gpu) == "all":
            torch.save( model.module.module.state_dict(), model_path.replace('.ckpt', '_cpu.ckpt'))
            print("Model saved in ", model_path.replace('.ckpt', '_cpu.ckpt'))
        
        total_metrics = {}
        for obs_p in obs_perc :
            metrics = predict(model, video_test_list, args, obs_p, n_class, actions_dict, device, results_save_path)
            total_metrics[obs_p] = metrics

        # Print mMOF
        values = [metric for metrics in total_metrics.values() for metric in metrics.values()]
        print(f"[Split {args.split}] ==> mMOF: {np.mean(values):.4f}")

        return total_metrics
    else :
        # Training
        g = torch.Generator()
        g.manual_seed(seed)

        trainset = BaseDataset(video_list, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=trainset.my_collate,
                                                    worker_init_fn=seed_worker,
                                                    generator=g)
        
        predict_fn = lambda model, obs_p: predict(model, video_test_list, args, obs_p, n_class, actions_dict, device, results_save_path=None)
        # predict_fn_train = lambda model, obs_p: predict(model, video_list, args, obs_p, n_class, actions_dict, device, results_save_path=None)
        predict_fn_train = None
                                                    
        best_model = train(args, model, train_loader, optimizer, scheduler, criterion,
                     model_save_path, pad_idx, device, predict_fn=predict_fn, predict_fn_train=predict_fn_train)
        
        return best_model


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
