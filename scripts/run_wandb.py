import sys
import os

import torch
# Add the parent directory to the path so we can import the opts module
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from opts import parser
from main import main

import wandb
import yaml
import pprint

from ablation import ablation_dict

number_splits = {
    'breakfast': 4,
    '50salads': 5,
    'EGTEA': 1,
    'EPIC': 1
}

train_all_splits = False
suffix_sweep = ''
dataset = 'breakfast'

if __name__ == '__main__':

    # Log in to W&B
    wandb.login(key='<ADD YOUR KEY HERE>')
    
    # Load sweep configurations from YAML file
    with open(f'scripts/sweeps/sweep_{dataset}{suffix_sweep}.yaml') as file:
        sweep_list = yaml.load(file, Loader=yaml.FullLoader)
        pprint.pprint(sweep_list)

    # Define the W&B project
    project_name = f'[TCCA] {dataset}' if train_all_splits else f'[TCCA] {dataset} split 1'
    
    # Create a sweep and obtain the sweep ID
    sweep_id = wandb.sweep(sweep_list, project=project_name)

    def set_default():
        args = parser.parse_args()
        args.dataset = dataset
        args.gpu = "0"
        def breakfast():
            args.n_query = 8
            args.sample_rate = 3
        
        def salads():
            args.max_pos_len = 3100
            
        def egtea():
            args.max_pos_len = 3100
            
        def epic():
            args.max_pos_len = 4000

            
        args.input_type = 'i3d_transcript'
        args.runs = 'id'
        
        if dataset == 'breakfast':
            breakfast()
        elif dataset == '50salads':
            salads()
        elif dataset == 'EGTEA':
            egtea()
        elif dataset == 'EPIC':
            epic()
  
        # Model parameters
        args.seg = True
        args.anticipate = True

        return args
    

    def set_wandb_config(args, configs):
        for key, value in configs.items():
            setattr(args, key, value)


    def training():
        if train_all_splits:
            splits = number_splits[dataset] + 1
            print("Training on all splits")
        else:
            splits = 2
            print("Training only on split 1")
        with wandb.init() as f:
            run_id = str(wandb.run.id)
            print(f"Current run ID: {run_id}")
            args = set_default()
            configs = wandb.config
            print("Starting run")
            pprint.pprint(configs)
            set_wandb_config(args, configs)
            if args.ablation:
                # Apply the ablation configuration
                assert args.ablation in ablation_dict, f"Invalid ablation configuration: {args.ablation}"
                print(f"Using configuration from ablation {args.ablation}")
                set_wandb_config(args, ablation_dict[args.ablation])
            metrics_final = {}
            metrics_with_splits = {}
            torch.cuda.empty_cache() # Empty the cache to avoid memory issues
            args.runs += f'_{run_id}' 
            
            for split in range(1, splits):
                args.split = str(split)
                # Run the training
                args.predict = False
                args.model_path = None
                model_path = main(args)

                # Run the prediction
                args.predict = True
                args.model_path = model_path
                metrics_split = main(args)
                metrics_with_splits[f'split_{split}'] = metrics_split
                

            if args.task == 'long':
                # Compute the average metrics for each observation percentage
                # Create a summary table for doing a perfect cross-validation between splits+
                num_obs_p = list(list(metrics_with_splits.values())[0])
                num_pred_p = list(list(list(metrics_with_splits.values())[0].values())[0])
                for obs_p in num_obs_p:
                    for pred_p in num_pred_p:
                        metrics_obs_p = [metrics_with_splits[f'split_{split}'][obs_p][pred_p] for split in range(1,splits)]
                        metrics_final[f'obs_{obs_p}_pred_{pred_p}'] = sum(metrics_obs_p) / len(metrics_obs_p)

                mean_metrics_final = sum(metrics_final.values()) / len(metrics_final.values())

                metrics_final['MOF'] = mean_metrics_final

                print(f"Metric final --> mean MOF: {mean_metrics_final}")
            elif args.task == 'multi-label':
                # In this framework, we average the metrics so that the end result is the average of the splits
                # between the ALL, MANYSHOT and RARESHOT
                # Extract all splits, thresholds, and metrics
                splits = metrics_with_splits.keys()
                thresholds = list(list(metrics_with_splits.values())[0].keys())
                metrics = list(list(list(metrics_with_splits.values())[0].values())[0].keys())

                # Compute means using list comprehensions
                means = [
                    sum(metrics_with_splits[split][threshold][metric] for split in splits for threshold in thresholds) / (len(splits) * len(thresholds))
                    for metric in metrics
                ]
                metrics_final = {metric: mean for metric, mean in zip(metrics, means)}

                mean_metrics_final = sum(means) / len(means)
                metrics_final['M_mAP'] = mean_metrics_final
                

            # Log the metrics to W&B
            wandb.log(metrics_final)


    # Run the training using W&B sweep
    wandb.agent(sweep_id, training)

        