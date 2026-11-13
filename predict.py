import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import pdb
import os
import copy
from collections import defaultdict
import numpy as np
from utils import normalize_duration, eval_file, edit_score, f_score
from tabulate import tabulate
from torchnet.meter.apmeter import APMeter


# Predict for long task
def predict_long(model, vid_list, args, obs_p, n_class, actions_dict, device, results_save_path=None):
    model.eval()
    with torch.no_grad():
       
        data_path = os.path.join('datasets', args.dataset)
        gt_path = os.path.join(data_path, 'groundTruth')
        features_path = os.path.join(data_path, 'features')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class-1
        pad_idx = n_class + 1
        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))
        edit_score_future = np.zeros((len(eval_p)))
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        inverse_actions_dict_with_NONE = {v: k for k, v in actions_dict.items()}

        segmentation_metrics = np.zeros((len(vid_list), 5))

        for index_vid, vid in enumerate(vid_list):
            file_name = vid.split('/')[-1].split('.')[0]

            output_name = f"{file_name}_{obs_p}.npy"

            # load ground truth actions
            gt_file = os.path.join(gt_path, file_name+'.txt')
            gt_read = open(gt_file, 'r')
            gt_seq = gt_read.read().split('\n')[:-1]
            gt_read.close()             

            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len) # alpha * T
            future_len = int(pred_p*vid_len) # beta * T

            past_seq = gt_seq[:past_len] # past sequence 
            if args.input_type == 'i3d_transcript': # load features
                features_file = os.path.join(features_path, file_name+'.npy')
                features = np.load(features_file).transpose()
            elif args.input_type == 'gt':
                features = np.array([actions_dict_with_NONE[act] for act in gt_seq]).astype(int)
                features = np.expand_dims(features, axis=1)
            features = features[:past_len]
            inputs = features[::sample_rate, :] 
            inputs = torch.Tensor(inputs).to(device)
            inputs_future_len = torch.Tensor([future_len/1800]).long().to(device)

            outputs = model(inputs=(inputs.unsqueeze(0), inputs_future_len.unsqueeze(0)), mode='test')
            if args.crf:
                output_label = outputs['crf_action']
            else:
                output_action = outputs['action']
                output_label = output_action.max(-1)[1]
            output_dur = outputs['duration']



            # CHANGE to elif
            if args.seg:
                # measure the segmentation accuracy
                if args.model == 'tcca_FACT':
                    output_seg = outputs['seg_pred']
                else:
                    output_seg = outputs['seg']
                    if output_seg.dim() == 4: # this is case of multistage segmentation like LTContext
                        output_seg = output_seg[0, -1, :] # take the last stage
                    output_seg = output_seg.max(-1)[1]
                output_seg = output_seg.squeeze().cpu().numpy()

            if args.seg:
                past_seq_sampled = past_seq[::sample_rate] # sample the past sequence since the input of the model is sampled
                past_seq_sampled = np.array([actions_dict_with_NONE[act] for act in past_seq_sampled]) # Transform string to int in past_seq_sampled
                
                f1_10 = f_score(recognized=output_seg, ground_truth=past_seq_sampled, overlap=0.1, ignored_classes=[pad_idx])
                f1_25 = f_score(recognized=output_seg, ground_truth=past_seq_sampled, overlap=0.25, ignored_classes=[pad_idx])
                f1_50 = f_score(recognized=output_seg, ground_truth=past_seq_sampled, overlap=0.5, ignored_classes=[pad_idx])
                seg_nT, seg_nF = eval_file(past_seq_sampled, output_seg, 0, classes=actions_dict_with_NONE)
                mof = seg_nT.sum()/(seg_nT.sum()+seg_nF.sum())
                edit = edit_score(recognized=output_seg, ground_truth=past_seq_sampled, ignored_classes=[pad_idx])

                segmentation_metrics[index_vid] = [mof, edit, f1_10, f1_25, f1_50]


            # fine the forst none class
            none_mask = None
            none_idx = output_label.size(1) - 1
            for i in range(output_label.size(1)) :
                if output_label[0,i] == NONE :
                    none_idx = i
                    break
                else :
                    none = None
            # create a mask after the first none class
            if none_idx is not None :
                none_mask = torch.ones(output_label.shape).type(torch.bool)
                none_mask[0, none_idx:] = False

            output_dur = normalize_duration(output_dur, none_mask.to(device))

            pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

            pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
            predicted = torch.ones(future_len)
            action = output_label.squeeze()

            for i in range(len(action)) :
                predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                pred_len[i+1] = pred_len[i] + pred_len[i+1]
                if i == len(action) - 1 :
                    predicted[int(pred_len[i]):] = action[i]

            if args.seg:
                prediction = np.repeat(output_seg, sample_rate)[:len(past_seq)] # Our segmentation of the encoder
                prediction = np.array([inverse_actions_dict_with_NONE[act] for act in prediction]) # Transform int to string in prediction
            else:
                prediction = past_seq

            for i in range(len(predicted)):
                prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

            # save info (gt and pred)
            if results_save_path is not None:
                # Transform string to int in gt_seq
                save_gt_seq = [actions_dict_with_NONE[act] for act in gt_seq[:len(prediction)]]
                save_pred_seq = [actions_dict_with_NONE[act] for act in prediction]

                information_to_save = np.array([save_gt_seq, save_pred_seq])
                np.save(os.path.join(results_save_path, output_name), information_to_save)

            #evaluation
            for i in range(len(eval_p)):
                p = eval_p[i]
                # MOF
                eval_len = int((obs_p+p)*vid_len)
                eval_prediction = prediction[:eval_len]
                T_action, F_action = eval_file(gt_seq, eval_prediction, obs_p, actions_dict)
                T_actions[i] += T_action
                F_actions[i] += F_action

                # Edit score for the future
                past_len = int(obs_p*vid_len)
                eval_prediction_future = prediction[past_len:eval_len]
                gt_seq_future = gt_seq[past_len:eval_len]
                edit_score_future[i] += edit_score(recognized=eval_prediction_future, ground_truth=gt_seq_future, ignored_classes=[pad_idx])


        # Segmenation metrics for obs_p
        mean_mof, mean_edit, mean_f1_10, mean_f1_25, mean_f1_50 = np.mean(segmentation_metrics, axis=0)

        results = {}
        results_edit_future = {}
        for i in range(len(eval_p)):
            # MOF
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                if total_actions[i,j] != 0:
                    acc += float(T_actions[i,j]/total_actions[i,j])
                    n+=1

            result = float(acc)/n

            # Edit score for the future
            result_edit = edit_score_future[i]/len(vid_list)
            results_edit_future[eval_p[i]] = result_edit

            result_str = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])+'--> MoC: %.4f'%(result) + ' Edit: %.4f'%(result_edit)
            results[eval_p[i]] = result
            if results_save_path is not None:
                print(result_str)

        # print results_edit_future:
        #   print(f"Edit {obs_p}: {[f'{value:.4f}' for value in list(results_edit_future.values())]}")
        if results_save_path is not None:
            print('--------------------------------')        
            print(f"Segmentation | MOF: {mean_mof:.4f}, Edit: {mean_edit:.4f}, F1@10: {mean_f1_10:.4f}, F1@25: {mean_f1_25:.4f}, F1@50: {mean_f1_50:.4f}")
            print('--------------------------------')
            print(f"Results saved in {results_save_path}")

            # Save in the results save path the args using tabulate in a file called args.txt
            # create the args.txt file if it does not exist
            with open(os.path.join(results_save_path, 'args.txt'), 'w') as f:
                f.write(tabulate(vars(args).items(), headers=['arg', 'value']))

            # Save the results in a file called results.txt
            with open(os.path.join(results_save_path, f'results_{obs_p}.txt'), 'w') as f:
                f.write(f"Segmentation | MOF: {mean_mof:.4f}, Edit: {mean_edit:.4f}, F1@10: {mean_f1_10:.4f}, F1@25: {mean_f1_25:.4f}, F1@50: {mean_f1_50:.4f}\n")
                for key, value in results.items():
                    f.write(f"obs. {int(100*obs_p)} pred. {int(100*key)} --> MoC: {value:.4f}\n , Edit: {results_edit_future[key]:.4f}\n")

            if args.crf:
                model_module = model.module.module if args.gpu == 'all' else model
                # Save start_transitions, end_transitions, transitions in files from the crf
                start_transitions = model_module.crf.start_transitions.cpu().detach().numpy()
                end_transitions = model_module.crf.end_transitions.cpu().detach().numpy()
                transitions = model_module.crf.transitions.cpu().detach().numpy()

                np.save(os.path.join(results_save_path, 'start_transitions.npy'), start_transitions)
                np.save(os.path.join(results_save_path, 'end_transitions.npy'), end_transitions)
                np.save(os.path.join(results_save_path, 'transitions.npy'), transitions)
            

        return results


# Predict for multi-label task
def predict_multilabel(model, vid_list, args, obs_p, n_class, actions_dict, device, results_save_path=None):
    model.eval()
    with torch.no_grad():
        data_path = os.path.join('datasets', args.dataset)
        gt_path = os.path.join(data_path, 'groundTruth')
        features_path = os.path.join(data_path, 'features')

        many_shot_path = os.path.join(data_path, 'many_shot_verbs.csv')
        many_shot_verbs = []
        with open(many_shot_path, 'r') as f:
           lines = f.readlines()[1:]
           many_shot_verbs = [int(line.split(',')[0]) for line in lines]
        many_shot_verbs = sorted(many_shot_verbs)
        eval_verbs = set()

        pred_p = 1 - obs_p
        n_class_without_NONE = n_class - 1
        sample_rate = args.sample_rate
        NONE = n_class-1
        pad_idx = n_class + 1
        inverse_action_dict = {v: k for k, v in actions_dict.items()}
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE
        inverse_actions_dict_with_NONE = {v: k for k, v in actions_dict_with_NONE.items()}
        
        predictions_list = np.zeros((len(vid_list), 2, n_class_without_NONE))

        map_meter = APMeter()
        map_meter_aggregation_emission = APMeter()
        
        segmentation_metrics = np.zeros((len(vid_list), 5))
        
        for index_vid, vid in enumerate(vid_list):
            file_name = vid.split('/')[-1].split('.')[0]
            
            # load ground truth actions
            gt_file = os.path.join(gt_path, file_name+'.txt')
            gt_read = open(gt_file, 'r')
            gt_seq = gt_read.read().split('\n')[:-1]
            gt_read.close()    
            
            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len) # alpha * T
            future_len = int(pred_p*vid_len) # beta * T

            past_seq = gt_seq[:past_len] # past sequence 

            future_seq = gt_seq[past_len:] # future sequence
            future_seq_set = set([actions_dict_with_NONE[action] for action in set(future_seq)])
            
            eval_verbs |= future_seq_set

            if args.input_type == 'i3d_transcript': # load features
                features_file = os.path.join(features_path, file_name+'.npy')
                features = np.load(features_file).transpose()
            elif args.input_type == 'gt':
                features = np.array([actions_dict_with_NONE[act] for act in gt_seq]).astype(int)
                features = np.expand_dims(features, axis=1)
            features = features[:past_len]
            inputs = features[::sample_rate, :] 
            inputs = torch.Tensor(inputs).to(device)
            inputs_future_len = torch.Tensor([future_len/1800]).long().to(device)
            outputs = model(inputs=(inputs.unsqueeze(0), inputs_future_len.unsqueeze(0)), mode='test')
            
            if args.seg:
                # Compute the segmentation metrics
                # measure the segmentation accuracy
                if args.model == 'tcca_FACT':
                    output_seg = outputs['seg_pred']
                else:
                    output_seg = outputs['seg']
                    if output_seg.dim() == 4: # this is case of multistage segmentation like LTContext
                        output_seg = output_seg[0, -1, :] # take the last stage
                    output_seg = output_seg.max(-1)[1]
                output_seg = output_seg.squeeze().cpu().numpy()
                
                past_seq_sampled = past_seq[::sample_rate] # sample the past sequence since the input of the model is sampled
                past_seq_sampled = np.array([actions_dict_with_NONE[act] for act in past_seq_sampled]) # Transform string to int in past_seq_sampled
                
                f1_10 = f_score(recognized=output_seg, ground_truth=past_seq_sampled, overlap=0.1, ignored_classes=[pad_idx])
                f1_25 = f_score(recognized=output_seg, ground_truth=past_seq_sampled, overlap=0.25, ignored_classes=[pad_idx])
                f1_50 = f_score(recognized=output_seg, ground_truth=past_seq_sampled, overlap=0.5, ignored_classes=[pad_idx])
                seg_nT, seg_nF = eval_file(past_seq_sampled, output_seg, 0, classes=actions_dict_with_NONE)
                mof = seg_nT.sum()/(seg_nT.sum()+seg_nF.sum())
                edit = edit_score(recognized=output_seg, ground_truth=past_seq_sampled, ignored_classes=[pad_idx])

                segmentation_metrics[index_vid] = [mof, edit, f1_10, f1_25, f1_50]
         
            output_matrix = outputs["anticipate_future_multilabel"] # (1, T, n_class)
            output_matrix = torch.sigmoid(output_matrix)
            output_matrix = output_matrix[0, 0, :].detach().cpu().numpy()
            y_scores_pred = output_matrix
            
            output_action = outputs['action'] # (1, T, n_class)
            # Apply softmax to the output
            output_action = F.softmax(output_action, dim=-1)
            # Sum everything in the time dimension and remove the NONE class
            output_action_multilabel = output_action.sum(1)[:, :-1]
            # Clip the values to 0 and 1
            output_action_multilabel = torch.clamp(output_action_multilabel, 0, 1)
            output_action_multilabel = output_action_multilabel.squeeze().detach().cpu().numpy()

            # Transform the set in to a list with 1 if the action is in the future and 0 otherwise. The size
            # of the list is the number of classes without the NONE class
            y_true = np.array([1 if inverse_action_dict[action] in set(future_seq) else 0 for action in inverse_action_dict.keys()])

            predictions_list[index_vid] = [y_scores_pred, y_true]
            map_meter.add(np.expand_dims(y_scores_pred, 0), np.expand_dims(y_true, 0))
            map_meter_aggregation_emission.add(np.expand_dims(output_action_multilabel, 0), np.expand_dims(y_true, 0))


        # Segmenation metrics for obs_p
        mean_mof, mean_edit, mean_f1_10, mean_f1_25, mean_f1_50 = np.mean(segmentation_metrics, axis=0)

        # Compute the mAP for the different sets. This code is taken from the original EGO-TOPO
        # https://github.com/facebookresearch/ego-topo/blob/81b5396e2a6ecc44f4d541d48cf4777670e36e42/anticipation/anticipation/datasets/epic_future_labels.py#L96
        # Create eval_set with 1 if the class is in the eval_verbs set and 0 otherwise
        # There are 3 categories:
        # 0: All verbs
        # 1: Many-shot verbs
        # 2: Rare-shot verbs
        eval_set = np.zeros((3, n_class_without_NONE))
        eval_set[0, list(eval_verbs)] = 1
        manyshot = eval_verbs & set(many_shot_verbs)
        rareshot = eval_verbs - set(many_shot_verbs)
        eval_set[1, list(manyshot)] = 1
        eval_set[2, list(rareshot)] = 1

        eval_set = eval_set.astype(bool)

        AP_per_class = map_meter.value().numpy()
        mAP = np.zeros(3)
        for i in range(3):
            mAP[i] = np.mean(AP_per_class[eval_set[i]])

        AP_per_class_aggregation_emission = map_meter_aggregation_emission.value().numpy()
        mAP_aggregation_emission = np.zeros(3)
        for i in range(3):
            mAP_aggregation_emission[i] = np.mean(AP_per_class_aggregation_emission[eval_set[i]])
 
        
        if results_save_path is not None:
            print(f"\nobs p. {obs_p} |  mAP_all: {mAP[0]:.4f}, mAP_many_shot: {mAP[1]:.4f}, mAP_rare_shot: {mAP[2]:.4f}")
            print('--------------------------------')
            print(f"Segmentation | MOF: {mean_mof:.4f}, Edit: {mean_edit:.4f}, F1@10: {mean_f1_10:.4f}, F1@25: {mean_f1_25:.4f}, F1@50: {mean_f1_50:.4f}")
            print('--------------------------------')
            print(f"Results saved in {results_save_path}")

            # Save in the results save path the args using tabulate in a file called args.txt
            # create the args.txt file if it does not exist
            with open(os.path.join(results_save_path, 'args.txt'), 'w') as f:
                f.write(tabulate(vars(args).items(), headers=['arg', 'value']))

            # Save the results in a file called results.txt
            with open(os.path.join(results_save_path, f'results_{obs_p}.txt'), 'w') as f:
                f.write(f"Segmentation | MOF: {mean_mof:.4f}, Edit: {mean_edit:.4f}, F1@10: {mean_f1_10:.4f}, F1@25: {mean_f1_25:.4f}, F1@50: {mean_f1_50:.4f}\n")
                for key, value in zip(['all', 'many_shot', 'rare_shot'], mAP):
                    f.write(f"mAP_{key}: {value:.4f}\n")

        
        return {"mAP_all": mAP[0], "mAP_many_shot": mAP[1], "mAP_rare_shot": mAP[2]}
