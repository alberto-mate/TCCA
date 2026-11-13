import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils import cal_performance, normalize_duration, get_matrix_future_multilabel, cal_accuracy
from model.extras.losses import TruncatedMeanSquareLoss
from copy import deepcopy
from einops import rearrange
from tqdm import tqdm
import time


def train(args, model, train_loader, optimizer, scheduler, dur_criterion, model_save_path, pad_idx, device, predict_fn=None, predict_fn_train=None):
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Print the number of parameters formated with commas
    print(f"==> Number of parameters: {n_params:,}")



    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    tsme_criterion = TruncatedMeanSquareLoss()
    print("Training Start")
    best_model_dict = model.state_dict()
    best_loss = 1e10
    best_epoch = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        epoch_loss_future_matrix = 0
        epoch_loss_anticipate_next = 0
        epoch_loss_anticipate_prev = 0
        epoch_loss_crf = 0

        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0

        epoch_time = time.time()
        
        # for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target, future_len = data
            features = features.to(device)  # [B, S, C]
            past_label = past_label.to(device)  # [B, S]
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)
            future_len = future_len.to(device)

            if args.anticipate_future_multilabel:
                matrix_future_target = get_matrix_future_multilabel(trans_future_target, args.n_class-1, [pad_idx, args.n_class-1]).to(device)

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future * trans_dur_future_mask
            target = trans_future_target

            if args.input_type == "i3d_transcript":
                inputs = (features, future_len, past_label, target)
            elif args.input_type == "gt":
                gt_features = past_label.int()
                inputs = (gt_features, future_len, past_label, target)

            outputs = model(inputs)
            losses = 0
            if args.seg:
            
                if args.model == "tcca_FACT":
                    loss_seg = outputs["seg_loss"]
                    losses += loss_seg
                    epoch_loss_seg += loss_seg.item()
                    
                    pred_seg = outputs["seg_pred"]
                    n_seg_correct, n_seg_total = cal_accuracy(pred_seg, past_label, pad_idx)
                    total_seg += n_seg_total
                    total_seg_correct += n_seg_correct
                    
                    output_seg = outputs["seg_probs"]
                    B, T, C = output_seg.size()
                    output_seg = output_seg.reshape(-1, C).to(device)  # [B*T, C]

                else:
                    output_seg_dict = outputs["seg"]
                    if output_seg_dict.dim() == 4: # B, N, T, C
                        num_stages = output_seg_dict.size(1) 
                    else: # B, T, C
                        num_stages = 1
                    
                    for stage in range(num_stages):
                        if output_seg_dict.dim() == 4:
                            output_seg = output_seg_dict[:, stage, :, :] # B, T, C
                        else:
                            output_seg = output_seg_dict

                        B, T, C = output_seg.size()
                        output_seg = output_seg.reshape(-1, C).to(device)  # [B*T, C]
                        target_past_label = past_label.view(-1)

                        loss_seg_ce, n_seg_correct, n_seg_total = cal_performance(output_seg, target_past_label, pad_idx)
                        if args.segmentation_loss == "cross_entropy":
                            loss_seg = loss_seg_ce
                        elif args.segmentation_loss == "cross_entropy+tmse":
                            mask = (past_label != pad_idx).long()
                            loss_seg_tmse = tsme_criterion(output_seg.reshape(B, T, C), mask)
                            loss_seg = loss_seg_ce + args.tsme_seg_weight * loss_seg_tmse

                        losses += loss_seg/num_stages
                        epoch_loss_seg += loss_seg.item()/num_stages
                        total_seg += n_seg_total
                        total_seg_correct += n_seg_correct

                last_output_seg = output_seg.reshape(B, T, C)
                # The last_output_seg_index is the index of the last element of the output_seg per batch that is not pad_idx. Shape: [B]
                last_output_seg_index = torch.sum((past_label != pad_idx).long(), dim=1) - 1
                # Get the last element of the output_seg per batch that is not pad_idx. Shape: [B, C]
                last_output_seg = last_output_seg[torch.arange(B), last_output_seg_index]
                # Add a -inf at the end to include the None action 
                last_output_seg = torch.cat((last_output_seg, torch.full((B, 1), -float("inf")).to(device)), dim=1)
                    
                

            if args.anticipate:
                if not args.crf: 
                    output = outputs["action"]
                    B, T, C = output.size()
                    output = output.view(-1, C).to(device)
                    target = target.contiguous().view(-1)
                    out = output.max(1)[1]  # oneshot
                    out = out.view(B, -1)
                    loss, n_correct, n_total = cal_performance(output, target, pad_idx)
                    acc = n_correct / n_total
                    loss_class = loss.item()
                    losses += loss
                    total_class += n_total
                    total_class_correct += n_correct
                    epoch_loss_class += loss_class
                else: # If crf is used, the output is a sequence and the loss is calculated inside the forward
                    crf_llh = outputs["crf_llh"]
                    crf_mask = outputs["crf_mask"]
                    loss_crf = -(crf_llh.sum() / crf_mask.float().sum()) # Apply loss as token_mean
                    losses += loss_crf
                    epoch_loss_crf += loss_crf.item()

                if args.task == "long":
                    output_dur = outputs["duration"]
                    output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                    target_dur = target_dur * trans_dur_future_mask
                    loss_dur = torch.sum(dur_criterion(output_dur, target_dur)) / torch.sum(
                        trans_dur_future_mask
                    )
                    losses += loss_dur
                    epoch_loss_dur += loss_dur.item()

            if args.anticipate_next:
                output_next = outputs["action_next"] # [B, T, C]
                output = outputs["action"]

                # Remove last element of output_next 
                output_next = output_next[:, :-1, :]
                # Remove first element of output
                output = output[:, 1:, :]

                B, T, C = output_next.size()
                output_next = output_next.reshape(-1, C).to(device)
                output = output.reshape(-1, C).to(device)

                # Apply KL divergence loss
                loss_anticipate_next = kl_criterion(F.log_softmax(output_next, dim=1), F.softmax(output, dim=1))
                
                losses += loss_anticipate_next
                epoch_loss_anticipate_next += loss_anticipate_next.item()

            if args.anticipate_prev:
                output_prev = outputs["action_prev"]
                output = outputs["action"]
                if not args.seg:
                    # Remove first element of output_prev
                    output_prev = output_prev[:, 1:, :]
                    # Remove last element of output and add last_output_seg at the beginning
                    # This way the decoder will be consistent with the last action of the past
                    output = output[:, :-1, :]
                else:
                    output = torch.cat((last_output_seg.unsqueeze(1), output[:, :-1, :]), dim=1)

                B, T, C = output_prev.size()
                output_prev = output_prev.reshape(-1, C).to(device)
                output = output.reshape(-1, C).to(device)

                # Apply KL divergence loss
                loss_anticipate_prev = kl_criterion(F.log_softmax(output_prev, dim=1), F.softmax(output, dim=1))

                losses += loss_anticipate_prev
                epoch_loss_anticipate_prev += loss_anticipate_prev.item()

            if args.anticipate_future_multilabel:
                output_future_matrix = outputs["anticipate_future_multilabel"]
                B, T, C = output_future_matrix.size()
                # Apply masking to turn all the padded values to 0
                
                output_future_matrix = output_future_matrix.view(-1, C).to(device)
                target_future_matrix = matrix_future_target.view(-1, C).to(device)

                # Apply sigmoid in the C dimension since it is a multilabel classification
                output_future_matrix = torch.sigmoid(output_future_matrix)
                extended_trans_dur_future_mask = trans_dur_future_mask.unsqueeze(2).expand(-1, -1, C).view(-1, C)
                output_future_matrix = output_future_matrix * extended_trans_dur_future_mask

                # Apply cross entropy binary loss
                loss_future_matrix = F.binary_cross_entropy(output_future_matrix, target_future_matrix)
                losses += loss_future_matrix
                epoch_loss_future_matrix += loss_future_matrix.item()

            epoch_loss += losses.item()
            losses.backward()
            # Apply gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

        epoch_time = time.time() - epoch_time
        epoch_loss = epoch_loss / (i+1)
        print()
        print(f"[{epoch_time:.2f}s] | Epoch [", (epoch+1), '/', args.epochs,'] ==> Loss: %.3f'%epoch_loss, end=' ')
        if args.anticipate :
            if not args.crf:
                accuracy = total_class_correct/total_class
                epoch_loss_class = epoch_loss_class / (i+1)
                print('| Training_Acc:%.3f'%accuracy, f'({total_class_correct}/{total_class}) | CE_loss:%.3f'%epoch_loss_class, end=' ')
            else:
                epoch_loss_crf = epoch_loss_crf / (i+1)
                print('| crf_loss:%.3f'%epoch_loss_crf, end=' ')
            if args.task == 'long' :
                epoch_loss_dur = epoch_loss_dur / (i+1)
                print('| dur_loss: %.5f'%epoch_loss_dur, end=' ')
        
        if args.anticipate_next:
            epoch_loss_anticipate_next = epoch_loss_anticipate_next / (i+1)
            print('| ant_next_loss:%.3f'%epoch_loss_anticipate_next, end=' ')

        if args.anticipate_prev:
            epoch_loss_anticipate_prev = epoch_loss_anticipate_prev / (i+1)
            print('| ant_prev_loss:%.3f'%epoch_loss_anticipate_prev, end=' ')

        if args.seg:
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('| seg_loss:%.3f'%epoch_loss_seg, end=' ')
            acc_seg = total_seg_correct / total_seg
            print('| seg_acc: %.5f'%acc_seg, f'({total_seg_correct}/{total_seg})' , end=' ')

        if args.anticipate_future_multilabel:
            epoch_loss_future_matrix = epoch_loss_future_matrix / (i+1)
            print('| matrix_future_loss:%.3f'%epoch_loss_future_matrix, end=' ')
        

            

        print()
        
        scheduler.step()
        model.eval()
        if args.task == 'long':
            eval_obs_p = [0.2, 0.3]
        elif args.task == 'multi-label':
            eval_obs_p = [0.25, 0.5, 0.75]
        if epoch >= args.epochs - 15:
            # Call the predict function
            if predict_fn is not None:
                metrics = []
                for obs_p in eval_obs_p:
                    temp = predict_fn(model, obs_p)
                    metrics.append(list(temp.values()))
                metrics = np.array(metrics)
                if args.task == 'long':
                    show_metrics = metrics.flatten()
                elif args.task == 'multi-label':
                    # Make the mean over the axis 0
                    show_metrics = np.mean(metrics, axis=0)
                mean_metrics = np.mean(metrics)
                print(f"[Predict] Mean Metrics: {mean_metrics:.4f} \t\t\t\t\t|| Metrics: {[round(metric, 4) for metric in show_metrics]}")
                epoch_loss = -mean_metrics

            if predict_fn_train is not None:
                metrics = []
                for obs_p in eval_obs_p:
                    temp = predict_fn_train(model, obs_p)
                    metrics.extend(list(temp.values()))
                mean_metrics = np.mean(np.array(metrics), axis=0)
                print(f"[Predict Train] Mean Metrics: {mean_metrics:.4f} \t\t\t\t\t|| Metrics: {[round(metric, 4) for metric in metrics]}")

            if epoch_loss < best_loss:
                print(f"[Predict] Best model updated at epoch {epoch} with value {abs(epoch_loss):.4f}")
                best_loss = epoch_loss
                best_model_dict = deepcopy(model.state_dict())
                best_epoch = epoch

            # Save query_embed and pos_embedding from the model
            # pos_embed = model.pos_embedding
            # torch.save(pos_embed, os.path.join(model_save_path, f"pos_embed_{str(epoch).zfill(3)}.ckpt"))
            # query_embed = model.query_embed.weight
            # torch.save(query_embed, os.path.join(model_save_path, f"query_embed_{str(epoch).zfill(3)}.ckpt"))
            # print(f"Query and Pos embeddings saved at epoch {epoch}")
            

        model.train()

    save_path = os.path.join(model_save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, "checkpoint" + str(best_epoch) + ".ckpt")
    torch.save(best_model_dict, save_file)
    print(f"Best model at {best_epoch} epoch saved at ", save_file)

    return save_file
