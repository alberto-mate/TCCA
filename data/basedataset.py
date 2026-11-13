import torch
import numpy as np
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from numpy.random import randint
from utils import *
import pdb
import random


class BaseDataset(Dataset):
    def __init__(self, vid_list, actions_dict, features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
        self.gt_path = gt_path
        self.mode = mode
        self.sample_rate = args.sample_rate
        self.vid_list = list()
        self.n_query = n_query
        self.args = args
        self.NONE = self.n_class - 1

        if self.mode == 'train' or self.mode == 'val':
            if args.task == 'long':
                print("Long-term task")
                if args.dataset == '50salads':
                    obs_perc_list = [0.2, 0.3, 0.4, 0.5]
                else:
                    obs_perc_list = [0.2, 0.3, 0.5]
                pred_perc_list = [0.5]
                for vid in vid_list:
                    for obs_perc in obs_perc_list:
                        for pred_prec in pred_perc_list:
                            if obs_perc + pred_prec > 1:
                                continue
                            self.vid_list.append([vid, obs_perc, pred_prec])
            elif args.task == 'multi-label':
                print("Multi-label task")
                obs_perc_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                for vid in vid_list:
                    for obs_perc in obs_perc_list:
                        self.vid_list.append([vid, obs_perc, 1-obs_perc])
        

        print(f"Dataset loaded. Total {len(self.vid_list)} videos")


    def __getitem__(self, idx):
        vid_file, obs_perc, pred_perc  = self.vid_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(vid_file, obs_perc, pred_perc)
        return item


    def _make_input(self, vid_file, obs_perc, pred_perc=0.5 ):
        vid_file = vid_file.split('/')[-1]

        gt_file = os.path.join(self.gt_path, vid_file)
        feature_file = os.path.join(self.features_path, vid_file.split('.')[0]+'.npy')
        features = np.load(feature_file)
        features = features.transpose()

        file_ptr = open(gt_file, 'r')
        all_content = file_ptr.read().split('\n')[:-1]
        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(pred_perc*vid_len)

        if self.args.random_start_frame:
            max_start_frame = vid_len - observed_len - pred_len
            if max_start_frame <= 0:
                start_frame = 0
            else:
                start_frame = randint(0, max_start_frame)
        else:
            start_frame = 0


        # feature slicing
        features = features[start_frame : start_frame + observed_len] #[S, C]
        if not self.args.random_sample_frame:
            features = features[::self.sample_rate]
        else:
            # Generate random index for sampling. If sample_rate is 3, then the divide the total features in equaled bin and sample from each bin 
            # using a uniform distribution
            n_bins = len(features)//self.sample_rate
            
            indices = []
            for i in range(n_bins):
                indices.append(randint(i*self.sample_rate, (i+1)*self.sample_rate))
            features = features[indices]

        past_content = all_content[start_frame : start_frame + observed_len] #[S]

        if not self.args.random_sample_frame:
            past_content = past_content[::self.sample_rate]
        else:
            past_content = np.array(past_content)[indices]
        past_label = self.seq2idx(past_content)


        if np.shape(features)[0] != len(past_content) :
            features = features[:len(past_content),]

        future_content = \
        all_content[start_frame + observed_len: start_frame + observed_len + pred_len] #[T]

        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)
        trans_future_target = trans_future #target
        
        # number_future_actions = len(trans_future)
        # print(vid_file, f"observed_len : {observed_len} | pred_len : {pred_len} | start_frame : {start_frame} | sample_rate : {self.sample_rate}")
        # print(f"Number of future actions : {number_future_actions}")
        # print(trans_future)


        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0 :
            tmp = np.ones(diff)*self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff+1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0 :
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else :
            tmp_len = np.ones(1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))


        item = {'features':torch.Tensor(features),
                'past_label':torch.Tensor(past_label),
                'trans_future_dur':torch.Tensor(trans_future_dur),
                'trans_future_target' : torch.Tensor(trans_future_target),
                'future_len': torch.Tensor([pred_len/1800]), # Transformed to minutes 30frames/s * 60s/min
                }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]
        b_future_len = [item['future_len'] for item in batch]

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)
        b_future_len = torch.cat(b_future_len, dim=0)

        batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target, b_future_len]

        return batch


    def __len__(self):
        return len(self.vid_list)

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i]]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i]
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)








