'''
utility functions
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

def normalize_duration(input, mask):
    input = torch.exp(input)*mask
    output = F.normalize(input, p=1, dim=-1)
    return output

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def eval_file(gt_content, recog_content, obs_percentage, classes):
    # github.com/yabufarha/anticipating-activities
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    key_is_string = isinstance(ground_truth[0], str)

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        class_idx = classes[ground_truth[i]] if key_is_string else ground_truth[i]
        if ground_truth[i] == recognized[i]:
            n_T[class_idx] += 1
        else:
            n_F[class_idx] += 1

    return n_T, n_F

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Apply label smoothing if needed'''

    loss = cal_loss(pred, gold.long(), trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
#    gold = gold.contiguous().view(-1)
    n_correct, n_word = cal_accuracy(pred, gold, trg_pad_idx)

    return loss, n_correct, n_word

def cal_accuracy(pred, gold, trg_pad_idx):
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Calculate cross entropy loss, apply label smoothing if needed'''

#    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class -1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx)
    return loss


### ACTION SEGMENATION METRICS ###
# Code taken from https://github.com/yabufarha/ms-tcn/blob/c1f537b18772564433445d63948b80a096a3529f/eval.py

def levenstein_distance(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col))
    else:
        score = D[-1, -1]

    return score

def edit_score(recognized, ground_truth, norm=True, ignored_classes=[-100]):
    P, _, _ = get_labels_start_end_time(recognized, ignored_classes)
    Y, _, _ = get_labels_start_end_time(ground_truth, ignored_classes)
    return levenstein_distance(P, Y, norm)


def f_score(recognized, ground_truth, overlap,  ignored_classes=[-100]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, ignored_classes)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, ignored_classes)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    f1_score = get_f1_score(tp, fp, fn)
    return f1_score



def get_f1_score(tp: float, fp: float, fn: float) -> float:
    if tp + fp != 0.0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    else:
        precision = 0.0
        recall = 0.0

    if precision + recall != 0.0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1

def get_labels_start_end_time(frame_wise_labels, ignored_classes=[-100]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in ignored_classes:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in ignored_classes:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in ignored_classes:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in ignored_classes:
        ends.append(i + 1)
    return labels, starts, ends

def get_matrix_future_multilabel(future_target: torch.Tensor, n_class: int, ignore_idxs: list):
    '''
    Get the next action matrix for the decoder to do a multi-class classification
    to know which action will happen next
    
    future_target: [B, T] tensor
    n_class: int, number of classes excluding the pad token and NONE token
    ignore_idxs: list of int, the indices to ignore


    Examples:
    input: [[1, 3, 2]]
    output: [[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]]

    '''
    B, T = future_target.size()
    matrix_future_target = torch.zeros(B, T, n_class) # +1 for NONE token

    for i in range(B):
        for j in range(T-1, -1, -1):
            class_idx = future_target[i, j].long()
            if class_idx in ignore_idxs:
                continue
            # Put all the row up to j to 1
            matrix_future_target[i, :j+1, class_idx] = 1

    return matrix_future_target

    

if __name__ == "__main__":
    print("Testing utility functions")
    tensor=torch.Tensor([[ 2.,  3., 11., 12., 13.,  0.,  1., 15.,  6.,  9., 19., 21., 21., 21.,21., 21., 21., 21., 21., 21.],
        [ 2.,  3.,  0.,  1., 15.,  6.,  9., 10.,  8.,  7., 10., 16., 18., 19.,21., 21., 21., 21., 21., 21.],
        [ 2.,  3.,  0.,  1., 11., 12., 13., 19., 21., 21., 21., 21., 21., 21.,21., 21., 21., 21., 21., 21.],
        [12.,  3., 13.,  0.,  1.,  4.,  5., 15.,  7.,  8., 19., 21., 21., 21.,21., 21., 21., 21., 21., 21.]]).long()
    # tensor = torch.Tensor([[1, 3, 2]]).long()
    matrix_future_target = get_matrix_future_multilabel(tensor, 20, [21])
    print(matrix_future_target[0])