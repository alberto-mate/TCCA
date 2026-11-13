import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb
from einops import repeat, rearrange
from model.extras.transformer import Transformer, Transformer_LTContext, Transformer_FACT
from model.extras.position import PositionalEncoding
from model.CRF.crf import CRF
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class TCCA(nn.Module):
    """Temporal Context Consistency Above All (TCCA) architecture for Long-Term Action Anticipation."""

    def __init__(self, n_class, hidden_dim, hidden_dim_decoder, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.hidden_dim = hidden_dim
        self.hidden_dim_decoder = hidden_dim_decoder
        self.input_embed = nn.Linear(args.input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_embed.weight)
        if args.model in ['tcca_LTC', 'tcca_FACT']: 
            self.input_embed = nn.Identity() # LTContext has its own input projection embedd
            
        if args.model == 'tcca': 
            self.transformer = Transformer(hidden_dim, hidden_dim_decoder, n_head, num_encoder_layers, num_decoder_layers,
                                        4, normalize_before=False, dropout=args.dropout, activation=args.activation, args=args)

        elif args.model == 'tcca_LTC':
            self.transformer = Transformer_LTContext(n_class-1, args.input_dim, hidden_dim_decoder, n_head, num_decoder_layers,
                                        4, normalize_before=False, dropout=args.dropout, activation=args.activation, args=args)
            
        elif args.model == 'tcca_FACT':
            self.transformer = Transformer_FACT(n_class, args.input_dim, hidden_dim_decoder, n_head, num_decoder_layers,
                                        4, normalize_before=False, dropout=args.dropout, activation=args.activation)

        self.n_query = n_query
        self.args = args
        # self.query_embed = nn.Parameter(torch.zeros(self.n_query, hidden_dim_decoder))
        # nn.init.xavier_uniform_(self.query_embed)
        # Initialize using sinoisoidal
        # sino = PositionalEncoding(hidden_dim_decoder, dropout=args.dropout, max_len=self.n_query)
        # convert sino.pos_table into nn.Parameter
        # self.query_embed = nn.Parameter(sino.pos_table.squeeze(0))
        # self.query_embed.requires_grad = True



        self.query_embed = nn.Embedding(self.n_query, hidden_dim_decoder)





        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class-1) #except SOS, EOS
            nn.init.xavier_uniform_(self.fc_seg.weight)

        if args.anticipate :
            self.fc = nn.Linear(hidden_dim_decoder, n_class)
            nn.init.xavier_uniform_(self.fc.weight)

            if args.task == 'long':
                if args.duration == 'independent':
                    self.fc_len = nn.Linear(hidden_dim_decoder, 1)
                elif args.duration == 'action_partially_dependent':
                    self.fc_len = nn.Linear(hidden_dim_decoder+n_class, 1)
                nn.init.xavier_uniform_(self.fc_len.weight)

        if args.crf:
            self.crf = CRF(n_class, batch_first=True, transition_weight=args.crf_transition_weight) # Include EOS
            # initialize transition matrix
            
            if hasattr(args, 'crf_init_matrix_path'):
                try:
                    transitions = np.load(args.crf_init_matrix_path)
                    shifted_transitions = transitions - np.min(transitions, axis=1, keepdims=True)
                    transitions = shifted_transitions / shifted_transitions.sum(axis=1, keepdims=True)
                    transitions = np.nan_to_num(transitions)
                    self.crf.set_transitions(transitions)
                    print("CRF transition matrix loaded from ", args.crf_init_matrix_path)
                except:
                    raise ValueError("CRF transition matrix not initialized properly")
            

        if args.anticipate_next:
            # Predict the action distribution at the next time step
            self.anticipate_next = nn.Linear(hidden_dim_decoder, n_class)
            nn.init.xavier_uniform_(self.anticipate_next.weight)

        if args.anticipate_prev:
            # Predict the action distribution at the previous time step
            self.anticipate_prev = nn.Linear(hidden_dim_decoder, n_class)
            nn.init.xavier_uniform_(self.anticipate_prev.weight)

        if args.anticipate_future_multilabel:
            self.fc_matrix_future = nn.Linear(hidden_dim_decoder, n_class-1) # except SOS, EOS
            nn.init.xavier_uniform_(self.fc_matrix_future.weight)

        if args.pos_emb != 'none':
            #pos embedding
            max_seq_len = args.max_pos_len
            if args.pos_emb == 'learned':
                self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, self.transformer.d_encoder))
                nn.init.xavier_uniform_(self.pos_embedding)
            elif args.pos_emb == 'sinusoidal':
                # Sinusoidal position encoding
                self.pos_embedding = PositionalEncoding(self.transformer.d_encoder, dropout=args.dropout, max_len=max_seq_len)

        if args.input_type =='gt':
            self.gt_emb = nn.Embedding(n_class+2, self.hidden_dim, padding_idx=n_class+1)
            nn.init.xavier_uniform_(self.gt_emb.weight)


    def forward(self, inputs, mode='train'):
        if mode == 'train' :
            
            src, future_len, src_label, future_target = inputs
            tgt_key_padding_mask = None
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
            memory_key_padding_mask = src_key_padding_mask.clone().to(self.device)
        else :
            src, future_len = inputs
            src_key_padding_mask = None
            memory_key_padding_mask = None
            tgt_key_padding_mask = None
            tgt_seg_padding_mask = None
            future_target = None
            src_label = None

        tgt_mask = None
        # [Not tested yet]
        # masked_decoder = False
        # if masked_decoder:
        #     tgt_mask = torch.ones(src.size(1), src.size(1)).triu(diagonal=1).to(self.device)
        #     # Turn 1 to True, 0 to False
        #     tgt_mask = tgt_mask == 0

        if self.args.input_type == 'i3d_transcript':
            B, S, C = src.size()
            src = self.input_embed(src) #[B, S, C]
        elif self.args.input_type == 'gt':
            src = src.long()
            if src.dim() == 3:
                src = src.squeeze(-1)
            B, S = src.size()
            src = self.gt_emb(src)
        

        # action query embedding
        action_query = self.query_embed.weight
        action_query = action_query.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(action_query)


        # pos embedding
        pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        src = rearrange(src, 'b t c -> t b c')
        tgt = rearrange(tgt, 'b t c -> t b c')
        pos = rearrange(pos, 'b t c -> t b c')
        action_query = rearrange(action_query, 'b t c -> t b c')

        if self.args.model == 'tcca':
            src, tgt = self.transformer(src, tgt,
                                        src_key_padding_mask=src_key_padding_mask,
                                        tgt_mask=tgt_mask, 
                                        tgt_key_padding_mask=None, 
                                        query_embed=action_query, 
                                        pos_embed=pos, 
                                        tgt_pos_embed=None)

        elif self.args.model == 'tcca_LTC':
            src, tgt = self.transformer(src, tgt,
                                        src_key_padding_mask=src_key_padding_mask,
                                        tgt_mask=tgt_mask, 
                                        tgt_key_padding_mask=None, 
                                        query_embed=action_query, 
                                        pos_embed=pos,
                                        tgt_pos_embed=None)
        elif self.args.model == 'tcca_FACT':
            src, tgt, seg_loss, seg_pred = self.transformer(src, src_label, tgt,
                                        src_key_padding_mask=src_key_padding_mask,
                                        tgt_mask=tgt_mask, 
                                        tgt_key_padding_mask=None, 
                                        query_embed=action_query, 
                                        pos_embed=pos,
                                        tgt_pos_embed=None,
                                        pad_idx=self.src_pad_idx)

        tgt = rearrange(tgt, 't b c -> b t c')
        output = dict()
        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(tgt) #[B, T, C]
            
            if self.args.task == 'long':
                if self.args.duration == 'independent':
                    duration = self.fc_len(tgt)
                elif self.args.duration == 'action_partially_dependent':
                    tgt_class = torch.cat((tgt, F.softmax(output_class, dim=2)), dim=2)
                    duration = self.fc_len(tgt_class)
                duration = duration.squeeze(2) #[B, T]
                output['duration'] = duration

            output['action'] = output_class

        if self.args.crf:
            emissions = output_class
            # emissions = self.dropout(emissions)
            # emissions = F.softmax(output_class, dim=2)
            if mode == 'train':
                future_target = future_target.long()
                B, T, C = tgt.size() # [B, T, C]
                mask = get_pad_mask(future_target, self.src_pad_idx)
                # Negate the mask and convert to ByteTensor
                mask = (~mask).bool()
                # Turn future_target pad_idx to 0 so that it in the range of [0, C-1]
                future_target = future_target * mask

                llh_crf = self.crf(emissions, future_target, mask=mask, reduction='none')
                output['crf_llh'] = llh_crf
                output['crf_mask'] = mask
            else:
                B, T, C = tgt.size()
                decoded_seq = self.crf.decode(emissions)
                decoded_seq = torch.tensor(decoded_seq).to(self.device)
                output['crf_action'] = decoded_seq

        if self.args.anticipate_next:
            # action anticipation
            output_class = self.anticipate_next(tgt) #[B, T, C]
            output['action_next'] = output_class

        if self.args.anticipate_prev:
            # action anticipation
            output_class = self.anticipate_prev(tgt)
            output['action_prev'] = output_class

        if self.args.seg :
            # action segmentation
            if self.args.model == 'tcca_FACT':
                output['seg_loss'] = seg_loss
                output['seg_pred'] = seg_pred
                src = rearrange(src, 't b c -> b t c')
                output['seg_probs'] = src
                
            elif self.args.model == 'tcca_LTC':
                # If using LTContext we return the src from the model for later apply multistage loss
                src = rearrange(src, 'b n c t -> b n t c')
                output['seg'] = src
            elif self.args.model == 'tcca':
                output['seg'] = src
            else:
                src = rearrange(src, 't b c -> b t c')
                tgt_seg = self.fc_seg(src)
                output['seg'] = tgt_seg

        if self.args.anticipate_future_multilabel:
            # action segmentation
            tgt_matrix_future = self.fc_matrix_future(tgt) # [B, T, C]
            output['anticipate_future_multilabel'] = tgt_matrix_future
        
        return output


def get_pad_mask(seq, pad_idx):
    return (seq==pad_idx)



