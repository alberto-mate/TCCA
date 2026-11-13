"""
TCCA Transformer class.

Copy-paste from github.com/facebookresearch/detr/blob/main/models/transformer.py with modifications.

"""

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import copy
from typing import Optional, List

from model.LTcontext.ltcontext import LTC
from model.FACT.blocks import FACT

class Transformer(nn.Module):

    def __init__(self, d_encoder=512, d_decoder=128, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_feedforward_multiplier=4, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()

        self.d_head = d_head = d_encoder // nhead

        encoder_layer = TransformerEncoderLayer(d_encoder, nhead, d_encoder*d_feedforward_multiplier,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_encoder) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_decoder, d_encoder ,nhead, d_decoder*d_feedforward_multiplier,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_decoder)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.nhead = nhead
        
        self.fc_seg = nn.Linear(d_encoder, args.n_class-1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_key_padding_mask, tgt_mask, tgt_key_padding_mask, query_embed, pos_embed, tgt_pos_embed):

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask, pos=pos_embed)
        memory_seg = rearrange(memory, 't b c -> b t c')
        memory_seg = self.fc_seg(memory_seg)
        
        hs = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
        return memory_seg, hs
    

class Transformer_LTContext(nn.Module):

    def __init__(self, n_class, input_dim=2048, d_decoder=128, nhead=8,
                 num_decoder_layers=6, d_feedforward_multiplier=4, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()


        self.encoder_LTContext = LTC(n_class, input_dim, dropout, args)

        d_encoder = self.encoder_LTContext.num_stages * n_class

        decoder_layer = TransformerDecoderLayer(d_decoder, d_encoder ,nhead, d_decoder*d_feedforward_multiplier,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_decoder)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.nhead = nhead

    def _reset_parameters(self):
        print("resetting parameters")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_key_padding_mask, tgt_mask, tgt_key_padding_mask, query_embed, pos_embed, tgt_pos_embed):
        if src_key_padding_mask is not None:
            not_src_key_padding_mask = ~src_key_padding_mask
        else:
            # In case of test, the src_key_padding_mask should be ones
            T, B, C = src.shape
            not_src_key_padding_mask = torch.ones(B, T).type(torch.bool).to(src.device) 
        memory = self.encoder_LTContext(src, not_src_key_padding_mask) # B, n_stages, C, T
        memory_reshaped = rearrange(memory, 'b n c t -> t b (n c)')
        hs = self.decoder(tgt, memory_reshaped, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
        return memory, hs
    
class Transformer_FACT(nn.Module):

    def __init__(self, n_class, input_dim=64, d_decoder=128, nhead=8,
                 num_decoder_layers=6, d_feedforward_multiplier=4, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()


        self.encoder_FACT = FACT(input_dim, n_class-1)
        d_encoder = n_class-1

        decoder_layer = TransformerDecoderLayer(d_decoder, d_encoder ,nhead, d_decoder*d_feedforward_multiplier,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_decoder)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # self._reset_parameters()

        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.nhead = nhead

    def _reset_parameters(self):
        print("resetting parameters")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_label, tgt, src_key_padding_mask, tgt_mask, tgt_key_padding_mask, query_embed, pos_embed, tgt_pos_embed, pad_idx):
        T, B, C = src.shape
        if src_key_padding_mask is not None:
            not_src_key_padding_mask = ~src_key_padding_mask
        else:
            # In case of test, the src_key_padding_mask should be ones
            not_src_key_padding_mask = torch.ones(B, T).type(torch.bool).to(src.device) 
        max_len_idx = T 
        memory, seg_loss, seg_pred = self.encoder_FACT(src, not_src_key_padding_mask, src_label, pad_idx) # (T+A) B C
        """
        pos_embed_f = self.encoder_FACT.frame_pe(memory[:max_len_idx])
        pos_embed_a = self.encoder_FACT.action_query
        pos_embed = torch.cat([pos_embed_f, pos_embed_a], dim=0).squeeze(1)
        # repeat in the batch dimension
        pos_embed = repeat(pos_embed, 't c -> t b c', b=B)
        """
        
        # TA = memory.shape[0]
        # pos_embed_temp = torch.zeros(TA, B, self.d_encoder).to(src.device)
        # pos_embed_temp.requires_grad = True
        # pos_embed_temp[:max_len_idx] = pos_embed
        # pos_embed = pos_embed_temp
        
        if src_key_padding_mask is not None:
            # Add 1 to src_key_padding_mask to have the same length as the memory
            src_key_padding_mask_temp = torch.zeros(memory.shape[1], memory.shape[0]).bool().to(src.device)
            src_key_padding_mask_temp[:,:max_len_idx] = src_key_padding_mask
            src_key_padding_mask = src_key_padding_mask_temp
            
        
        hs = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
        return memory, hs, seg_loss, seg_pred


class TransformerEncoder(nn.Module) :

    def __init__(self, encoder_layer, num_layers, norm=None) :
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = v = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, d_encoder, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.enc_to_dec_proj = nn.Linear(d_encoder, d_model) if d_model != d_encoder else nn.Identity()
        if not isinstance(self.enc_to_dec_proj, nn.Identity):
            nn.init.xavier_uniform_(self.enc_to_dec_proj.weight)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] =None):
        q = k = v = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.enc_to_dec_proj(self.with_pos_embed(memory, pos)),
                                   value=self.enc_to_dec_proj(self.with_pos_embed(memory, pos)),
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
