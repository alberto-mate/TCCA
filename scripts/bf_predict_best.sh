echo "split$1"
python main.py --dataset breakfast -p --seg --task long --anticipate \
    --model tcca_LTC --hidden_dim 64 --hidden_dim_decoder 128 --n_decoder_layer 2 \
    --n_query 8  --max_pos_len 2000 --sample_rate 3 --crf \
    --anticipate_next --anticipate_prev \
    --ltc_long_term_attn_g 16 --ltc_num_stages 4 --crf_transition_weight 0.1\
    --model_path "checkpoints_best/breakfast/split$1/models/checkpoint.ckpt" \
    --split=$1
