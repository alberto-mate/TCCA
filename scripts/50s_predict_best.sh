echo "split$1"
python main.py --dataset 50salads -p --seg --task long --anticipate \
    --model tcca_LTC --hidden_dim 128 --hidden_dim_decoder 256 --n_decoder_layer 3 \
    --n_query 20  --max_pos_len 3100 --sample_rate 3 --crf  \
    --anticipate_next --anticipate_prev \
    --ltc_long_term_attn_g 16 --ltc_num_stages 6  \
    --model_path "checkpoints_best/50salads/split$1/models/checkpoint.ckpt" \
    --split=$1
