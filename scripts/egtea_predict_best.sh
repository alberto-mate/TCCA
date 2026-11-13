echo "split$1"
python main.py --dataset EGTEA -p --seg --task multi-label --anticipate \
    --model tcca_LTC --hidden_dim 128 --hidden_dim_decoder 256 --n_decoder_layer 2 \
    --n_query 30  --max_pos_len 3100 --sample_rate 1 --crf  \
    --anticipate_next --anticipate_prev --anticipate_future_multilabel \
    --ltc_long_term_attn_g 16 --ltc_num_stages 4 \
    --model_path "checkpoints_best/EGTEA/split$1/models/checkpoint.ckpt" \
    --split=$1