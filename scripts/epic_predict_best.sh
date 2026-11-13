echo "split$1"
python main.py --dataset EPIC -p --seg --task multi-label --anticipate \
    --model tcca_LTC --hidden_dim 64 --hidden_dim_decoder 128 --n_decoder_layer 2 \
    --n_query 250  --max_pos_len 4000 --sample_rate 2 --crf  \
    --anticipate_next --anticipate_prev --anticipate_future_multilabel \
    --ltc_long_term_attn_g 64 --ltc_num_stages 4 \
    --model_path "checkpoints_best/EPIC/split$1/models/checkpoint.ckpt" \
    --split=$1
