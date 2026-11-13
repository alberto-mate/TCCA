import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", default="tcca", choices=["tcca", "tcca_LTC", "tcca_FACT"], help='model type')
parser.add_argument("--dataset", type=str, default='breakfast')
parser.add_argument('--predict', "-p", action='store_true', help="predict for whole videos mode")
parser.add_argument("--model_path", default="", help="model path for prediction")
parser.add_argument("--seed", type=int, default=13452, help="seed for reproducibility")
parser.add_argument("--ablation", type=str, default="", help="ablation study")

#Dataset
#breakfast
parser.add_argument("--split", default="1", help='split number')
parser.add_argument("--task", type=str, help="Next Action Anticipation/long-term anticipation", default='long')

#Training options
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--workers", type=int, default= 24)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--gpu", type=str , default='0' , help='gpu id. If -1, use cpu')
parser.add_argument("--sample_rate", type=int, default=3)
parser.add_argument("--n_query", type=int, default=8)
parser.add_argument("--random_start_frame", action='store_true', help='random start frame just for training')
parser.add_argument("--random_sample_frame", action='store_true', help='random sample frame just for training')
parser.add_argument("--clip_grad_norm", type=float, default=-1, help='clip grad norm')
parser.add_argument("--tsme_seg_weight", type=float, default=0.2, help='tsme segmentation weight')


#TCCA specific parameters
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--hidden_dim_decoder", type=int, default=-1, help='hidden dim for decoder. If equal to -1, the decoder hidden dim is equal to encoder hidden dim')
parser.add_argument("--n_encoder_layer", type=int, default=2)
parser.add_argument("--n_decoder_layer", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--activation", type=str, default='gelu', help='activation function')
parser.add_argument("--segmentation_loss", type=str, choices=["cross_entropy+dice", "cross_entropy", "cross_entropy+tmse"], default="cross_entropy")
parser.add_argument("--duration", type=str, choices=['independent', 'action_partially_dependent'] , default='action_partially_dependent', help='duration prediction head type') 

# LTC specific parameters
# ltc_long_term_attn_g
parser.add_argument("--ltc_long_term_attn_g", type=int, default=16, help='number of global long term attention')
parser.add_argument("--ltc_num_stages", type=int, default=4, help='number of stages')
parser.add_argument("--ltc_num_layers", type=int, default=3, help='number of layers')

#Model parameters
parser.add_argument("--seg", action='store_true', help='action segmentation')
parser.add_argument("--anticipate", action='store_true', help='future anticipation')
parser.add_argument("--pos_emb", default='learned', choices=["none", "learned", "sinusoidal"], help='position embedding')
parser.add_argument("--max_pos_len", type=int, default=2000, help='position embedding number for linear interpolation')
parser.add_argument("--anticipate_future_multilabel", action='store_true', help='use matrix future multilabel')
parser.add_argument("--anticipate_next", action='store_true', help='anticipate next action')
parser.add_argument("--anticipate_prev", action='store_true', help='anticipate previous action')
parser.add_argument("--crf", action='store_true', help='use crf after the decoder')
parser.add_argument("--crf_init_matrix", default="", help='succesor representation matrix')
parser.add_argument("--crf_transition_weight", type=float, default=1, help='crf transition weight')

#Test on GT or decoded input
parser.add_argument("--input_type", default="i3d_transcript", help="select input type: [\"decoded\", \"gt\"]")
parser.add_argument("--input_dim", type=int, default=2048)
parser.add_argument("--runs", default=0, help="save runs")
