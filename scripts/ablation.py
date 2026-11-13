ablation_dict = dict()

ablation_dict['n_breakfast_A'] = dict(
    anticipate_next=False
)

ablation_dict['n_breakfast_B'] = dict(
    anticipate_prev=False
)

ablation_dict['n_breakfast_C'] = dict(
    anticipate_prev=False,
    anticipate_next=False
)

ablation_dict['n_breakfast_D'] = dict(
    crf=False,
    anticipate_prev=False,
    anticipate_next=False
)

