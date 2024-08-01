#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

mkdir -p ../../EHRSHOT_ASSETS/figures

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_results_dir ../../EHRSHOT_ASSETS/results \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures-60k-steps \
    --shot_strat all \
    --is_skip_tables \
    --model_heads "[('clmbr', 'lr_lbfgs'), \
                    ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

                    ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

                    ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

                    ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

                    ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

                    ('mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

                    ('hyena-medium-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
                    ('hyena-medium-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
                    ('hyena-medium-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
                    ('hyena-medium-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
                    ('hyena-medium-16384--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \
                ]"

# Everything

    # --model_heads "[('clmbr', 'lr_lbfgs'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \
    #             ]"
