nohup python ./train_bfn.py --config_file configs/default.yaml --exp_name loss_w_1_1_5 \
--revision v1.1.1 --v_loss_weight 1 --bond_loss_weight 5 \
--epochs 10 --no_wandb --batch_size 24 \
> logs/train_bfn_bond_conv_later_260319.log 2>&1 &
