python sample_for_pocket.py ~/zhouwenbiao/data/debug/molgen/7kac_protein_noligand.pdb \
~/zhouwenbiao/data/debug/molgen/6nfy_ligand.sdf \
./logs/user002_bond_added/loss_w_1_1_5/v1.1.1/checkpoints/epoch01-val_loss5.19-mol_stable0.00-complete0.90-vina_score0.00.ckpt \
./logs/user002_bond_added/loss_w_1_1_5/v1.1.1/config.yaml

python ./train_bfn.py --config_file configs/default.yaml --exp_name test_eval --revision v1.1.1 --no_wandb --batch_size 100  --test_only --ckpt_path ./logs/user002_bond_added/loss_w_1_1_5/v1.1.1/checkpoints/epoch07-val_loss4.89-mol_stable0.00-complete0.92-vina_score0.00.ckpt --num_samples 1

#evaluation
TEST_SET_DIR="./data/test_set_with_h" CKPT_PATH="./logs/user002_bond_added/loss_w_1_1_5/v1.1.1/checkpoints/epoch07-val_loss4.89-mol_stable0.00-complete0.92-vina_score0.00.ckpt" CONFIG_PATH="./logs/user002_bond_added/loss_w_1_1_5/v1.1.1/config.yaml" ./run_pipeline.sh --test