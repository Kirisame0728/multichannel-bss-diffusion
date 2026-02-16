# multichannel-bss-diffusion
## Folder structure
- src/datasets: dataset loaders
- src/models: TF-GridNet and others
- src/baselines: inference scripts (IVA / TF-GridNet / Neural-FCP)
- src/evaluation: metric wrappers and CSV summarization
- scripts: shell entrypoints
- exp: checkpoints
- outputs: separated wavs and CSVs

## Quick start (TF-GridNet inference + evaluation)
1) Prepare test set:
   /mnt/d/datasets/libri_test_2sp_6ch_8k

2) Run:
   bash scripts/run_tfgridnet_infer.sh
   EOF
