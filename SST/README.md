To run experiments with the Distillation Correction method, use\
`python SST_convex_combo.py --corruption_type $1`.

To run experiments with the Trusted Only method, use\
`python SST_gold_only.py --corruption_type $1`.

To run experiments with all other methods, use\
`python SST_experiments_pytorch.py --method $1 --corruption_type $2`,\
where the methods follow the naming conventions in the CIFAR folder.

Note that these scripts run experiments for all gold fractions and corruption strengths.
