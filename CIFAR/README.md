To run an individual experiment, use\
`python train_<method>.py --gold_fraction $1 --corruption_prob $2 --corruption_type $3`.

An example batch job launcher is in `makebatches.sh`. The name of each experiment script indicates the method it uses. These are described below.

* `train_confusion.py`: Confusion Matrix
* `train_convex_combo.py`: Distillation Correction
* `train_forward.py`: Forward Correction
* `train_forward_gold.py`: Forward Gold
* `train_gold_only.py`: Trusted Only
* `train_ideal.py`: perfect C_hat estimate (see Discussion section in paper)
* `train_ours.py`: GLC
* `train_ours_adjusted.py`: base rate adjustment (see Discussion section in paper)
* `train_ours_calibrated.py`: confidence calibration (see Discussion section in paper)
