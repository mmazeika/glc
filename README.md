# Gold Loss Correction

This repository contains the code for the paper

[Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise](http://arxiv.org/abs/1802.05300).

The code requires Python 3+ and PyTorch 0.3+.

<img align="center" src="glc_plots_figure.png" width="750">

## Overview

The Gold Loss Correction (GLC) is a semi-verified method for label noise robustness in deep learning classifiers. Using a small set of data with trusted labels, we estimate parameters of the label noise, which we then use to train a corrected classifier on the noisy labels. We observe large gains in performance over prior work, with a subset of results shown below. Please consult the paper for the full results and method descriptions.
 

<img align="center" src="glc_vision_results.png" width="750">

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2018glc,
      title={Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise},
      author={Hendrycks, Dan and Mazeika, Mantas and Wilson, Duncan and Gimpel, Kevin},
      journal={arXiv preprint arXiv:1802.05300},
      year={2018}
    }
