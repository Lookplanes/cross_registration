## Command:
CUDA_VISIBLE_DEVICES=0,1 nohup /data2/xujr/conda-envs/transmorph/bin/python -u train_TransMorph_supervise.py > nohup_train-no_affine.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup /data2/xujr/conda-envs/transmorph/bin/python -u train_TransMorph_supervise.py > nohup_train-no_appearance.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 nohup /data2/xujr/conda-envs/transmorph/bin/python -u train_TransMorph_unsupervise.py > nohup_train-unsup.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 /data2/xujr/conda-envs/transmorph/bin/python /home/xujr/cross_registration/test_pipeline_cross_modality_registration.py


## Results:
ch1_to_ch0

================================================================================
[Method: Our Pipeline]
--------------------------------------------------------------------------------
* Intra-modal Metrics (Fake C0 vs Real C0):
  - Pre-Reg  -> ZNCC : 0.4961  |  MSE : 0.0339
  - Post-Reg -> ZNCC : 0.6940  |  MSE : 0.0228
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI  : 1.0633  |  Fore-Dice : 0.9420  |  cross-ZNCC : 0.1226
  - Post-Reg -> NMI  : 1.0720  |  Fore-Dice : 0.9245  |  cross-ZNCC : 0.2023
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 6.4935
  - Folding (Negative Jac) : 3.2837%
================================================================================

================================================================================
[Method: Unsupervised TransMorph Baseline]
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI: 1.0633  |  Fore-Dice: 0.9420  |  cross-ZNCC: 0.1226
  - Post-Reg -> NMI: 1.0773  |  Fore-Dice: 0.9507  |  cross-ZNCC: 0.1738
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 7.2733
  - Folding (Negative Jac) : 0.0000%
================================================================================

================================================================================
[Method: Traditional ANTs/SyN (MI) Baseline]
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI: 1.0633  |  Fore-Dice: 0.9420  |  cross-ZNCC: 0.1226
  - Post-Reg -> NMI: 1.0777  |  Fore-Dice: 0.9275  |  cross-ZNCC: 0.2307
--------------------------------------------------------------------------------
* Speed Metrics:
  - Optimization Time: 1.2934 s/pair
================================================================================


ch0_to_ch1

================================================================================
[Method: Our Pipeline]
--------------------------------------------------------------------------------
* Intra-modal Metrics (Fake C0 vs Real C0):
  - Pre-Reg  -> ZNCC : 0.2170  |  MSE : 0.0430
  - Post-Reg -> ZNCC : 0.5689  |  MSE : 0.0273
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI  : 1.0552  |  Fore-Dice : 0.9429  |  cross-ZNCC : 0.1194
  - Post-Reg -> NMI  : 1.0721  |  Fore-Dice : 0.9270  |  cross-ZNCC : 0.1951
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 6.2510
  - Folding (Negative Jac) : 3.0735%
================================================================================

================================================================================
[Method: Unsupervised TransMorph Baseline]
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI: 1.0552  |  Fore-Dice: 0.9429  |  cross-ZNCC: 0.1194
  - Post-Reg -> NMI: 1.0771  |  Fore-Dice: 0.9520  |  cross-ZNCC: 0.1786
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 7.2736
  - Folding (Negative Jac) : 0.0000%
================================================================================

================================================================================
[Method: Traditional ANTs/SyN (MI) Baseline]
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI: 1.0552  |  Fore-Dice: 0.9429  |  cross-ZNCC: 0.1194
  - Post-Reg -> NMI: 1.0836  |  Fore-Dice: 0.9324  |  cross-ZNCC: 0.2540
--------------------------------------------------------------------------------
* Speed Metrics:
  - Optimization Time: 1.2915 s/pair
===============================================================================


ch1_to_ch0
改版后
================================================================================
[Method: Our Pipeline]
--------------------------------------------------------------------------------
* Intra-modal Metrics (Fake C0 vs Real C0):
  - Pre-Reg  -> ZNCC : 0.4961  |  MSE : 0.0339
  - Post-Reg -> ZNCC : 0.7942  |  MSE : 0.0171
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI  : 1.0633  |  Fore-Dice : 0.9420  |  cross-ZNCC : 0.1226
  - Post-Reg -> NMI  : 1.0811  |  Fore-Dice : 0.9344  |  cross-ZNCC : 0.2580
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 6.1499
  - Folding (Negative Jac) : 0.2578%
================================================================================

加了mask 和 DICE 的膨胀后
================================================================================
[Method: Our Pipeline]
--------------------------------------------------------------------------------
* Intra-modal Metrics (Fake C0 vs Real C0):
  - Pre-Reg  -> ZNCC : 0.4961  |  MSE : 0.0339
  - Post-Reg -> ZNCC : 0.8240  |  MSE : 0.0175
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI  : 1.0633  |  Fore-Dice : 0.9636  |  cross-ZNCC : 0.1226
  - Post-Reg -> NMI  : 1.0892  |  Fore-Dice : 0.9720  |  cross-ZNCC : 0.2934
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 3.2121
  - Folding (Negative Jac) : 0.0011%
================================================================================


================================================================================
[Method: Unsupervised TransMorph Baseline]
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI: 1.0633  |  Fore-Dice: 0.9636  |  cross-ZNCC: 0.1226
  - Post-Reg -> NMI: 1.0773  |  Fore-Dice: 0.9663  |  cross-ZNCC: 0.1738
--------------------------------------------------------------------------------
* Flow Quality Metrics:
  - EPE (End Point Error)  : 7.2733
  - Folding (Negative Jac) : 0.0000%
===============================================================================

================================================================================
[Method: Traditional ANTs/SyN (MI) Baseline]
--------------------------------------------------------------------------------
* Cross-modal Metrics (Real C1 vs Real C0):
  - Pre-Reg  -> NMI: 1.0633  |  Fore-Dice: 0.9636  |  cross-ZNCC: 0.1226
  - Post-Reg -> NMI: 1.0778  |  Fore-Dice: 0.9435  |  cross-ZNCC: 0.2361
--------------------------------------------------------------------------------
* Speed Metrics:
  - Optimization Time: 1.3169 s/pair