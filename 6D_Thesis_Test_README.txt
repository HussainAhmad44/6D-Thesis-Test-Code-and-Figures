6D CHAOTIC S-BOX THESIS TEST SUITe
================================
Requires Python 3.10+, numpy, matplotlib.
Install: pip install numpy matplotlib

Place 6D_Thesis_Test_Suite.py and 6d_xor_annealed_sbox.csv in the same folder.
Run default:
  python 6D_Thesis_Test_Suite.py --out thesis_test_figures

Select parameter and state:
  python 6D_Thesis_Test_Suite.py --parameter c --variable Z --min 0 --max 0.3 --samples 600 --transient 1000 --keep 150 --out figures_cZ

Outputs: nine numbered PNGs, sbox_SAC_matrix.csv, sbox_DDT_matrix.csv, test_results.json.

Methods: 6D cyclic fractional map with five control parameters a,b,c,d,e=0.08; initial state (0.11,0.23,0.37,0.41,0.59,0.73). Nonlinearity is the minimum over 255 nonzero output masks. Differential uniformity is the largest nonzero-input-difference DDT entry. SAC is measured for all 8 input and 8 output bit combinations. BIC-SAC independence error is only a diagnostic, not BIC-NL. Lyapunov indicator uses a finite-difference wrapped-torus tangent method and may be affected by modulo discontinuities and numerical precision. No claim of proven chaos, hyperchaos, or security.

Recommended thesis captions: 'Numerical bifurcation diagram of the proposed fractional 6D map with parameter a varied and other parameters held at 0.08'; 'Empirical SAC matrix of the optimized 8-bit S-box'; 'Differential distribution table of the optimized S-box'. State sample counts and methods in captions. Do not label these plots as encryption avalanche or randomness tests; those require separate analyses.
