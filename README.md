# Coupled Critically Damped Langevin Dynamics for Learning Joint Distributions

## Abstract
In this work, we investigate capturing cross interactions in domains that require modeling coupled evolution. To this end, we introduce the coupled critically damped Langevin dynamics (CCLD) framework for facilitating information exchange while improving sampling efficiency and joint-distribution recovery. We demonstrate our framework's ability to significantly reduce KL divergence and correlation error on $N$-body coupled Ornstein–Uhlenbeck processes, and to reduce pointwise reconstruction error on coupled multiphysics PDE fields, while suggesting improved training stability over existing critically-damped Langevin approaches. Thus, our framework produces new insights into inducing cross interaction biases through the stochastic dynamics rather than refining model architectures. 

Please refer to the paper for full mathematical details and experimental results: https://openreview.net/forum?id=tHVtoKVAYE

## Setup

### Install

```
conda create -n coupledsho python=3.10
conda activate coupledsho
pip install -e .
```

### Coupled Ornstein-Uhlenbeck processes

10 seeds, 10,000 training iterations, 40,000 evaluation samples, swept over
$n\in\{8,16,32,64,128\}$ diffusion steps, at $N=2,3,4,5$ coupled populations:

```
bash run_stepcount_sweep.sh 10 10000 40000
```

```
python -m synthetic.make_stepcount_figures
```

### Coupled PDE field reconstruction

First, download the electro-thermal (`TE_heat`) split of the Multiphysics-Bench benchmark:

```
python -m pde.download_multiphysics_bench --out-dir pde/multiphysics-bench
```

`pde/multiphysics-bench` is the default data root every script below assumes; export
`PDE_DATA_ROOT=/wherever/you/downloaded/it` instead if you'd rather keep the data elsewhere.

10 seeds (0-9), 150 epochs, batch size 1024, learning rate $10^{-2}$, 64 diffusion steps:

```
for METHOD in ccld ccld_independent ddpm sdm; do
  python -m pde.run_experiment \
    --config pde/config.yaml --data-root pde/multiphysics-bench --problem TE_heat \
    --method ${METHOD} --seeds 0,1,2,3,4,5,6,7,8,9 --n-epochs 150 --batch-size 1024 --lr 1e-2 \
    --n-diff-steps 64 --out-dir results/experiment_2_physics/TE_heat_comparison
done
```

Equivalently, `bash run_pde_baselines.sh` runs this same CCLD/CLD/DDPM/SGM sweep.

## Future Work
Open directions include testing whether the coupling advantage generalizes to coupling strengths
other than the single value (0.6) used above (see `PROTOCOL.md`), extending non-mean-field
(antisymmetric) coupling to the PDE task, extending the PDE benchmark to direct field generation
rather than diffusing over a shared encoder's latent, predicting score residuals relative to CLD's
closed form rather than the entire score as the original CLD paper does, and designing more
efficient sampling methods to reduce discretization errors arising from the score term and
symmetric mode of the CCLD framework.