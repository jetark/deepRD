"""
End-to-end reproduction pipeline for the paper's E3-equivariant dimer noise
sampler (the ``deep4`` E3DimerCVAE + fluctuation-dissipation "Fix-A" correction).

The pipeline has four stages, each a standalone script that only depends on the
public ``deepRD`` API (no test-tree or scratch dependencies):

    train.py            seed -> trained deep4 E3DimerCVAE checkpoint
    calibrate_gains.py  checkpoint -> friction gains (teacher-forced, fd_gains.json)
    gain_sweep.py       friction gains -> smallest FD gain landing dvx-std ~ 1.0
    generate_rollout.py trained model + gains + gain -> long reduced-dimer rollout

``run_pipeline.py`` chains all four with the frozen ``configs/deep4.yaml`` recipe
and the published seed (302) so a single command reproduces the paper model.
See README.md for the exact commands.
"""

from .fd_sampler import E3DimerRolloutSamplerFD

__all__ = ["E3DimerRolloutSamplerFD"]
