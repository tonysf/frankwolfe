# SHARE-seq full-population test on four A100s

The original 25,000-cell GPU and report jobs were cancelled before starting.
The active campaign uses all 31,890 paired cells passing the unchanged joint QC,
reusing the already frozen representations. `prepare_full.py` regenerates full
distance matrices and correspondences without refitting or subsampling.

This extension implements the user's request to shard the next large real-data
comparison across four Ruche A100 GPUs. It leaves completed benchmark runtimes
and results unchanged. The initial single-GPU guidance in the attached protocol
is superseded by the user's explicit request for this experiment.

Each rank holds contiguous rows of the coupling, momentum, source distance
matrix A, and A squared. B, B squared, and marginal vectors are replicated.
The Rademacher probes are generated once by rank zero and broadcast. Thin
products and row marginals are gathered; column sums are reduced. KL column
normalization uses global stabilized logsumexp. Exact diagnostics broadcast
bounded coupling row blocks. No full coupling is gathered on a GPU.

Both methods retain the full squared-loss GW objective and the same unbiased
q=128 sketch estimator. Distributed reduction and fused arithmetic can change
floating-point summation order; this is numerical equivalence, not bitwise
identity. The kernel and allocated GPU tests compare against the frozen dense
reference in both precisions and from both initializations.

`campaign.json` freezes a bounded exploratory experiment: product initialization,
three paired stochastic seeds, 300 solver seconds per method, and constants
transferred unchanged from the selected SNARE-10,000 configurations. There is
no SHARE-seq tuning and no claim that these are optimal SHARE-seq settings.
The methods are Smoothed row FW + momentum (sketch gradient) and Transport
KL-proximal gradient + momentum (sketch gradient). Each method receives all
four GPUs sequentially, with paired order alternating across seeds.

The launch order is preparation, allocated four-GPU correctness tests, two
30-second 31,890-cell pilots, resource gates, then six final runs. The pilot must
validate the exact float64 loss and rounded feasibility, with every GPU's
reserved-memory peak below 80% of physical capacity and aggregate process RSS
peaks below 80% of the 128 GiB host allocation. A failed gate stops the campaign.

Solver time includes CUDA synchronization, boundary barriers, and algorithm
communication. The small scalar collective that combines timer readings is
instrumentation and remains in total elapsed time. Feasible-output time is
solver time plus the current output's rounding cost. Diagnostics, setup,
serialization, and float64 validation are reported separately. Multiply solver
seconds by four for solver GPU-seconds; Slurm allocation elapsed times four
reports total allocated GPU time. Neither equals an assumed fourfold speedup.

Final float64 validation releases solver state before loading promoted costs,
then validates raw and rounded plans one at a time without re-rounding. Final
plans are saved in row shards (about 48.8 GB across six runs); pilot outputs are
hashed and validated without retaining dense plans. Current outputs alone are
reported; there is no uncounted incumbent or correspondence-based selection.

All substantial preprocessing and GPU execution use Slurm. `prepare_full.sbatch`
requests cpu_med/8 cores/32 GiB; `gpu.sbatch` requests gpua100/one node/four GPUs,
eight CPU cores, and 128 GiB without exclusive-node allocation. Both use fwllm.
Submissions use isolated source payloads with checksums and a recorded revision.
See DATA.md for source filtering, preprocessing, correspondence, and QC details.
