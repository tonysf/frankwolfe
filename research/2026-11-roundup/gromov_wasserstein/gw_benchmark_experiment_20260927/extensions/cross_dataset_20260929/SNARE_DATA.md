# SNARE-seq adult mouse cortex preparation

This extension freezes two RNA/chromatin Gromov–Wasserstein instances from the same paired cells, with RNA and ATAC order randomized independently. The 3,000-cell instance is nested within the instance containing up to 10,000 jointly retained cells. The preparation recipe was fixed before any optimization on these instances. Ground-truth correspondences are retained separately for evaluation, and are not arguments to optimization.

## Primary sources and exact input files

The six public processed files come from [NCBI GEO GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074), adult mouse cerebral cortex (`AdBrainCortex`), reported in [Chen, Lake and Zhang, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6893138/). The authors report 10,309 adult nuclei. The supplied matrices have 33,160 RNA features and 244,544 chromatin peaks. The barcode order differs between the two matrices; row-position matching would be incorrect.

`data_sources.json` records all six compressed input filenames, primary download URLs, byte counts and SHA-256 digests. Local download directory: `raw/snare_adult_20260929/`. Files were downloaded on 2026-09-29. The unmodified compressed bytes are verified against the frozen source manifest before preparation. Duplicate barcodes, missing or altered files, matrix/identifier dimension disagreement, negative/nonfinite counts and noninteger counts abort preparation.

## Frozen recipe

1. Align the modalities by exact barcode intersection, sorted lexicographically. Apply one joint quality-control mask: at least 200 detected RNA genes **and** at least 500 accessible ATAC peaks. Counts must be positive to be detected; explicit sparse zeros are removed. The output records raw cell counts, matched cell count, both individual failure counts, their union, and retained cell count. At least 3,000 retained cells are required.
2. Fit each modality's representation independently using all jointly retained cells, before taking benchmark subsets. No cell type annotation, cross-modality alignment or correspondence-based loss enters either representation fit.
3. RNA: divide counts by each cell's library size, multiply by 10,000, take natural `log1p`, and select the 2,000 genes with largest population variance of these lognormalized values. Exact variance ties use original feature index. Center the selected features without dividing by feature standard deviations, then compute 50 principal components using scikit-learn randomized PCA, 7 power iterations, 10 oversamples, seed 20260929. Store PCA scores, selected original feature indices/identifiers and explained variance ratios. RNA scores are not normalized by row.
4. ATAC: binarize positive peak counts. Remove peaks unobserved across retained cells. Define `TF_ij = binary_ij / number_of_accessible_peaks_in_cell_i` and `IDF_j = number_of_retained_cells / number_of_cells_accessing_peak_j`. Use `log1p(10000 * TF_ij * IDF_j)`, then uncentered randomized truncated SVD with 50 components, 7 iterations, 10 oversamples and seed 20260930. Discard component 1 and divide each remaining 49-dimensional row by its Euclidean norm. The exact formula is the recipe; “TF-IDF” alone is not a sufficient specification. All peak features with nonzero prevalence participate; there is no variable-peak selection.
5. Draw one fixed permutation of all retained paired cells with NumPy `default_rng`, seed 20260931. Prefixes select the shared cell sets for `snare3000` and `snare_full`, where the latter contains `min(10000, retained_count)` cells. Draw independent full-population source and target permutations, seeds 20260932 and 20260933; filter each permutation to each selected cell set. Thus each size contains the same cells in both modalities, with independently permuted orders, and both cell sets and the restricted view orders nest across sizes. Identical cell sets are necessary to define exact one-to-one evaluation correspondences.
6. Compute ordinary Euclidean distances (not squared Euclidean) in float64. For each instance and modality separately, divide by its own maximum. Store symmetric float32 matrices with zero diagonals and uniform float64 marginals. As a result, absolute objective losses across sizes refer to differently normalized problems and are not directly comparable.
7. Reject numerical degeneracy: retained ATAC row norm at most `64*eps*max(1, full_SVD_row_norm)`, or distance maximum at most `64*eps*max(1, maximum_absolute_coordinate)*sqrt(dimension)`, where `eps` is float64 machine epsilon. Do not turn floating-point noise from a constant representation into a normalized metric.

This is a fully specified new preprocessing recipe, not a claim to reproduce every preprocessing choice in the original SNARE-seq paper. RNA and ATAC remain different geometries: paired barcodes do not imply that zero GW loss is attainable. Both optimization algorithms receive the same frozen costs and marginals. Pairing labels define evaluation and joint quality control; no cell-type labels are used.

## Invocation and dependencies

Run full preparation as a scheduled CPU job on Ruche, never on its login nodes:

```bash
python prepare_snare.py --raw-dir RAW_DIRECTORY --out-dir NEW_EMPTY_OUTPUT_DIRECTORY --sources data_sources.json --threads 8
```

`--sources` defaults to `data_sources.json` next to the script. `--threads` defaults to `SLURM_CPUS_PER_TASK` when set, otherwise 1, and limits Matrix Market reading and all loaded BLAS/OpenMP pools through `threadpoolctl`. The manifest records the requested thread count. The output directory must be empty or absent. Preparation never overwrites frozen outputs. Nonzero exit or missing `preparation.json` means preparation is incomplete; partial artifacts must not be accepted as inputs.

Additional wheels for the existing Ruche Python 3.10 / NumPy 2.2.6 / SciPy 1.15.3 environment:

- `scikit-learn==1.7.2` (CPython 3.10, manylinux2014 x86_64 wheel)
- `joblib==1.5.2`
- `threadpoolctl==3.6.0`

No anndata, scanpy, h5py or PyTorch is required by this preparation script. Install the extra wheels into an isolated directory and add it to `PYTHONPATH`; preserve the experiment's existing environment. Bind OpenMP/BLAS thread counts to the scheduler allocation to prevent oversubscription. Full processing is scheduled; local tests use only tiny synthetic matrices. Exact representation bytes are tied to NumPy, SciPy, scikit-learn, BLAS/runtime and architecture; fixed seeds alone do not assert cross-platform bitwise reproducibility. Save one prepared dataset and share its hashes across both methods and every run.

## Outputs and consumer contract

`preparation.json` is written last, only after every prescribed output is complete. Its top-level keys include `status` (`completed`), `instances`, `qc`, `recipe`, `binding`, `representations`, `versions` and `python`. `instances` is keyed by `snare3000` and `snare_full`; each value includes:

```json
{
  "id": "snare3000",
  "n": 3000,
  "m": 3000,
  "filename": "snare3000.npz",
  "sha256": "SHA256_OF_INSTANCE_NPZ",
  "metadata_file": "snare3000.json",
  "metadata_sha256": "SHA256_OF_INSTANCE_JSON",
  "correspondence_file": "snare3000.correspondence.npz",
  "correspondence_sha256": "SHA256_OF_CORRESPONDENCE_NPZ"
}
```

All filenames in the manifest are relative to the output directory. Each record also contains cost-array hashes, distance scaling factors, representation shapes, family/objective names and the provenance binding.

- `snare3000.npz`, `snare_full.npz`: runner-compatible `A`, `B`, `a`, `b`; original matrix-row indices under `source_indices` and `target_indices`. No true-target maps or barcodes are present in these files. The solver reads only the costs and marginals.
- `snare3000.json`, `snare_full.json`: metadata bound to each instance's NPZ and correspondence NPZ hashes, raw file hashes, recipe hash, script hash, source manifest hash and common representation hash.
- `*.correspondence.npz`: `forward[i]` is the true ATAC column corresponding to RNA row `i`; `inverse[j]` is the true RNA row corresponding to ATAC column `j`. Also includes `source_cell_ids`, `target_cell_ids`, view indices into the common retained population, and the selected population prefix. `source_cell_ids == target_cell_ids[forward]` and `inverse[forward] == arange(n)` must hold. These arrays are evaluation-only.
- `representations.npz`: full retained-population RNA/ATAC representations, paired barcode IDs, original cell row indices, selected RNA feature indices/IDs and retained ATAC peak indices/IDs. It is an audit artifact, not optimizer input.

The recipe hash is SHA-256 of sorted-key, separator-minimized JSON encoding of the `RECIPE` object. The script and source-manifest hashes cover exact file bytes. Raw file digests cover downloaded compressed bytes; cost/representation array digests cover C-contiguous bytes. Consumers should verify manifest-bound hashes before starting trials.

## Verification

Seven small unit tests check barcode alignment and the joint mask, invalid/duplicate inputs, nested independently ordered subsets and both correspondence directions, representation reproducibility and normalization, deterministic gene-variance tie handling, saved metric equivalence for matched synthetic geometries, separate evaluation maps, overwrite refusal and raw provenance tamper rejection. The constant-representation fixture also verifies that negligible numerical noise is rejected instead of amplified by maximum normalization.

```bash
PYTHONPATH=ISOLATED_EXTRA_DEPS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s extensions/cross_dataset_20260929 -p test_prepare_snare.py -v
```
