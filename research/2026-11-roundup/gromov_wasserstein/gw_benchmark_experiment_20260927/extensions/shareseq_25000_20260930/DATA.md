# SHARE-seq mouse-skin 25,000-cell instance

This extension freezes one paired RNA/ATAC Gromov–Wasserstein instance with 25,000 cells in each modality. The dense costs are standalone memory-mapped NPY arrays so distributed solvers can read row shards without loading the full problem on each process. Preparation is a scheduled CPU job; local verification uses only tiny synthetic inputs.

## Sources and upstream processing

The original [Ma et al., Cell 2020 paper](https://pubmed.ncbi.nlm.nih.gov/33098772/) reports 34,774 paired mouse-skin profiles. This experiment uses the publicly available raw-count H5AD exports listed in the [official scGLUE dataset documentation](https://scglue.readthedocs.io/en/v0.3.1/data.html):

| Input | Bytes | Shape (cells × features) | Nonzero entries |
|---|---:|---:|---:|
| [Ma-2020-RNA.h5ad](https://download.gao-lab.org/GLUE/dataset/Ma-2020-RNA.h5ad) | 50,822,146 | 32,231 × 21,478 | 19,789,230 |
| [Ma-2020-ATAC.h5ad](https://download.gao-lab.org/GLUE/dataset/Ma-2020-ATAC.h5ad) | 274,596,957 | 32,231 × 340,341 | 127,118,599 |

Files were downloaded and inspected on 2026-09-30. `data_sources.json` freezes their full SHA-256 digests and URLs. Both files have sparse CSR `X` with encoding version `0.1.0`, stored float32 raw counts. Their 32,231 canonical `obs/cells` IDs match exactly, including order; preparation nevertheless aligns by explicit identifiers. The H5AD reader does not use cell-type labels, variable-feature flags, embeddings, or layers.

The [scGLUE collection notebook](https://github.com/gao-lab/GLUE/blob/5a30146c6148a52210f547c9e9c330ecfbdf1090/data/collect/Ma-2020.ipynb) constructs `X` directly from the original GEO count matrices, without count normalization. It restricts cells to an author-provided correspondence/annotation table and excludes its `Mix` category; keeps RNA genes with genomic annotation and at least one count; excludes ATAC peaks overlapping the mm10 blacklist and peaks without counts. Its barcode conversion makes the paired modalities use matching canonical IDs. Thus 32,231 is the upstream curated paired population, not the full 34,774 reported in the paper. These upstream filters are inherited and disclosed; our 25,000-cell sample is not conditioned on optimization results. The notebook's source URL and content hash are recorded in the source manifest. GEO accession: GSE140203; RNA sample GSM4156608 and ATAC sample GSM4156597.

## Fixed preparation recipe

The scientific transformations and seeds match `../cross_dataset_20260929/prepare_snare.py`; the relevant functions are copied into this script so its bundle is self-contained.

1. Verify the two downloaded files against the frozen byte lengths and SHA-256 digests. Read only sparse count `X`, observation IDs, and feature IDs with h5py. Accept explicitly encoded CSR/CSC versions 0.1.0 or 0.2.0, with valid dimensions and sparse structure. Reject duplicate/empty identifiers and negative, nonfinite, or noninteger values before duplicate-coordinate aggregation. Never infer another count layer or invert a normalization.
2. Intersect cell IDs lexicographically, apply one joint QC mask requiring at least 200 detected RNA genes and at least 500 detected ATAC peaks, and retain original row indices. Abort if fewer than 25,000 paired cells pass. Explicit sparse zeros do not count as detected features. Record raw, paired, retained and separate/union failure counts.
3. Fit both representations independently on every jointly retained cell, before selecting the 25,000-cell subset. RNA: divide by cell library size, multiply by 10,000, natural `log1p`, choose 2,000 genes by descending population variance with ties resolved by original feature index. Center without feature scaling, use randomized PCA with 50 components, 7 power iterations, 10 oversamples and seed 20260929. Keep unnormalized PCA scores.
4. ATAC: binarize positive counts, discard unobserved peaks, set `TF = binary / accessible_peaks_per_cell`, `IDF = retained_cell_count / cells_accessing_peak`, and transform with `log1p(10000 * TF * IDF)`. Compute uncentered randomized truncated SVD with 50 components, 7 iterations, 10 oversamples and seed 20260930. Discard the first component and normalize the remaining 49 coordinates of each cell to unit Euclidean norm. Reject retained row norms at most `64*eps*max(1, full_SVD_row_norm)`.
5. Select 25,000 cells as the prefix of a full retained-population permutation from NumPy `default_rng(20260931)`. Independently permute all retained cells with seeds 20260932 and 20260933 for the RNA and ATAC orders, then filter each order to the selected set. Save both exact correspondence directions separately from solver arrays. If smaller controls are later generated with the helper, the fixed population permutation and view orders make them nested; the command-line preparation currently emits only the predeclared 25,000-cell instance.
6. For each modality, calculate ordinary Euclidean distances in float64. The first tiled pass finds the global maximum; the second recomputes distances, divides in float64, then writes float32. Upper-triangle tiles and their transposes preserve symmetry, and diagonals are zero. Default tile width is 512, so no full float64 distance matrix is held in memory. Reject maximum distances at most `64*eps*max(1, max_abs_coordinate)*sqrt(dimension)`. Each modality has its own normalization maximum. The resulting objective scale is instance-specific.
7. Write uniform float64 marginals. Hash all arrays and correspondence files. Write the per-instance `manifest.json` and finally the top-level `preparation.json`; missing top-level completion means the output is incomplete. Never overwrite a populated output directory.

This creates a new fully specified geometry from the upstream counts. It does not claim to reproduce the original paper's full analysis. Pairing labels serve QC and evaluation, not cross-modality alignment or optimization. Lower GW loss does not by itself establish improved biological matching.

## Invocation and dependencies

Full processing must run inside the site's documented CPU scheduler allocation. The script itself performs no network requests or installations:

```bash
python prepare_shareseq.py \
  --raw-dir RAW_DIRECTORY \
  --out-dir NEW_EMPTY_OUTPUT_DIRECTORY \
  --sources data_sources.json \
  --threads 8 --distance-block-rows 512
```

Dependencies are NumPy, SciPy, h5py, scikit-learn, joblib and threadpoolctl. Reuse the existing isolated SNARE preparation scikit-learn 1.7.2, joblib 1.5.2 and threadpoolctl 3.6.0 dependencies. Add a Python 3.10 compatible h5py wheel in the isolated dependency directory if the allocated interpreter cannot already import it; anndata and scanpy are unnecessary. The script imports scikit-learn before applying threadpool limits. Scheduler CPU allocation and BLAS/OpenMP environment variables must agree. A 64 GB CPU allocation provides room for sparse float64 copies and randomized SVD; actual measured resource use is recorded by the scheduler, not asserted in advance.

The two float32 cost files require approximately 5.00 GB combined; marginals and representation/correspondence files add little. A single float32 25,000 × 25,000 coupling is approximately 2.50 GB; each saved raw/rounded pair is approximately 5.00 GB. Runtime outputs are additional storage and must be budgeted separately. The representations and hashes are frozen once and shared across methods and repetitions. Seeds alone do not promise identical representation bytes across different BLAS/library versions.

## Output contract

```text
OUT/
  preparation.json
  representations.npz
  shareseq25000/
    manifest.json
    A.npy
    B.npy
    a.npy
    b.npy
    correspondence.npz
```

`A.npy` and `B.npy` contain float32 symmetric distances; `a.npy` and `b.npy` contain uniform float64 marginals. Load them with `np.load(path, mmap_mode='r', allow_pickle=False)`. `manifest.json` has `n`, `m`, `id`, `array_files_sha256` mapping exactly those four filenames to whole-file SHA-256 digests, and `arrays` records containing each file's relative filename, hash, dtype, shape and bytes. It also binds the correspondence hash, distance scales, source/recipe/script/representation hashes and preprocessing metadata. Paths in this per-instance manifest are relative to its own directory, except `binding.representations_file`, which refers to the preparation output root.

`preparation.json` contains `status: completed`, QC, package versions, the recipe, and `instances.shareseq25000`, including `directory: shareseq25000`, `metadata_file: shareseq25000/manifest.json`, and its SHA-256. It duplicates the instance record for audit; array paths remain relative to the instance directory. Consumers must check completion, manifest and array hashes before accepting a run's data provenance.

`correspondence.npz` contains `forward[i]`, the true ATAC column for RNA row `i`, and `inverse[j]`, the true RNA row for ATAC column `j`. It also includes source/target cell IDs, original count-matrix row indices, retained-population indices and selected permutation prefix. No pairing arrays occur in the four solver inputs. `representations.npz` stores full retained-population representations, selected feature IDs/indices and original cell IDs/indices for audit.

## Verification

Seven small unit tests cover CSR/CSC count ingestion and string IDs; rejection of normalized/invalid counts, malformed sparse structure, unsupported encodings and duplicate identifiers; shuffled-ID joint QC; raw provenance tampering; exact equality of blocked distances with float64 dense normalization; representation reproducibility and ATAC binarization; and saved correspondence/array hash consistency, nesting, memory mapping and overwrite refusal.

```bash
PYTHONPATH=ISOLATED_EXTRA_DEPS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s extensions/shareseq_25000_20260930 \
  -p test_prepare_shareseq.py -v
```

The seven tests passed locally on 2026-09-30. Only small fixtures were processed; actual retained count, representations and 25,000-cell costs still require the scheduled preparation job.
