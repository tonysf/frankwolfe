# Reference helper validation

The 19 manual checks use the preserved CPU parser report; they do not load mathematical instance arrays, run an optimizer, or claim final A100 results. The test script locates the experiment root relative to its own path and needs only Python's standard library and the checked-in helper.

From the experiment root:

```bash
.venv/bin/python reports/reference_helper_validation/manual_checks.py
```

The command refreshes `validation.json`. Preserve `reports/campaign_summarizer_smoke/report/report.json` with this directory; it is the input fixture. Temporary adversarial copies and temporary output files are created with `tempfile` and removed automatically.

`cli_output/` is an exact copy of the already completed CLI smoke output. Its embedded command retains the original temporary path; `cli_validation.json` records that path, successful return status, and hashes of the preserved outputs. No provenance fields were rewritten when archiving.

To reproduce the CLI smoke into a new output directory:

```bash
.venv/bin/python summarize_feasible_references.py --report reports/campaign_summarizer_smoke/report/report.json --out /tmp/reference_helper_cli_new
```

Choose an unused output path: the helper deliberately refuses to overwrite a previously archived reference. Production invocation must use the complete, provenance-valid combined final report after all prescribed final and supplemental phases finish.

The native timing regression preserves an output that was already repaired at the last callback even when the solver later spends time in its epilogue and returns that same plan. The helper trusts the reporter-validated availability time, retains both finite nonnegative clocks, and does not move the output to the later solver finish time. Negative solver/output clocks remain rejected.

`cli_output_native_timing_fix/` and `cli_validation_native_timing_fix.json` preserve the current helper CLI smoke after that correction. The original CLI archive remains unchanged for provenance.
