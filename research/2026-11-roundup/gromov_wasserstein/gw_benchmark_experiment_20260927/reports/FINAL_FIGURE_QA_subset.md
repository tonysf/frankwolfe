# Final figure QA: panels 03, 06 and 07

Inspected all eight actual PNGs visually; checked each paired PDF for existence, nonzero size and PDF header. Re-viewed both gap PNGs after the presentation-only margin correction. No title, axis-label or legend boundary clipping, hidden missing-coverage condition, or artificial uncertainty bars was found. One minor legend/IQR overlap remains, recorded below. No original data, runtime, or figure-generator code was changed by this review.

Current report: [stage_c_final/report.json](stage_c_final/report.json), SHA-256 `39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c`. Reporter SHA-256 `ffe846943058a57797488ed79749501145b53f939dba9ab968ca3a2424583216`. Review date: 2026-09-28T07:49:08.972304+00:00.

| PNG inspected | Result | PDF bytes |
|---|---|---:|
| [panel_03_raw_feasibility.png](stage_c_final/panel_03_raw_feasibility.png) | Pass. Both residual axes, seven-method legends and full IQRs are visible. Open FW markers overlap near-zero methods but are correctly encoded. | 36825 |
| [panel_03_rounded_objective.png](stage_c_final/panel_03_rounded_objective.png) | Pass. Actual-time and objective IQRs are visible; open FW markers disclose partial checkpoint coverage. | 28239 |
| [panel_06_original_gap.png](stage_c_final/panel_06_original_gap.png) | Pass after re-view. Zero is retained on the signed symlog axis; the PG first-window IQR is fully visible. | 23423 |
| [panel_06_raw_feasibility.png](stage_c_final/panel_06_raw_feasibility.png) | Pass. Both axes and scientific-notation multipliers are intact; near-zero methods overlap as expected. | 25550 |
| [panel_06_rounded_objective.png](stage_c_final/panel_06_rounded_objective.png) | Pass. PG first-window IQR is visible; later and comparator values overlap at zero. | 22825 |
| [panel_07_original_gap.png](stage_c_final/panel_07_original_gap.png) | Pass after re-view. Regenerated scale-space margins remove excessive empty negative space and retain the full PG first-window IQR. | 23464 |
| [panel_07_raw_feasibility.png](stage_c_final/panel_07_raw_feasibility.png) | Pass. Titles, both residual axes, scientific-notation multipliers and legends are intact. | 25569 |
| [panel_07_rounded_objective.png](stage_c_final/panel_07_rounded_objective.png) | Minor cosmetic overlap: the legend covers an interior segment of the 30-second FW IQR bar. Its median and both endpoints remain visible; no data are clipped. | 22437 |

Coverage and uncertainty checks:

- Panel 03 is the single BMMC instance with five distinct randomized initializations per method. Stochastic methods also vary their oracle seed. All seven displayed methods have five observations at 1 and 30 seconds; FW has three of five at 3 and 10 seconds, disclosed by open markers and the caption. Other methods have full coverage throughout. The native method is prospectively excluded from the randomized panel, not silently dropped as a failed run.
- The wide FW horizontal bar at the first BMMC window is the observed time IQR: 1.19195–20.87058 seconds including current-output repair. Its median is 1.99665 seconds. This is recorded timing variation, not interpolation or a nominal-time error bar.
- Panels 06 and 07 each show three methods over five independent problems, with all five observed at every 1/3/10/30-second window. The descriptive median/IQR is across one within-instance median per problem. Deterministic methods therefore have legitimate across-problem dispersion. No duplicate scientific units are recorded.
- Median/IQR captions explicitly distinguish dispersion from confidence intervals. Error bars derive directly from recorded quartiles and appear only for metric counts greater than one; small or zero IQRs are allowed to collapse. The broad noisy FW objective IQR (0.000421166–0.0318201) is present in the report aggregates.
- Rounded-output plots use actual solver plus current-output repair time; raw-feasibility plots use actual solver time. Signed FW gaps are passed directly to a symlog axis without taking absolute values or clipping negatives. In these two small-PG strata the plotted gap quartiles happen to be nonnegative; near-zero positive values are visually indistinguishable from zero at this scale.
- The margin correction changed only panel_07_original_gap.png among this subset; the other seven PNGs are byte-identical to their already-viewed archived versions. [FINAL_PLOT_MARGIN_FIX.json](FINAL_PLOT_MARGIN_FIX.json) records unchanged scientific report fields and aggregates. The old negative-space region represented padding, not negative observed gaps.

Plot provenance (current files):

| PNG | PNG SHA-256 | PDF SHA-256 |
|---|---|---|
| panel_03_raw_feasibility.png | `1264c02b2728f27414b4a35367764b5d5b4989e26508ef61504eea887f4a715b` | `581d06beae5ad1cd60c373cd2b91b43d23ed9d622be892cf63189bb3af8d87d7` |
| panel_03_rounded_objective.png | `f69e37bccbe50c38235a824b82a827adaf9666afa662ac63ebf57a68f9df19ad` | `a486a5a597254aa46d8a2253f02e109788c50cf2f80738dc8a378a7ea5099a7a` |
| panel_06_original_gap.png | `d2eb5c431101baec4bf5cb311365d9056e5dda817bc9d94961b1a8bfaee73131` | `4c3f6fda7b5abd8a8f5b1ce3009e1b2cdc260104d4d9bf6a3a8b2bc5a6b6be62` |
| panel_06_raw_feasibility.png | `05eeadccef93d5d1c5acc191fe6d10b339e8cb77ec5b3b7c907c31e8d3c617e8` | `ebfccc15e36a88721c45f3ec33597a96ef7a752a3b8061a74e364e5bedb1f3ae` |
| panel_06_rounded_objective.png | `218b88cd0677dfc99de7f77b2bc898b6c1fedc963f66a94f37d81f4fa36173a5` | `eff9865d477d567acc29bb9bc52f7e16dd50cf0fa474c1546021b79387560e23` |
| panel_07_original_gap.png | `7d5463c6ac62f4107145e9d4800872119c15d28d4941b09f7d0b811cd602ef49` | `b16db8cb40db49ee691ab75e5e0811ba940c3322c37cd73f0d569b6a10210675` |
| panel_07_raw_feasibility.png | `cc824adcdaf2e4a8ac4f8baa71632bca928992399ba687e7109cfd0d221ddb70` | `a7e1af52c55069b060e13830732aa40f6728bb21fd4839cca2ecaea34f6f79a2` |
| panel_07_rounded_objective.png | `b41986e837e9f17c1d2feea5db810ac52a9a018a43287e9a7b2d7b51b0f67c11` | `2aba12336081886296cda58e112949b2aa6bfb80a24a48efa7028530fa1c4ce6` |

Scope: this is eight of the 22 final figure pairs. The overall campaign report separately retains five native scientific failures; none is concealed by a successful-final-output assertion here. PDF rendering was not separately inspected; visual inspection used the paired PNG from the same figure export.
