"""Add the model, data, schedules, and comparator settings to the PDF report."""

import json
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt


INK = "#152531"
ACCENT = "#075985"
MUTED = "#435564"
DATASET_URL = "https://arxiv.org/html/2306.05907"
PROJECTOR_URL = "https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html"
SPRING_URL = "https://www.pure.ed.ac.uk/ws/portalfiles/portal/240336653/SIIMS_manuscript_final_accepted.pdf"
PROXSARAH_URL = "https://jmlr.org/papers/volume21/19-248/19-248.pdf"
PALM_URL = "https://bolte.perso.math.cnrs.fr/BST2013.pdf"


class SpecPage:
    """Typeset short blocks in a single column with a fixed page rhythm."""

    def __init__(self, title, number):
        self.fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        self.text("2DETECT · CT FACTORIZATION PILOT", .952, 9, color=ACCENT, weight="bold")
        self.text(title, .917, 22, weight="bold")
        self.text(f"Experimental specification · {number} of 4", .048, 8, color=MUTED)
        self.y = .864

    def text(self, value, y, size, x=.09, **kwargs):
        self.fig.text(x, y, value, va="top", fontsize=size, color=kwargs.pop("color", INK),
                      **kwargs)

    def section(self, title):
        self.text(title, self.y, 12, weight="bold", color=ACCENT)
        self.y -= .027

    def paragraph(self, value):
        lines = wrap(value, width=89)
        self.text("\n".join(lines), self.y, 10, linespacing=1.35)
        self.y -= len(lines) * .0175 + .006

    def equation(self, value, size=13):
        self.text(value, self.y, size, x=.112)
        self.y -= .043

    def table(self, headings, rows, widths=None):
        height = .032 * (len(rows) + 1)
        ax = self.fig.add_axes([.09, self.y - height, .82, height])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headings, colWidths=widths,
                         bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#e1edf4" if row == 0 else "#f0f5f8" if row % 2 else "#f8fafb")
            cell.set_text_props(color=INK, weight="bold" if row == 0 else "normal")
        self.y -= height + .017

    def links(self, links):
        for label, url in links:
            self.text(label, self.y, 9, color=ACCENT, url=url)
            self.y -= .019

    def save(self, pdf):
        if self.y < .068:
            raise ValueError("Experimental specification exceeds the printable page")
        pdf.savefig(self.fig)
        plt.close(self.fig)


def metadata(root):
    return [json.loads(path.read_text()) for path in sorted((root / "runs").glob("*/metadata.json"))]


def objective_page(pdf):
    page = SpecPage("Objective and constraint geometry", 1)
    page.section("Composite SFRAMES template")
    page.equation(r"$\min_{x\in C}\ f(x)+g(Tx),\qquad x=(U,V),\qquad U,V\in\mathbb{R}^{512\times64}. $")
    page.paragraph(
        "The reconstructed attenuation image is X = UVᵀ on a 512 × 512 grid. Every "
        "measurement view of one slice constrains the same two factors. Increasing "
        "the number of views changes the loss terms and leaves the variables fixed.")
    page.equation(r"$C=\{(U,V):\ \Vert U\Vert_F\leq R_U,\ \Vert V\Vert_F\leq R_V\},$")
    page.equation(r"$D=\{(U,V):\ U\geq 0,\ V\geq 0\},\quad g=\iota_D,\quad T=\mathrm{Id}.$")
    page.paragraph(
        "The inequalities in D are entrywise. Its indicator g is zero on D and infinite "
        "elsewhere. C uses separate Frobenius balls. T acts on the factors and is the identity. "
        "The fan-beam projector acts inside f on their product.")
    page.section("Finite-sum smooth term f")
    page.equation(r"$q_i(U,V)=\frac{\Vert P_i(UV^\top)-y_i\Vert_2^2}{2d},\qquad d=956,$")
    page.equation(r"$f(U,V)=\frac{1}{N}\sum_{i\in\mathcal{I}_{\mathrm{train}}}q_i(U,V),"
                  r"\qquad N=|\mathcal{I}_{\mathrm{train}}|.$")
    page.paragraph(
        "Pᵢ is the calibrated fan-beam forward operator for a single angle. yᵢ contains "
        "the corresponding binned, log-corrected low-dose measurements. Each view and "
        "each detector bin have equal weight. Bilinearity makes the factor-space objective "
        "nonconvex.")
    page.section("Sampling and gradient")
    page.paragraph(
        "Each SFRAMES update draws a fresh, uniform subset B of 64 training views without "
        "replacement within the batch and independently across updates. All 956 detector bins "
        "in each selected view contribute. The batch gradient at a fixed iterate is unbiased "
        "for ∇f. Momentum averages these batch gradients.")
    page.equation(r"$E_B=\frac{\sum_{i\in B}P_i^\ast(P_i(UV^\top)-y_i)}{64d},"
                  r"\qquad \widehat{\nabla f}_B=(E_BV,\ E_B^\top U).$", 12)
    page.paragraph(
        "Each joint gradient computes one subset forward projection and one subset adjoint. "
        "The polynomial loss is smooth on compact C. Global smoothness on the ambient "
        "space would require an extension outside a neighborhood of C.")
    page.save(pdf)


def data_page(pdf, original):
    page = SpecPage("Measured CT data and view splits", 2)
    page.section("2DeteCT acquisitions")
    page.paragraph(
        "We fit real low-dose Mode 1 measurements from the 2DeteCT fan-beam collection. "
        "A separately acquired high-dose Mode 2 reconstruction supplies an image-quality "
        "reference for each slice.")
    page.table(("Phase", "Slice IDs", "Training views", "Role"),
               [("Development", "04300", "360", "Select one rate per method"),
                ("Evaluation", "04531, 04800", "360 and 1,080", "Reuse selected rates")],
               [.18, .21, .20, .41])
    page.section("Preprocessing and geometry")
    page.paragraph(
        "Starting from the 3,601 × 1,912 Mode 1 sinogram, discard the repeated final "
        "angle and sum adjacent detector cells. Apply the same binning to the dark field "
        "and the mean of the two flat fields. Apply −log to the dark-subtracted "
        "transmission ratio after clipping the ratio below at 10⁻⁶, then reverse the "
        "detector axis. The clipping count is zero on these slices.")
    page.paragraph(
        "ASTRA 2.5.0 uses its CPU line_fanflat projector with physical lengths in mm. "
        "Source-to-origin is 431.019989 mm, origin-to-detector 97.980499 mm, binned "
        "detector spacing 0.1496 mm, and field width 116.528143 mm. The field is centered "
        "at (−0.113797, −0.113797) mm. Angle i is π − 2πi/3600.")
    page.paragraph(
        "Align the Mode 2 reference by vertically flipping its published 1024 × 1024 "
        "TIFF, dividing by the native 0.113797 mm voxel width and averaging 2 × 2 blocks "
        "onto the fitting grid. Evaluate relative image error against this reconstruction.")
    page.section("Exact view indices")
    page.paragraph(
        "Indices run from 0 to 3599. The 360-view training set uses i mod 10 = 0. "
        "The 1,080-view training set uses residues 0, 3, and 6. All runs hold out "
        "residue 9 for validation, exactly 360 views. Calibration uses i mod 30 = 0 "
        "and therefore the same 120 training angles at both view counts.")
    page.section("Data provenance")
    page.paragraph(
        "The processed inputs and TIFF-level checksums are recorded in the experiment’s "
        "data/slice*.json sidecars. Slices 04300 and 04800 came from the 2DeteCT Zenodo "
        "archives. Slice 04531 came from DeepInverse’s single-slice mirror.")
    row = json.loads((original / "data/slice04531.json").read_text())
    page.links([("2DeteCT data paper", DATASET_URL),
                ("DeepInverse preprocessing and geometry example", PROJECTOR_URL),
                ("Slice 04531 single-slice archive", row["raw_source"])])
    page.save(pdf)


def schedules_page(pdf, runs):
    page = SpecPage("SFRAMES schedules and paired starts", 3)
    page.section("Open-loop updates")
    page.equation(r"$\gamma_k=\gamma_0(k+1)^{-3/4},\quad \rho_k=(k+1)^{-1/2},"
                  r"\quad \mu_k=\mu_0(k+1)^{-1/4}.$")
    page.paragraph(
        "Iteration k starts at zero. Use the first sampled gradient to initialize momentum. "
        "Thereafter blend the current batch gradient into momentum with weight ρₖ. Add "
        "the Moreau penalty gradient after the momentum update and apply the Frank–Wolfe "
        "convex combination.")
    page.equation(r"$s_k=\mathrm{LMO}_C\!\left(m_k+\frac{x_k-\Pi_D(x_k)}{\mu_k}\right),"
                  r"\qquad x_{k+1}=(1-\gamma_k)x_k+\gamma_k s_k.$", 12)
    page.paragraph(
        "Πᴅ clips negative entries. For a block direction G and radius R, the Frobenius "
        "ball LMO returns −RG/‖G‖F for nonzero G and zero when G is zero.")
    page.section("Slice-level calibration and rates")
    rows = []
    for slice_id in ("slice04300", "slice04531", "slice04800"):
        run = next(item for item in runs if item["config"]["slice_id"] == slice_id
                   and item["config"]["method"] == "sframes")
        rows.append((slice_id[-5:], *(f"{value:.8f}" for value in run["caps"]),
                     *(f"{value:.8f}" for value in run["curvature"]),
                     f"{run['optimizer']['mu0']:.8f}",
                     "0.10 and 0.25" if slice_id == "slice04300" else "0.25"))
    page.table(("Slice", "Rᵤ", "Rᵥ", "Lᵤ", "Lᵥ", "μ₀", "γ₀"), rows,
               [.105, .145, .145, .145, .145, .145, .17])
    page.paragraph(
        "For the feasible start, divide the positive backprojection by the sensitivity "
        "P*P1 floored at 10⁻³ of its maximum. Factor this image with rank-64 NNDSVD. "
        "Add 10⁻³ times each factor’s RMS times independent uniform [0,1) draws. Fit a "
        "common least-squares amplitude to the calibration views with floor 10⁻⁶, then "
        "scale each factor by its square root. Each cap is twice the resulting factor norm.")
    page.paragraph(
        "Estimate the image-space curvature λ using 10 power steps on P*P for the 120 "
        "calibration views and divide by 120d. Use Lᵤ = λ‖Vcal‖op² and Lᵥ = λ‖Ucal‖op², "
        "then set μ₀ = 1/max(Lᵤ,Lᵥ). These are empirical initial curvature scales.")
    page.section("Paired initialization and random streams")
    page.paragraph(
        "The rerun keeps the original caps and calibration scales. Draw two independent "
        "standard Gaussian block directions and apply one Frobenius LMO to each. This "
        "starts on the boundary of C and outside D. It also removes the original data-informed "
        "warm start. Both variants start with empty gradient momentum.")
    page.paragraph(
        "Base seed is 110 for development and 201 for evaluation. NumPy's independent "
        "minibatch stream uses base + 100,000 in both variants. The LMO direction stream "
        "uses base + 200,000. The two evaluation view counts share the same LMO factors "
        "within each slice.")
    page.paragraph(
        "Schedule source: Improved convergence and stochastic estimation for the "
        "Frank-Wolfe with Moreau Envelope Smoothing (FraMES) algorithm, local manuscript "
        "dated 27 September 2026, Corollary 2.10 and Proposition 2.11.")
    page.save(pdf)


def comparison_page(pdf):
    page = SpecPage("Work budgets and comparator settings", 4)
    page.section("Run budgets and diagnostics")
    page.table(("Phase", "Target views", "Probe spacing", "SFRAMES endpoint"),
               [("Development", "5,760", "720 views", "90 updates / 5,760 views"),
                ("Evaluation", "21,600", "2,160 views", "338 updates / 21,632 views")],
               [.20, .23, .20, .37])
    page.paragraph(
        "Stop after the first completed update reaching the target. Save at iteration 0, "
        "after update 1, when crossing a probe threshold and at the endpoint. A forward "
        "view and a backprojected view are counted separately. Optimizer time includes "
        "projections and oracle calls and excludes calibration, diagnostics and checkpoint I/O.")
    page.paragraph(
        "Feasibility repair clips negative entries and rescales each block if needed. "
        "Repaired training loss, held-out loss and reference error use those factors. Raw "
        "loss uses the actual iterate. Cap excess and distance to D describe the raw factors. "
        "Diagnostics leave optimizer state unchanged.")
    page.section("Original four-method pilot")
    page.paragraph(
        "All four methods share the original feasible warm start. Each development grid "
        "had two rate multipliers. The winner had the smaller repaired training loss at "
        "the last probe within 5,760 forward views. The LMO ablation changes only "
        "SFRAMES’ initialization and keeps the original rates.")
    page.table(("Method", "Development rates", "Selected", "Step rule"),
               [("SFRAMES", "0.10, 0.25", "0.25", "Rate is γ₀. ρ₀ = 1."),
                ("SPRING-SARAH", "0.30, 0.70", "0.70", "Block steps rate/Lᵤ, rate/Lᵥ."),
                ("ProxSARAH", "0.40, 0.90", "0.90", "Joint step rate/(Lᵤ + Lᵥ)."),
                ("PALM", "0.60, 1.10", "1.10", "Block steps rate/Lᵤ, rate/Lᵥ.")],
               [.22, .25, .15, .38])
    page.paragraph(
        "SPRING-SARAH alternates blocks with independent full refresh probability 1/16 "
        "per block and uses 64-view SARAH differences. ProxSARAH uses a joint step, "
        "256-view snapshots, 12 recursive updates per snapshot and relaxation 0.8. "
        "Its recursive batches also have 64 views. PALM uses the full training set for "
        "each block step. These are the practical settings tested on this pilot.")
    page.paragraph(
        "Elite ran ASTRA forward and adjoint operations in FP32 with FP64 factor arithmetic "
        "and one BLAS thread per worker. Timing reflects concurrent arms on separate "
        "physical cores of the same host.")
    page.links([("SPRING, Algorithm 1.1", SPRING_URL),
                ("ProxSARAH, Algorithm 1 and §3.2", PROXSARAH_URL),
                ("PALM, original paper", PALM_URL)])
    page.save(pdf)


def add_problem_spec(pdf, original_root, rerun_root):
    """Prepend the frozen pilot specification to the paired-run plots."""
    original = Path(original_root)
    runs = metadata(original)
    objective_page(pdf)
    data_page(pdf, original)
    schedules_page(pdf, runs)
    comparison_page(pdf)
