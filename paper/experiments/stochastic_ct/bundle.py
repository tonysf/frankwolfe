"""Validate or rebuild the captured CT report using only its bundled figures."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit


BUNDLE = Path(__file__).resolve().parent
STEM = "sframes_ct_section"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest paths must be relative to the report bundle")
    path = root / relative
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("A report-bundle path escapes its directory") from error
    return path


def local_markdown_links(path):
    for target in re.findall(r"\]\(([^\n)]+)\)", Path(path).read_text(encoding="utf-8")):
        parsed = urlsplit(target)
        if not parsed.scheme and not parsed.netloc and parsed.path:
            yield target, unquote(parsed.path)


def validate_bundle(root=BUNDLE, require_archive=False):
    """Check captured hashes, local LaTeX dependencies and relocated doc links.

    Archive links can be checked in a complete repository checkout; validation
    and rebuilding of the report itself also work with this directory alone.
    """
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported report-bundle manifest")
    if manifest.get("snapshot", {}).get("comparison") != "gradient-only view budget":
        raise ValueError("The snapshot is not the completed gradient-budget comparison")
    files = manifest["files"]
    for name, record in files.items():
        path = contained_path(root, name)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Report bundle file differs from its manifest: " + name)
        if record.get("copy_mode") == "byte-identical" and record["sha256"] != record["original_sha256"]:
            raise ValueError("A captured source digest differs: " + name)
    report = root / "report"
    driver = (report / (STEM + ".tex")).read_text(encoding="utf-8")
    body = (report / "ct_experiments.tex").read_text(encoding="utf-8")
    for source in (driver, body):
        for name in re.findall(r"\\input\{([^}]+)\}", source):
            dependency = contained_path(report, name)
            if not dependency.is_file():
                raise ValueError("Missing bundled LaTeX input: " + name)
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    if figures != ["trajectories_360.pdf", "trajectories_1080.pdf"]:
        raise ValueError("The report must contain the two captured trajectory figures")
    if body.count(r"\begin{figure}") != 2 or body.count(r"\begin{table}") != 1:
        raise ValueError("The captured report's figure/table count changed")
    if set(re.findall(r"\\bibitem\{([^}]+)\}", driver)) != {"detect", "spring", "proxsarah", "palm"}:
        raise ValueError("The self-contained report bibliography changed")
    link_source = (report / "ct_reproducibility_link.tex").read_text(encoding="utf-8")
    if "{../CT_REPRODUCIBILITY.md}" not in link_source or not (root / "CT_REPRODUCIBILITY.md").is_file():
        raise ValueError("The report's reproducibility link is not portable")
    for name in (STEM + ".pdf", *figures):
        if not (report / name).read_bytes().startswith(b"%PDF-"):
            raise ValueError("A captured PDF is invalid: " + name)
    internal_links, archive_links = 0, 0
    for filename in ("README.md", "CT_REPRODUCIBILITY.md"):
        source_path = root / filename
        for target, relative in local_markdown_links(source_path):
            destination = (source_path.parent / relative).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                expected = "../../../results/stochastic_matrix_factorization/ct/"
                if not relative.startswith(expected):
                    raise ValueError("Unexpected external local document link: " + target)
                archive_links += 1
                if require_archive and not destination.exists():
                    raise ValueError("Missing archived evidence link: " + target)
            else:
                internal_links += 1
                if not destination.exists():
                    raise ValueError("Missing bundled document link: " + target)
    return {"status": "verified", "snapshot_date": manifest["snapshot"]["date"],
            "files": len(files), "figures": 2, "tables": 1,
            "internal_document_links": internal_links, "archive_document_links": archive_links,
            "archive_links_verified": bool(require_archive),
            "captured_pdf_sha256": files["report/" + STEM + ".pdf"]["sha256"]}


def build_report(output, root=BUNDLE, compiler="latexmk"):
    """Rebuild into a new directory without overwriting the captured PDF."""
    root = Path(root).resolve()
    validate_bundle(root)
    output = Path(output).resolve()
    if output in (root, root / "report"):
        raise ValueError("The captured report directory cannot be a build destination")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty build directory: " + str(output))
    command = shutil.which(compiler)
    if command is None:
        raise RuntimeError("latexmk is required; install a local LaTeX distribution before rebuilding")
    destination = output / "report"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (root / "report").iterdir():
        if source.name != STEM + ".pdf":
            shutil.copyfile(source, destination / source.name)
    shutil.copyfile(root / "CT_REPRODUCIBILITY.md", output / "CT_REPRODUCIBILITY.md")
    log = output / "build.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([command, "-pdf", "-interaction=nonstopmode", "-halt-on-error", STEM + ".tex"],
                       cwd=destination, stdout=stream, stderr=subprocess.STDOUT, check=True)
    pdf = destination / (STEM + ".pdf")
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("LaTeX did not produce the report PDF; inspect " + str(log))
    source_log = destination / (STEM + ".log")
    if source_log.exists():
        contents = source_log.read_text(encoding="utf-8", errors="replace")
        if r"Overfull \hbox" in contents or "LaTeX Warning: There were undefined references" in contents:
            raise RuntimeError("The rebuilt report has unresolved references or overflowing text; inspect " + str(log))
    return {"status": "rebuilt", "pdf": str(pdf), "build_log": str(log),
            "captured_pdf_unchanged": validate_bundle(root)["status"] == "verified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate the captured report (default)")
    mode.add_argument("--build", action="store_true", help="compile a separate PDF from the bundled sources and figures")
    parser.add_argument("--require-archive", action="store_true", help="also check links to the repository's archived evidence")
    parser.add_argument("--output-dir", type=Path, default=BUNDLE / "build", help="new or empty build directory")
    args = parser.parse_args(argv)
    checked = validate_bundle(BUNDLE, require_archive=args.require_archive)
    result = build_report(args.output_dir) if args.build else checked
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
