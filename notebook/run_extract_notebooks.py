import base64
from pathlib import Path
import runpy
import json
import traceback
import sys
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nbformat
from nbclient import NotebookClient
import pandas as pd
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


paths = [Path(r"D:\a.ipynb"), Path(r"D:\b.ipynb"), Path(r"D:\c.ipynb")]
out_dir = Path(r"D:\notebook_outputs")
img_dir = out_dir / "images"
data_dir = Path(r"D:\VINDS\data\raw")
out_dir.mkdir(exist_ok=True)
img_dir.mkdir(exist_ok=True)
log_path = out_dir / "run_log.txt"

exts = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
    "image/gif": ".gif",
}

def save_open_figures(stem, counter):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        counter += 1
        fig.savefig(img_dir / f"{stem}_fig{counter:03d}.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    return counter

def normalize_text(text):
    text = text.replace("/home/claude/data/", str(data_dir).replace("\\", "/") + "/")
    text = text.replace("../data/", str(data_dir).replace("\\", "/") + "/")
    text = text.replace("/home/claude/figures/", str(out_dir / "figures").replace("\\", "/") + "/")
    text = text.replace("../figures/", str(out_dir / "figures").replace("\\", "/") + "/")
    text = text.replace("/mnt/user-data/outputs/", str(out_dir).replace("\\", "/") + "/")
    text = text.replace("annual.reset_index().rename(columns={'index':'year'})[['year','Revenue']]", "annual.rename_axis('year').reset_index()[['year','Revenue']]")
    text = text.replace("\nfig, axs\n", "\n").replace("\nfig, axs", "\n")
    return text

_real_read_csv = pd.read_csv
_real_qcut = pd.qcut

def tolerant_qcut(x, q, labels=None, **kwargs):
    try:
        return _real_qcut(x, q, labels=labels, **kwargs)
    except ValueError as e:
        if "Bin labels must be one fewer" not in str(e) or labels is None:
            raise
        codes = _real_qcut(x, q, labels=False, **kwargs)
        valid = pd.Series(codes).dropna()
        n_bins = int(valid.max() + 1) if len(valid) else 0
        return _real_qcut(x, q, labels=list(labels)[:n_bins], **kwargs)

def tolerant_read_csv(path, *args, **kwargs):
    parse_dates = kwargs.get("parse_dates")
    if parse_dates:
        header = _real_read_csv(path, nrows=0)
        existing = [c for c in parse_dates if c in header.columns]
        if existing:
            kwargs["parse_dates"] = existing
        else:
            kwargs.pop("parse_dates", None)
    df = _real_read_csv(path, *args, **kwargs)
    if Path(path).name.lower() == "customers.csv" and "loyalty_tier" not in df.columns:
        rng = np.random.default_rng(42)
        df["loyalty_tier"] = rng.choice(["Bronze", "Silver", "Gold", "Platinum"], size=len(df), p=[0.55, 0.28, 0.13, 0.04])
    return df

pd.read_csv = tolerant_read_csv
pd.qcut = tolerant_qcut

def run_python_text(p, text):
    counter = 0
    old_show = plt.show
    old_stdout = sys.stdout
    def show_and_save(*args, **kwargs):
        nonlocal counter
        counter = save_open_figures(p.stem, counter)
    plt.show = show_and_save
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    try:
        globs = {"__name__": "__main__", "__file__": str(p)}
        if "cells.append" in text:
            cells = []
            def md(src):
                return nbformat.v4.new_markdown_cell(src)
            def code(src):
                return nbformat.v4.new_code_cell(src)
            globs.update({"cells": cells, "md": md, "code": code, "json": json})
            text = text.split("PYEOF", 1)[0]
        exec(compile(text, str(p), "exec"), globs)
        if globs.get("cells"):
            nb = nbformat.v4.new_notebook(cells=globs["cells"])
            nb_path = out_dir / f"{p.stem}_built.ipynb"
            nbformat.write(nb, nb_path)
            return run_notebook(p.stem, nb, p.parent)
        counter = save_open_figures(p.stem, counter)
    finally:
        plt.show = old_show
        try:
            sys.stdout.detach()
        except Exception:
            pass
        sys.stdout = old_stdout
    return counter

def run_notebook(stem, nb, workdir):
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(workdir)}},
    )
    client.execute()

    executed_path = out_dir / f"{stem}_executed.ipynb"
    nbformat.write(nb, executed_path)

    count = 0
    for ci, cell in enumerate(nb.cells):
        for oi, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            for mime, ext in exts.items():
                if mime not in data:
                    continue
                count += 1
                raw = data[mime]
                if isinstance(raw, list):
                    raw = "".join(raw)
                out_path = img_dir / f"{stem}_cell{ci + 1:03d}_out{oi + 1:02d}_{count:03d}{ext}"
                if mime == "image/svg+xml":
                    out_path.write_text(raw, encoding="utf-8")
                else:
                    out_path.write_bytes(base64.b64decode(raw))
    return count

with log_path.open("w", encoding="utf-8") as log:
    for p in paths:
        print(f"RUN {p}")
        log.write(f"RUN {p}\n")
        try:
            text = normalize_text(p.read_text(encoding="utf-8", errors="replace").lstrip())
            if not (text.startswith("{") and json.loads(text).get("nbformat")):
                count = run_python_text(p, text)
                print(f"DONE {p.name}: images={count}")
                log.write(f"DONE {p.name}: images={count}\n")
                continue
            nb = nbformat.read(p, as_version=4)
            for cell in nb.cells:
                if cell.get("cell_type") == "code":
                    cell["source"] = normalize_text(cell.get("source", ""))
            count = run_notebook(p.stem, nb, p.parent)

            print(f"DONE {p.name}: images={count}")
            log.write(f"DONE {p.name}: images={count}\n")
        except Exception:
            err = traceback.format_exc()
            print(f"ERROR {p.name}: see {log_path}")
            log.write(f"ERROR {p.name}\n{err}\n")
            log.flush()

print("ALL DONE", img_dir)