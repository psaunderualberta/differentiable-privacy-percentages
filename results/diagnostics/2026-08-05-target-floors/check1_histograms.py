"""Check 1 — class histograms of the loaded target arrays.

Loads each transfer target through the real transfer code path
(build_target_config -> both config scopes -> get_dataset_loader) and reports:
  * split sizes / shapes
  * full-file class histograms for train and the val/test pool
  * the eval-split majority fraction (the accuracy floor a collapsed model hits)
  * one-hot validity
  * pixel stats of a real preprocessed training batch
and dumps a small image grid per target with its decoded label for eyeballing.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src")

OUT = Path("/home/psaunder/.claude/jobs/ae8bba61/tmp/check1")
OUT.mkdir(parents=True, exist_ok=True)


def hist(labels_onehot, name):
    """Return (n, n_classes, counter of argmax, rowsum stats)."""
    y = np.asarray(labels_onehot)
    idx = y.argmax(axis=1)
    counts = Counter(idx.tolist())
    rowsums = y.sum(axis=1)
    return {
        "name": name,
        "n": int(y.shape[0]),
        "n_classes_declared": int(y.shape[1]),
        "n_classes_present": len(counts),
        "counts": counts,
        "rowsum_min": float(rowsums.min()),
        "rowsum_max": float(rowsums.max()),
    }


def report(h):
    print(
        f"  [{h['name']}] n={h['n']}  declared classes={h['n_classes_declared']}  "
        f"present={h['n_classes_present']}  one-hot rowsum in [{h['rowsum_min']}, {h['rowsum_max']}]"
    )
    top = h["counts"].most_common(12)
    for cls, c in top:
        print(f"      class {cls:>3}: {c:>8}  ({100.0 * c / h['n']:6.3f}%)")
    if h["n_classes_present"] > 12:
        print(f"      ... {h['n_classes_present'] - 12} more classes")
    maj = top[0][1] / h["n"]
    print(f"      MAJORITY FRACTION = {100.0 * maj:.4f}%")
    return maj


def run_target(target_name, eps=10.0, T=5000, batch_size=250):
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from util.dataloaders import get_dataset_loader
    from util.transfer import TargetSpec, build_target_config

    print(f"\n{'=' * 78}\nTARGET: {target_name}\n{'=' * 78}")
    spec = TargetSpec(name=target_name, eps=eps, delta=1e-7, T=T, arch="")
    config = build_target_config(spec, batch_size)

    with SingletonConfig.override(config), using(RunContext(config)):
        loader = get_dataset_loader()
        print(f"  n_train={loader.n_train}  n_val={loader.n_val}  n_test={loader.n_test}")
        print(f"  sample_shape={loader.sample_shape}  label_shape={loader.label_shape}")
        print(f"  x_path={loader.x_path}")
        print(f"  val_x_path={loader.val_x_path}")

        # Full-file histograms straight off the cached label arrays.
        y_train = np.load(loader.y_path, mmap_mode="r")
        y_valfile = np.load(loader.val_y_path, mmap_mode="r")
        print("\n  --- full-file label histograms ---")
        report(hist(y_train, "train file"))
        report(hist(y_valfile, "val file (raw, pre val/test split)"))

        # The split the reported accuracy is actually measured on, through the
        # real permuted-chunk accessor.
        print("\n  --- eval splits via the real loader accessors ---")
        val_idx = np.arange(loader.n_val)
        _, yv = loader.load_val_chunk(val_idx)
        maj_val = report(hist(yv, "val split (loader.load_val_chunk)"))
        if loader.n_test:
            _, yt = loader.load_test_chunk(np.arange(loader.n_test))
            report(hist(yt, "test split (loader.load_test_chunk)"))

        # A real preprocessed training batch: pixel stats + label sanity.
        rng = np.random.default_rng(0)
        bidx = np.sort(
            rng.choice(loader.n_train, size=min(batch_size, loader.n_train), replace=False)
        )
        xb, yb = loader.load_train_batch(bidx)
        xb = np.asarray(xb)
        print("\n  --- preprocessed train batch ---")
        print(f"      x shape={xb.shape} dtype={xb.dtype}")
        print(
            f"      x min={xb.min():.4f} max={xb.max():.4f} mean={xb.mean():.4f} std={xb.std():.4f}"
        )
        n_const = int((xb.reshape(xb.shape[0], -1).std(axis=1) == 0).sum())
        print(f"      constant-valued samples in batch: {n_const}/{xb.shape[0]}")
        report(hist(yb, "train batch labels"))

        _dump_images(target_name, xb, np.asarray(yb))
        return maj_val


def _dump_images(target_name, xb, yb):
    """Save a 4x4 grid of real preprocessed samples annotated with their labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(16, xb.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img = xb[i]
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        vmin, vmax = float(img.min()), float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(f"y={int(yb[i].argmax())}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{target_name}: preprocessed samples + labels")
    fig.tight_layout()
    path = OUT / f"{target_name}_samples.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"      wrote {path}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["eyepacs", "imagenet", "chexpert"]
    summary = {}
    for t in targets:
        try:
            summary[t] = run_target(t)
        except Exception as exc:  # keep going; one broken target shouldn't hide the others
            import traceback

            traceback.print_exc()
            summary[t] = f"FAILED: {type(exc).__name__}: {exc}"
    print(f"\n{'=' * 78}\nSUMMARY — val-split majority fraction (the collapse floor)\n{'=' * 78}")
    for t, v in summary.items():
        print(f"  {t:>12}: {v if isinstance(v, str) else f'{100.0 * v:.4f}%'}")
