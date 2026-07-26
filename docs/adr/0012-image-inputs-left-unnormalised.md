# Image inputs left unnormalised

## Context

Every image dataset is scaled to [0,1] and never centered
(`util/dataloaders.py:43,46` — `x_raw.astype(np.float32) / 255.0`), while the tabular
`california` dataset *is* z-scored (`:670` — `(X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)`).
CIFAR-10's per-channel means are ≈0.49/0.48/0.45, so every input coordinate is positive.

This is not neutral under DP. Uncentered inputs make first-layer per-sample gradients
strongly correlated across coordinates and inflate their norms, so at a fixed clip `C` the
clipping binds harder — attenuating signal while the injected noise is unchanged.
Per-channel standardisation is the standard CIFAR recipe and is the cheapest available
lever on absolute accuracy.

The question arose while redesigning the CIFAR-10 arch axis, from the premise that CIFAR
"fails to learn". The cached sweep refuted that premise (MLPs 45–46%, CNNs 56–63% against
10% chance), but left open whether preprocessing should be fixed while the axis is being
re-run anyway.

## Decision

Leave the preprocessing as-is. Record the handicap rather than remove it.

## Consequences

- **The reported comparison is unaffected.** Learned and Constant runs share the
  preprocessing, so the Δacc the overlay reports — the robustness claim the arch axis
  exists to support — is invariant to this choice. Normalisation would shift the operating
  point, not the contrast.
- **Absolute accuracies are a floor, not a ceiling**, and must be presented that way. Any
  comparison against published DP-SGD numbers has to note that those use standardised
  inputs; ours do not. 56–63% on CIFAR-10 from scratch at ε=10 without pretraining or
  augmentation is a plausible operating point, but it is not the best this pipeline could do.
- **Changing it later invalidates far more than ADR 0010 does.** That fix invalidates CNN
  runs on the arch axis; this one would invalidate every image run ever performed — the
  T-sweep, the transfer matrix, and the SR-distilled equations all assume comparable
  inputs. Reversing this is a whole-project re-run and should be taken as a deliberate
  standalone decision, never as a rider on another sweep.
- The tabular/image inconsistency stays. It is defensible (`california` is a regression
  problem on unbounded features where scaling is load-bearing, images are already on a
  bounded common scale) but it is an inconsistency, and a reader comparing the two paths
  will notice.
