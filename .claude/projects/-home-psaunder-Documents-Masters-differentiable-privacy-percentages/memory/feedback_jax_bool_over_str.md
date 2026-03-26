---
name: JAX pytree: use bool not str for static fields
description: Prefer bool over str for static config fields on eqx.Module to avoid JAX pytree issues
type: feedback
---

Use `bool` fields (e.g. `use_exp: bool`) rather than `str` fields (e.g. `positivity: str`) for static discriminator fields on `eqx.Module` subclasses.

**Why:** JAX's tree functions don't play well with strings in pytree structure (equinox treats static fields as compile-time constants, and strings can cause recompilation or tracing issues).

**How to apply:** When adding a flag to a schedule config/module that selects between two variants (e.g. softplus vs exp), use a bool field rather than a string enum.
