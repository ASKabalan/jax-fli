# 04 · SPMD basics: how jax-fli shards the pipeline

jax-fli runs the whole forward model — initial conditions → N-body → lightcone painting →
lensing — under JAX's **SPMD** model (single program, multiple data): one program runs on every
device, and each array is *partitioned* across a shared **device mesh** so the devices cooperate
on one large problem. The **same mesh shards every stage**; you pick its shape once, up front.

This page is the conceptual reference. For runnable, end-to-end examples see the two distributed
notebooks:

- [Distributed PM](05-Distributed-PM.ipynb) — sharded N-body: halo exchange and distributed
  painting to 3-D / flat-sky / HEALPix.
- [Weak Lensing](../2-advanced-usage/09-Lensing.ipynb) — sharded convergence (`born`) and shear
  (`get_shear`) on the sphere.

## Distributing over devices: the M / N convention

jax-fli shards every field over a **2-D device mesh** whose two axes have fixed *roles*, read
**positionally**:

- **M** — the **first** mesh axis — carries **pixels**: `NPIX` for HEALPix maps (and the first
  spatial dimension of a 3-D box).
- **N** — the **second** mesh axis — carries the **tomographic bins** of the convergence and
  shear.
- A leading **batch** axis `B` (when present) is always **replicated**.

With the usual axis names `("x", "y")` this reads **M = `"x"` = pixels** and **N = `"y"` = bins**.
The *same* mesh shards the whole pipeline:

| Field | Array shape | `PartitionSpec` | Sharded over |
|------|------|------|------|
| 3-D density | `(B,) X, Y, Z` | `P([None,] "x", "y", None)` | X → M, Y → N |
| flat density | `(B,) X, Y` | `P([None,] "x", "y")` | X → M, Y → N |
| spherical density | `(B,) NPIX` | `P([None,] "x")` | NPIX → M |
| convergence κ | `(B,) BINS, NPIX` | `P([None,] "y", "x")` | BINS → N, NPIX → M |
| shear (γ₁, γ₂) | `(B,) BINS, 2, NPIX` | `P([None,] "y", None, "x")` | BINS → N, NPIX → M |

`born` emits the convergence already in the `P("y", "x")` layout, and `get_shear` returns the
matching shear `P("y", None, "x")` (the two components are replicated).

> **The lensing layout lives on the array, not on `field_sharding`.** Every field keeps its static
> `field_sharding` at the **canonical 3-D mesh layout** (`P("x", "y")`); each field type's
> `apply_sharding()` derives the (possibly transposed) layout in the table above and applies it to
> `field.array`. So after `born`, `kappa.field_sharding.spec` is `P("x", "y")` while
> `kappa.array.sharding` is `P("y", "x")`. Storing the canonical layout (never the transposed one)
> is what lets the rest of the pipeline re-derive each field's layout without transposing twice.

### Choosing the mesh shape `(M_size, N_size)`

For `D` devices and `NBINS` tomographic bins (`M_size · N_size = D`):

| Mesh | NPIX | BINS | Use it for | What happens |
|------|------|------|------|------|
| `(D, 1)` | sharded ×D | replicated | **density-only** sims | bins not distributed — `born` warns |
| `(1, D)` | replicated | sharded ×D | **Avoid using it** | npix not sharded — `nbody` warns (spherical) |
| `(D // NBINS, NBINS)` | sharded | sharded | **lensing** (optimal) | each bin on its own N-slice, pixels split over the rest |
| `(k, n)`, `NBINS % n ≠ 0` | — | — | — | **raises** `ValueError` |

**Rule:** the N axis must divide the bin count — `NBINS % N_size == 0`.

- A **density-only** run wants `(D, 1)`: every device works on pixels, and a pure density field
  has no bins axis to fill.
- A **lensing** run wants `(D // n_bins, n_bins)`: pixels split across `D // n_bins` devices while
  each tomographic bin sits on its own slice of N.

### Why bins on N?

The spherical Kaiser–Squires κ→γ step (`get_shear`) uses a spherical-harmonic transform that is
**not pixel-partition-aware**. Sharding the **bins** over N lets each device run the transform on
its own subset of bins independently — no cross-device pixel exchange — which is exactly why
`(D // n_bins, n_bins)` is the efficient lensing layout. (Internally the transform is sealed inside
a `jax.shard_map` so each device keeps its `npix` whole; the bins-on-N split is what makes that a
no-communication, per-device transform.)

## Two things to know

### Ray-tracing returns un-sharded (host NumPy) maps

Only `born` produces a sharded convergence. `raytrace` (Dorian) is **NumPy + MPI** under the hood:
it all-gathers the lightcone to a full host array on every process, runs the ray-tracer there, and
returns a `SphericalKappaField` whose `array` is a replicated host array with
`field_sharding = None`. Treat ray-traced κ as single-device — there are **no sharded κ maps** from
ray-tracing. (If you need it distributed afterwards, attach a `field_sharding` and call
`field.apply_sharding()`.)

### `get_shear` transform backend (`method=`)

`SphericalKappaField.get_shear(method=...)` selects the spherical-harmonic transform backend, which
`jax_healpy` forwards to **[s2fft](https://astro-informatics.github.io/s2fft/)**:

| `method` | Backend | Use it for |
|------|------|------|
| `"jax"` (default) | pure-JAX s2fft transform | portable + differentiable; runs on CPU / GPU / TPU |
| `"jax_cuda"` | s2fft's custom CUDA HEALPix primitives | fastest on NVIDIA GPUs (GPU-only) |
| `"jax_healpy"` | JAX, healpy-compatible path | drop-in healpy-ordering parity |

The numerical result is the same across backends up to precision — only speed and hardware support
differ. See the [s2fft docs](https://astro-informatics.github.io/s2fft/) for the backend details
and accuracy trade-offs. (The same `method=` flows through `get_convergence`, `angular_cl`, and the
other HEALPix transforms.)
