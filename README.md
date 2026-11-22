<div align="center">

# KV Cache Budget Lab

### Understand memory. Replay cache reuse. Inspect data movement.

A lightweight Python lab for **LLM KV cache memory budgeting** and **bounded CPU–GPU page residency simulation**.

**Python 3.10+ · No runtime dependencies · No GPU required · MIT licensed**

</div>

![Synthetic cache reuse and thrashing experiments](docs/demo.svg)

## Why this project?

Long-context inference can consume substantial KV cache memory. CPU offloading changes the problem: keeping fewer pages on a GPU can require repeatedly transferring data back from the CPU.

This project makes two questions easy to explore:

1. **How much KV memory does a workload need?** Estimate capacity from architecture, context length, batch size, precision, and page size.
2. **When does cache reuse reduce transfers?** Replay ordered page requests through a fixed-capacity LRU cache and compare with refetching at every step.

**Scope:** an analytical calculator and a synthetic cache simulator. It does not execute an LLM, implement attention, or measure GPU speed, power, energy, or carbon emissions. The included numbers are reproducible simulation outputs, not hardware benchmarks.

## Quick start

Download this repository and open a terminal in its root folder. Use Python 3.10 or newer:

```bash
python -m kv_cache_lab --config examples/config.json --trace examples/reuse_trace.json --output reports/reuse
```

Open `reports/reuse/report.html` in a browser. The command also writes a machine-readable `report.json`.

No installation, model download, API key, or GPU is required. If your system uses `python3`, substitute it for `python`.

Optional package installation:

```bash
python -m pip install -e .
kv-cache-lab --config examples/config.json --trace examples/reuse_trace.json --output reports/reuse
```

## Included experiments

Both experiments use a **four-page GPU residency cap**, a cold-start GPU cache, and **16 MiB per logical page**. A logical page bundles 128 token positions across every decoder layer and KV head in the synthetic architecture.

| Synthetic trace | Policy | Residency hits | H2D transferred | Peak resident pages |
| :--- | :--- | ---: | ---: | ---: |
| Repeated shared pages | Refetch every step | 0 / 24 | 384 MiB | 3 / 4 |
| Repeated shared pages | LRU reuse | 14 / 24 | 160 MiB | 4 / 4 |
| Five-page cyclic selection | Refetch every step | 0 / 20 | 320 MiB | 4 / 4 |
| Five-page cyclic selection | LRU reuse | 0 / 20 | 320 MiB | 4 / 4 |

For the reuse trace, LRU avoids **58.3% of simulated H2D bytes**. For the five-page cycle, every page misses under a four-page LRU cap in this access order. This demonstrates why a bounded cache can help one workload and provide no transfer reduction for another.

A residency hit is checked **when a page is accessed**. It does not mean the page was prefetched, that every selected page was resident at the beginning of the step, or that attention accuracy is preserved.

Run the second scenario:

```bash
python -m kv_cache_lab --config examples/config.json --trace examples/thrashing_trace.json --output reports/thrashing
```

Pre-generated reports for both traces are included under `examples/reports/`.

## Memory model

For a decoder with uniform KV layout across layers:

```text
KV bytes per token = 2 × layers × KV heads × head dimension × bytes per element
Raw KV bytes       = KV bytes per token × context tokens × batch size
```

The factor of two accounts for keys and values. Use **KV heads**, which can differ from query heads in grouped-query attention.

The default synthetic architecture has 32 layers, 8 KV heads, head dimension 128, and 2 bytes per KV element. This gives **128 KiB per token** and **4 GiB of KV storage** for four requests with 8,192 tokens each.

The planner reserves `parameters_billion × 10^9 × weight_bytes` for weights and subtracts a user-specified runtime reserve. It rounds each request's KV allocation up to complete pages. It reports whether the workload fits and the maximum equal context length for the configured batch size.

These are storage estimates. Quantization metadata, alignment, temporary allocations, and framework-specific overheads need additional accounting. A workload that fits this estimate can still run out of memory on hardware.

## Change the experiment

Edit `examples/config.json`:

| Setting | Meaning |
| :--- | :--- |
| `model.layers`, `kv_heads`, `head_dim` | Uniform decoder KV architecture |
| `model.kv_bytes` | Storage bytes per K or V element; e.g. 2 for FP16/BF16 |
| `workload.gpu_gib` | Device capacity in binary GiB |
| `workload.parameters_billion` | Parameter count in decimal billions |
| `workload.weight_bytes` | Average bytes per weight, excluding metadata |
| `workload.runtime_reserve_gib` | Your estimate of non-weight, non-KV memory |
| `workload.context_tokens`, `batch_size` | Equal context length and concurrent request count |
| `workload.page_tokens` | Token positions per logical page |
| `resident_pages` | Fixed GPU page cap for trace replay |

The workload budget and trace replay are separate experiments. The workload budget estimates full KV residency; the trace represents an arbitrary sequence of selected pages. It is not derived from actual model attention or required to visit the entire workload context.

Supply a trace as a JSON array of steps:

```json
[[0, 1, 2], [0, 1, 3], [0, 1, 4]]
```

Page IDs can be integers or strings. Order matters. Duplicate IDs within a step are consumed once. IDs must be unique across independent requests unless shared KV is intentional. Empty steps are allowed inside a non-empty trace.

## Residency rules

- CPU memory already contains an immutable copy of all requested pages.
- **LRU:** pages remain resident between steps; the least recently used page is evicted on a miss when the pool is full.
- **Refetch:** residency is cleared between steps, so each selected page is fetched again.
- Every miss transfers one full logical page from CPU to GPU.
- The pool never grows beyond `resident_pages`. Selections larger than the cap are consumed sequentially; this is not an implementation of multi-pass attention.
- The initial CPU copy, cache construction, new-token appends, D2H transfers, and prefetch scheduling are outside the simulation.

## Validation

```bash
python -m unittest discover -s tests -v
```

The 11 tests cover GQA/MHA memory scaling, per-request page padding, capacity overflow, LRU eviction order, hard residency caps, reuse and thrashing, malformed inputs, and report integration.

A GitHub Actions workflow is included for Python 3.10 and 3.12. It runs the tests and generates both example reports when enabled on GitHub. Its configuration follows the [GitHub Python workflow documentation](https://docs.github.com/en/actions/tutorials/build-and-test-code/python). The workflow has not yet been run in a published repository.

## Project map

| Path | Contents |
| :--- | :--- |
| `kv_cache_lab/core.py` | Memory equations and bounded cache replay |
| `kv_cache_lab/cli.py` | JSON input, CLI validation, and report generation |
| `kv_cache_lab/report.py` | Self-contained HTML dashboard |
| `examples/` | Two traces, a configuration, and generated reports |
| `tests/` | Standard-library unit and integration tests |
| `docs/METHODOLOGY.md` | Assumptions and interpretation |
| `docs/PUBLISHING.md` | Beginner-friendly GitHub publishing instructions |
| `scripts/make_preview.py` | Regenerate the README figure from report JSON |

## Possible extensions

Future work could add measured request traces, transfer timings, prefetch policies, per-layer residency, and hardware energy measurements. These features are not implemented in version 0.1.0.

## Author and development

**Ali Haider** — PhD researcher in Computer Science, University of Leicester.

Research interests: efficient LLM inference, KV cache management, physics-informed machine learning, and energy-efficient data centres.

This initial educational implementation and documentation were developed with AI assistance. Its claims are limited to the analytical model and included synthetic experiments; it is not a reproduction of a published attention method.

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Released under the [MIT License](LICENSE).
