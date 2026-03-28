# AGENTS.md — Coding Agent Guidelines for hs-trains

This file provides authoritative guidance for agentic coding tools (Claude Code, Copilot, Cursor, etc.)
working in this repository.

---

## Repository Overview

hs-trains is a **Discrete Event Simulator (DES)** for train networks written in Rust (edition 2024),
with Python helper scripts. It simulates physics-based and timing-based trains over a shared event queue.

---

## Build / Run / Test Commands

### Rust

```bash
# Build
cargo build --release

# Run
cargo run --release -- <config.yaml> <output.parquet>

# Run all tests
cargo test

# Run a single test by name (substring match)
cargo test test_braking

# Run a single test — exact name
cargo test test_braking -- --exact

# Show stdout/println output during tests
cargo test -- --nocapture

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt

# Check formatting without modifying
cargo fmt -- --check

# Coverage (HTML report → target/llvm-cov/html/)
cargo llvm-cov --html
```

### Python (helper scripts only — not part of the simulator binary)

```bash
# Run any script
uv run scripts/make_timing_parquet.py
uv run scripts/run_100trains.py
uv run scripts/plot.py

# Run Python tests
uv run pytest

# Run a single Python test
uv run pytest python/tests/test_rollingstock.py::TestValueTable::test_foo
```

---

## Project Structure

```
src/
  main.rs       — CLI (clap), config loading, event loop, Rayon parallelism, Parquet output
  scheduler.rs  — Min-heap EventQueue with lazy token-based cancellation
  physics.rs    — Davis equation model, terminal velocity (bisection), kinematics
  timing.rs     — Parquet berth-trace loader, binary-search + linear interpolation
  model.rs      — Core POD structs (TrainDescription, SimulatedState, DriverInput, …)
  assets.rs     — Empty stub; placeholder for future signal/infrastructure logic
config/         — YAML simulation configs (physics, timing, mixed)
python/         — Helper scripts and RailML pydantic-xml models
railml/         — RailML 3.3-SR1 XSD reference schemas (read-only)
```

`assets.rs` is not yet wired into `main.rs` (`mod assets` is absent). Do not add it without a purpose.

---

## Code Style — Rust

### Naming

- `snake_case` — variables, functions, modules, struct fields, file names
- `PascalCase` — structs, enums, traits, type aliases
- `SCREAMING_SNAKE_CASE` — constants (`const G: f64 = 9.81`)
- Prefer descriptive names; avoid single-letter names except in tight math loops (`v`, `a`, `dt`, `f`)

### Formatting

- Default `rustfmt` settings (no `rustfmt.toml`). Always run `cargo fmt` before committing.
- Use numeric literals with underscores for readability: `1_000_000`, `0.001_f64`
- Spaces around binary operators: `v0 + a * dt`, not `v0+a*dt`
- Struct literals: use standard field-per-line format for structs with more than two fields

### Imports

- Order: standard library → third-party crates → local (`crate::`) modules
- Separate each group with a blank line
- Use explicit paths from `crate::` for internal modules, e.g. `use crate::model::TrainDescription`
- Avoid glob imports (`use foo::*`) except inside `#[cfg(test)]` modules where `use super::*` is idiomatic

### Types

- `f64` for all floating-point physics values; never use `f32` in the simulator core
- `i64` for timestamps in milliseconds (Parquet schema compatibility)
- `usize` for indices and counts
- `Option<f64>` for nullable output columns (position/speed/acceleration absent for timing trains)
- Annotate all public function signatures explicitly; avoid relying on type inference for public API

### Serde / Config

- Use the attribute path style: `#[derive(serde::Deserialize)]`, not `use serde::Deserialize`
- Enums use `#[serde(tag = "kind", rename_all = "snake_case")]` for internally-tagged dispatch:
  the `"kind"` field in YAML selects the enum variant, and `rename_all` maps `PascalCase` variants
  to `snake_case` strings (e.g. `PhysicsConfig` → `"physics"`)
- Provide default values via named functions: `fn default_flush_rows() -> usize { 1_000_000 }`
  combined with `#[serde(default = "default_flush_rows")]`. Serde requires a function path here —
  inline expressions are not supported — so this is the standard idiom for optional config fields
  with sensible fallbacks
- Config YAML field names mirror Rust struct field names exactly (all `snake_case`)

### Error Handling

- **Binary (`main.rs`)**: fail fast on startup errors using the pattern:
  ```rust
  some_fallible_call().unwrap_or_else(|e| {
      eprintln!("Fatal: {e}");
      std::process::exit(1);
  });
  ```
  Do **not** use `.unwrap()` or `.expect()` for I/O or config errors in the binary — always give a message.
- **Library functions** (`timing.rs`, etc.): propagate errors with `?`, returning `PolarsResult<T>`
- Use `unreachable!()` only for genuinely impossible enum arms; add a comment explaining why

### Logging and Output

- There is currently no logging crate (`tracing`, `log`, `env_logger`) in the project. This is a
  known gap — adding one is expected as the codebase grows
- For now: `println!` for informational simulation output, `eprintln!` for errors before process
  exit, and `pb.println(...)` via `indicatif` for progress-bar-compatible messages in the hot loop

### Parallelism

- All physics-train advancement is parallelised with **Rayon** `par_iter_mut`
- The codebase is fully synchronous — do **not** introduce `async`/`await` or `tokio`
- Keep shared state minimal; prefer passing data through the event queue

### Dead Code and Stubs

- `#![allow(dead_code)]` is present in `model.rs` and `scheduler.rs` because many fields are
  placeholders for future event-driven logic. This is deliberate — do not remove these attributes.
- `#[allow(dead_code)]` on individual functions in `physics.rs` serves the same purpose
- When implementing a previously-stubbed feature, remove the corresponding `allow` attribute

---

## Code Style — Python

### Naming

- `snake_case` for variables, functions, modules
- `PascalCase` for classes
- Type annotations required on all public function signatures (PEP 484)

### Dependencies

- Managed via `pyproject.toml` with `hatchling` build backend
- Run scripts through `uv run <script>` — do not assume a global Python environment
- Key libraries: `polars`, `pydantic-xml`, `pyarrow`, `pytest`

### Tests

- Use `pytest`; tests live in `python/tests/`
- Tests are organised into classes (`class TestFoo:`), which groups related tests under a shared
  namespace — e.g. `TestValueTable` contains all tests for the `ValueTable` type. pytest discovers
  and runs them automatically; there are no shared fixtures on the class itself
- Module-level helper functions provide shared test data (no `@pytest.fixture` required for simple cases)
- `pythonpath = ["python"]` is configured in `pyproject.toml` — imports resolve from `python/`

---

## Testing Guidelines

### Rust Tests

- Unit tests live in a `#[cfg(test)] mod tests { use super::*; ... }` block at the **bottom** of the
  relevant source file (see `physics.rs` for the canonical example)
- Use factory functions for test fixtures: `fn test_params() -> TrainDescription { ... }`
- Tolerance-based numeric assertions with a helper (`assert_within`) rather than exact equality
- Current tolerances in physics tests: speed ±1.0 m/s, position ±5.0 m (Euler vs closed-form)
- No integration tests directory yet — add `tests/` at the crate root when needed
- Avoid `#[should_panic]`; return `Result` from tests where error paths need checking

### Adding New Tests

1. Write the test in the `#[cfg(test)]` block of the module under test
2. Verify with `cargo test <test_name> -- --nocapture`
3. Ensure `cargo test` (all tests) still passes

---

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- `cargo test`
- `cargo llvm-cov --html` (uploads coverage artifact)
