# cuvs_build agent instructions

## Purpose

This repository manages local cuVS compilation, runtime packaging, and execution
of tests and benchmarks on RunPod. The main cuVS source checkout is a separate
repository selected by `CUVS_SOURCE_DIR` in the ignored local `.env` file.
RunPod executes the uploaded binaries; it does not build cuVS.

## Protect the main cuVS repository

The convenience or objectives of `cuvs_build` must not interfere with the main
cuVS repository's source, existing builds, development environment, or user work.

- Implement workflow adaptations in `cuvs_build`: configuration, launchers,
  environment variables, packaging, validation, and documentation.
- Do not edit cuVS source, tests, CMake files, tracked ignore rules, or Git state
  merely to make this workflow work. Changes to the main repository must be
  explicitly requested by the user as part of the task.
- Preserve unrelated changes in both repositories. Do not revert, stage, commit,
  or otherwise absorb them into a workflow change.
- If an upstream defect affects this workflow, prefer an adaptation here that
  preserves the intended behavior and validation. Report the upstream defect
  separately; do not silently patch cuVS or disable checks to hide it.

## Isolate build and setup state

- Read `.env`, `config/build.json`, and `README.md` before changing the workflow.
  Resolve configured paths before writing, cleaning, configuring, or installing anything.
  Inspect setup and dependency-install commands for writes to shared state; a
  configured path does not make that state workflow-owned.
- Use the dedicated configured build directory, currently
  `${CUVS_SOURCE_DIR}/cpp/build-runpod`. Do not reconfigure, clean, or overwrite
  the main repository's existing `cpp/build` or another developer build directory.
- Reuse an existing toolchain only when it is compatible. Do not replace,
  upgrade, downgrade, or install packages into a shared environment merely to
  satisfy `cuvs_build`; use a separate workflow environment when changes are
  needed. The current configuration points to `${CUVS_SOURCE_DIR}/.conda-env`.
- Keep machine-specific checkout paths in the ignored `.env`, using placeholders
  in versioned configuration. Do not commit or bundle `.env`.
- Keep generated state, bundles, results, and temporary work in workflow-owned
  locations. Keep local artifacts out of cuVS Git status without editing its
  tracked ignore rules. Do not modify global shell or Git configuration.
- Keep runtime environment settings scoped to the workflow's subprocesses and
  generated launchers. Prefer existing cuVS interfaces, such as `CI=true` and
  `RAPIDS_DATASET_ROOT_DIR` for Vamana test data.
- Apply binary relocation and RPATH changes only to copied bundle files, never
  to original binaries or libraries in an existing build or shared environment.

## Verification

- Preserve the build, package, test, and reporting flow when changing integration
  points. A replacement check must still detect failures and record its result.
- Run `make check` for workflow code changes and appropriate focused checks for
  affected behavior. Documentation-only changes do not require a CUDA build.
- Inspect the final diff and the main cuVS repository's status to confirm that
  the task has not changed unrelated source or setup. Report the checks actually
  performed and any remaining limitations.
