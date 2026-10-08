# Isolated-install smoke tests

Install **one** nightly cuOpt component (`client`, `mathopt` or `routing`) by itself in a clean
container, build a small gtest against only that package, and run it. This mimics a user who
installs a single component rather than the `libcuopt` metapackage.

```bash
ci/isolated_install/run.sh <client|mathopt|routing> <pip|conda> [image]
```

`client` needs no GPU; `mathopt` and `routing` run with `docker run --gpus all`.

| File | Purpose |
|------|---------|
| `run.sh` | Host-side driver: picks the image and starts the container. |
| `container_pip.sh` | Inside the container: installs the nightly wheel from the RAPIDS nightly wheel index into a venv, asserts that only the expected cuopt wheels are installed, then builds and runs the gtest. |
| `container_conda.sh` | Same flow with the `rapidsai-nightly` conda channel in a miniforge container. |
| `tests/` | One sample gtest per component, linking only `cuopt::<component>`. |

Environment: `CUDA_MAJOR` (default `13`), `CUDA_IMAGE_TAG` (pip base image tag),
`NIGHTLY_WHEEL_INDEX` (override the wheel index).

## Known problems this is meant to catch

These were found while writing the tests (local runs; only `routing` on pip passed):

1. **mathopt wheel:** `cuopt_mathopt-targets.cmake` hardcodes the build-machine path
   `/usr/lib64/libcudss/13/./libcudss.so.0` in `INTERFACE_LINK_LIBRARIES`, so consumers fail to link.
2. **client wheel:** its CMake config (from the shared export set) requires `CUDAToolkit`/`nvcc`,
   `rmm` and `raft`, none of which the wheel declares as dependencies.
3. **conda component packages:** none ships `cuopt-config.cmake` (only the `libcuopt` metapackage
   does), so `find_package(cuopt)` fails after installing a single component.
4. **wheels:** components are split across separate wheel directories, so a consumer has to pass
   every wheel root and every config directory in `CMAKE_PREFIX_PATH` for `find_package(cuopt)` to
   resolve sibling components.

The workflow `.github/workflows/isolated-install-smoke.yaml` runs the matrix nightly.
