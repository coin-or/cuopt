# libcuopt-client

Host-side cuOpt client: problem representation, parsers and the gRPC client. Contains no CUDA kernels.

This package ships one component of cuOpt's C++ library. Install [`libcuopt`](https://pypi.org/project/libcuopt/) instead to get all of them.

## Building against this package with CMake

The wheel ships headers and CMake package files (`find_package(cuopt)`, target `cuopt::client`).
Because cuOpt's wheels are split across separate site-packages directories, CMake needs to
be told where each one lives:

```bash
SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
# every wheel root (holds lib64/cmake/cuopt) and every directory holding a *-config.cmake
ROOTS=$(find "$SP" -mindepth 3 -maxdepth 3 -path '*/lib64/cmake' -printf '%h\n' | xargs -n1 dirname | sort -u | tr '\n' ';')
CFGS=$(find "$SP" \( -name '*-config.cmake' -o -name '*Config.cmake' \) -printf '%h\n' | sort -u | tr '\n' ';')
cmake -S . -B build -DCMAKE_PREFIX_PATH="$ROOTS$CFGS"
```

The client is a runtime and CMake leaf: `find_package(cuopt)` needs no CUDA Toolkit, `rmm` or
`raft` for a client-only install. The one exception is `cuopt/error.hpp`, which includes
`raft/core/error.hpp`; if you include it, install `libraft-cu13` and `librmm-cu13` as well.
Note that the problem parsers and solver headers ship with `libcuopt-mathopt`, not here.
