# Hive LNS improvement candidate

This branch contains the highest-scoring candidate from the completed
`cuopt-lns-repair-sgm1-101642q` experiment and the integration it requires.

- Base: `cpufj-stable`, commit `06a8c1ace653d430354a23fc972edc821f670757`.
- Winning snapshot: `e6079930983c4ca680ea84f960b01150`.
- Best recorded fitness: **4.837227815043089**.
- Metric: `exp(mean(log(1 + n_i))) - 1`, where `n_i` is the validated
  LNS new-best count for each of seven instances.
- Instances: highschool1-aigio, rail02, nursesched-medium-hint03,
  neos-3024952-loue, brazil3, dws008-01, roi5alpha10n8.
- Evaluation: seed 0, 60 seconds per instance, one L4 GPU, 14 CPU cores,
  64 GiB RAM, 13 solver threads plus the LNS worker.
- Search budget: 29 sandboxes, three hours; stochastic evaluator.

[Experiment dashboard](https://nv.platform.live.hiverge.ai/static/experiments/46655e89-5812-4bc7-b53a-9892336ae101/overview?organization_id=nvidia)

The heuristic lives in `cpp/src/mip_heuristics/lns_improvement.hpp`. It runs
on a dedicated worker using read-only copies of feasible population members.
The interface exposes CPUFJ and sub-MIP repair on private neighborhoods.
The frozen publication path checks submitted solutions against both the
current population best and the published global best under their locks.

The experiment's 874 frozen files were verified against the downloaded
snapshot before extraction. Repository formatting and copyright headers are
applied for this branch; the winning algorithm is otherwise unchanged.
Original winning header SHA-256:
`113aa30772ef9a256784491c0a784573dafcda719ec09e667b622d3303c4cf4d`.

The branch contains the runtime integration and benchmark incumbent tracing.
The downloaded experiment archive retains the full evaluation harness and
inputs. The recorded score comes from Hive's GPU evaluation; a local C++
smoke check separately exercises feasibility, immutable population inputs,
empty-pool cancellation, and mock repair callbacks.
