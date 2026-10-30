# Review Rules — Pre-PR Self-Review and CodeRabbit

One rule set for two consumers: the author (human or agent) before `git push`, and CodeRabbit
during review. Each rule has a stable ID so a review comment can cite it (`COMMENT-1`).

**How to use**

- **Author / agent:** run through this list against `git diff origin/main...HEAD` before pushing.
- **CodeRabbit:** raise a violation as a short nit that cites the rule ID. Do not raise anything
  that is not listed here as style, and do not restate rules enforced by `clang-format` or `ruff`.
- **Reviewer:** cite the ID instead of re-explaining. If a rule is wrong or missing, change this
  file in the same PR.

When neighbouring code in the same file does the opposite of a rule, ask rather than assume
which one is right.

## Comments

- **COMMENT-1 — Describe the current code, not its history.** No "used to be", "previously",
  "earlier version", "fixes the bug where", "load-bearing", or accounts of failed attempts. No
  numbers taken from one benchmark instance. Comments are for readers of the code as it is now.
- **COMMENT-2 — No volatile details.** No line numbers, commit hashes, PR numbers or timestamps.
  See [conventions.md](conventions.md#comments).
- **COMMENT-3 — Short and plain.** One or two lines saying what the code does or why, not prompts
  or notes addressed to an agent. A comment that restates the next line should be deleted. Define
  any jargon you use (ping-pong buffer, head, endpoint) or use the plain name.
- **COMMENT-4 — Numerics get the math.** A non-trivial numerical block states what the inputs are
  and what is computed (for example which residual, which row/coefficient).

## Terminology and naming

- **TERM-1 — Two problem spaces.** The *user problem* is the original; the *presolved problem* is
  what the solver sees after presolve. Do not invent others ("solver space", "barrier space",
  "internal objective", "user-space").
- **TERM-2 — Project vocabulary.** "barrier", not "IPM"; "problem", not "model"; "multi-GPU", not
  "distributed"; "MIP", not "MILP"; `mathematical_optimization`, not `linear_programming`, for new
  names; "dual residual", not "stationary residual".
- **TERM-3 — Spell names out.** No non-standard abbreviations in identifiers (`vub`, `fin`,
  `pb`); write `variable_upper_bound`. Name for what it is: a CSC structure is not "CSR".
- **TERM-4 — Consistent casing across an API.** Pick `MIP`/`LP`/`QP`/`cuOpt` once and use it in
  every identifier, doc and example of that API.

## Types, ownership, control flow

- **TYPE-1 — `i_t` / `f_t`.** Templated code uses `i_t`/`f_t`, not bare `int`/`double`, and
  `const i_t x`, not `i_t const x`. Template instantiation follows the existing `#ifdef` pattern
  in the neighbouring files.
- **TYPE-2 — No gratuitous casts.** Drop `static_cast` that does not change the value. Index with
  the signed type. See [conventions.md](conventions.md#container-indexing--no-gratuitous-static_castsize_t).
- **TYPE-3 — Ownership.** `std::unique_ptr` by default; `std::shared_ptr` only with a comment
  saying why sharing is essential. Prefer `std::span` over pointer + size. No raw owning
  pointers.
- **TYPE-4 — No large lambdas or hidden control flow.** Short predicates only; no scope guards
  unless commented (see [conventions.md](conventions.md#c-implementation-style)). A condition
  used in two places is one named function.
- **TYPE-5 — Enums and named constants.** Modes are an `enum class`, not 0/1/2. Limits such as
  `1e20` and tolerances are named constants or come from settings, not literals. A `switch` over
  an enum or status has no `default:`, so adding a value is a compile error.
- **TYPE-6 — Invariants are asserts.** A condition the solver itself must maintain is a
  `cuopt_assert`; `CUOPT_EXPECTS` is for conditions a caller can violate.
- **THREAD-1 — One threading model.** OpenMP, not `std::thread`/`pthread`. Random numbers
  use `pcgenerator_t` with seeds derived via SplitMix/`derive_seed`, not `std::mt19937`.
  See the RNG call-site policy in `SKILL.md`.

## Reuse and placement

- **REUSE-1 — Search before writing.** Before adding a norm, reduction, matvec, transpose,
  bound-propagation, compensated sum, copy or nvtx helper, search `cpp/src/linear_algebra`,
  `cpp/src/utilities` (including `copy_helpers.hpp`) and `cpp/src/presolve`. Reuse or extend the
  existing one. If you keep a second copy, say why.
- **REUSE-2 — Put code where its domain lives.** Presolve logic under `src/presolve`; GPU code
  under the GPU directory that uses it (`dual_simplex` is CPU-only); CPU-only data structures do
  not hold device buffers. The gRPC layer is a plain mapper: no solver or problem logic.
- **REUSE-3 — Do not duplicate the representation.** Prefer an existing problem type over a new
  one that holds the same fields.

## Public API surface

- **API-1 — Regular parameter first.** A new option is a regular `CUOPT_*` parameter, which every
  binding picks up automatically. No special-case setters, enums or getters in the Python, Java or
  C layers unless the parameter framework cannot express it.
- **API-2 — Hyperparameters stay internal.** Do not expose them in a public API or document them
  in the user docs. Name them with the algorithm prefix before `HYPER`, as their neighbours are.
- **API-3 — C API first, bindings derive.** Constants, enum values and statuses are generated or
  read from the C API, not copied by hand into another language. Do not add a binding-only
  accessor for something the C API does not expose.

## Tests

- **TEST-1 — A test must be able to fail.** Revert the feature mentally: does the test still
  pass? A warm-start test whose start is already the optimum, or a bug-fix PR without the
  reproducer, does not count.
- **TEST-2 — Test the contract, not the implementation.** No log-capture assertions for algorithm
  behaviour, no exact-reason matching, no test that an API is *not* supported. Parametric tests
  use `TEST_P`, not a loop over a parameter grid inside one test.
- **TEST-3 — Cheap by default.** Toy instances and tight time limits. A test on a large benchmark
  instance or a multi-minute run needs a stated reason.
- **TEST-4 — Skip plumbing tests.** No tests of trivial getters or save/restore with no numerical
  content. If a test is hard to read, simplify it or drop it.
- **TEST-5 — Siblings get sibling coverage.** A newly exposed algorithm path gets coverage
  comparable to the existing one (for example primal vs. dual simplex).

## Numerics

- **NUM-1 — Careful accumulation and rounding.** Use the existing compensated sum/dot routines
  for sensitive accumulation. When converting to integer coefficients, round in the safe
  direction (`ceil` for weights, not `floor`). Take tolerances from settings (`integer_tol`),
  not a local literal. Consider overflow.

## Documentation and user-visible text

- **DOC-1 — Plain sentences.** Complete sentences, capitalized headings and bullets, "Quickstart"
  as one word, "MIP" not "MILP". No sentences about earlier versions, internals, or "mirrors the
  Python/C API". No references to deprecated APIs in new docs.
- **DOC-2 — Generate, do not hand-copy.** Settings tables, constants and defaults come from the
  source of truth. A hand-written copy drifts.
- **LOG-1 — Log lines.** Say what to change (name the parameter), put elapsed time at the end of
  the line, and keep number formats consistent with neighbouring lines.

## PR hygiene

- **PR-1 — Minimal diff.** Revert unrelated files and reformatting. Version bumps and dependency
  upgrades go in their own PR. No build output committed.
- **PR-2 — Follow-ups are issues.** A deferred item is a tracking issue, linked from the PR, not a
  `TODO` that only describes it.
- **PR-3 — Description and release notes.** Fill in the PR description. Release-note entries are
  written for users, not copied from PR titles.
