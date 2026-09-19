/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fj_cpu_binary.cuh"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Compact CDCL solver used only for large, objective-free Boolean feasibility models. Literals are
// encoded as 2 * variable + satisfying value.
struct fj_sat_t {
  static constexpr int clause_header = 4;
  static constexpr int flag_learned = 1;
  static constexpr int flag_deleted = 2;
  static constexpr int flag_locked  = 4;
  struct watch_t {
    int id;
    int blocker;
  };
  std::vector<int> arena;
  std::vector<int> clause_refs;
  std::vector<std::vector<watch_t>> watches;
  std::vector<int8_t> value, phase;
  std::vector<int> level, reason, trail, limits, heap, position;
  std::vector<uint8_t> seen;
  std::vector<double> activity;
  std::vector<uint64_t> tie;
  size_t head{0};
  double increment{1.0};
  double decay{0.95};
  int conflicts{0};
  int64_t steps{0};

  int* lits_of(int ref) { return arena.data() + ref + clause_header; }
  const int* lits_of(int ref) const { return arena.data() + ref + clause_header; }
  int clause_size(int ref) const { return arena[ref]; }
  int literal_value(int lit) const
  {
    const int x = value[lit >> 1];
    return x < 0 ? -1 : x == (lit & 1);
  }
  bool higher(int a, int b) const
  {
    return activity[a] != activity[b] ? activity[a] > activity[b] : tie[a] > tie[b];
  }
  void heap_up(int p)
  {
    const int v = heap[p];
    while (p > 0) {
      const int parent = (p - 1) / 2;
      if (!higher(v, heap[parent])) break;
      heap[p] = heap[parent];
      position[heap[p]] = p;
      p = parent;
    }
    heap[p] = v;
    position[v] = p;
  }
  void insert(int v)
  {
    if (position[v] >= 0) return;
    position[v] = static_cast<int>(heap.size());
    heap.push_back(v);
    heap_up(position[v]);
  }
  int pop()
  {
    const int result = heap.front();
    const int last = heap.back();
    heap.pop_back();
    position[result] = -1;
    if (!heap.empty()) {
      int p = 0;
      while (2 * p + 1 < static_cast<int>(heap.size())) {
        int child = 2 * p + 1;
        if (child + 1 < static_cast<int>(heap.size()) && higher(heap[child + 1], heap[child])) ++child;
        if (!higher(heap[child], last)) break;
        heap[p] = heap[child];
        position[heap[p]] = p;
        p = child;
      }
      heap[p] = last;
      position[last] = p;
    }
    return result;
  }
  void bump(int v)
  {
    activity[v] += increment;
    if (activity[v] > 1e100) {
      for (double& score : activity) score *= 1e-100;
      increment *= 1e-100;
    }
    if (position[v] >= 0) heap_up(position[v]);
  }
  bool enqueue(int lit, int why)
  {
    const int v = lit >> 1;
    const int requested = lit & 1;
    if (value[v] >= 0) return value[v] == requested;
    value[v] = phase[v] = static_cast<int8_t>(requested);
    level[v] = static_cast<int>(limits.size());
    reason[v] = why;
    trail.push_back(lit);
    return true;
  }
  int add(std::vector<int> lits, bool learned = false, int lbd = 0)
  {
    const int ref = static_cast<int>(arena.size());
    arena.push_back(static_cast<int>(lits.size()));
    arena.push_back(lbd);
    arena.push_back(conflicts);
    arena.push_back(learned ? flag_learned : 0);
    arena.insert(arena.end(), lits.begin(), lits.end());
    clause_refs.push_back(ref);
    if (lits.size() >= 2) {
      watches[lits[0]].push_back({ref, lits[1]});
      watches[lits[1]].push_back({ref, lits[0]});
    }
    return ref;
  }
  template <typename Stop>
  int propagate(Stop& stop)
  {
    while (head < trail.size()) {
      const int false_lit = trail[head++] ^ 1;
      auto& watched         = watches[false_lit];
      const int8_t* values = value.data();
      int* arena_base      = arena.data();
      size_t out            = 0;
      for (size_t k = 0; k < watched.size(); ++k) {
        const watch_t w = watched[k];
        const int blocker_value = values[w.blocker >> 1];
        if (blocker_value >= 0 && blocker_value == (w.blocker & 1)) {
          watched[out++] = w;
          continue;
        }
        const int ref = w.id;
        int* header   = arena_base + ref;
        if (header[3] & flag_deleted) continue;
        int* lits           = header + clause_header;
        const size_t n_lits = static_cast<size_t>(header[0]);
        if (lits[0] == false_lit) std::swap(lits[0], lits[1]);
        if (literal_value(lits[0]) == 1) {
          watched[out++] = {ref, lits[0]};
          continue;
        }
        bool moved = false;
        for (size_t j = 2; j < n_lits; ++j) {
          if (literal_value(lits[j]) == 0) continue;
          std::swap(lits[1], lits[j]);
          watches[lits[1]].push_back({ref, lits[0]});
          moved = true;
          break;
        }
        if (moved) continue;
        watched[out++] = {ref, lits[0]};
        if (!enqueue(lits[0], ref)) {
          while (++k < watched.size()) watched[out++] = watched[k];
          watched.resize(out);
          return ref;
        }
      }
      watched.resize(out);
      if ((head & 255) == 0 && stop()) return -2;
    }
    return -1;
  }
  void backtrack(int target)
  {
    if (static_cast<int>(limits.size()) <= target) return;
    const size_t keep = limits[target];
    for (size_t k = trail.size(); k > keep;) {
      const int v = trail[--k] >> 1;
      value[v] = -1;
      reason[v] = -1;
      insert(v);
    }
    trail.resize(keep);
    head = std::min(head, keep);
    limits.resize(target);
  }
  std::vector<int> analyze(int conflict, int& back, int& lbd)
  {
    std::vector<int> learned(1, -1);
    int paths = 0;
    int pivot = -1;
    int index = static_cast<int>(trail.size()) - 1;
    do {
      arena[conflict + 2]    = conflicts;
      const int* clause_lits = lits_of(conflict);
      for (int q = 0; q < clause_size(conflict); ++q) {
        const int lit = clause_lits[q];
        const int v = lit >> 1;
        if ((pivot >= 0 && v == (pivot >> 1)) || seen[v] || level[v] == 0) continue;
        seen[v] = 1;
        bump(v);
        if (level[v] == static_cast<int>(limits.size())) ++paths;
        else learned.push_back(lit);
      }
      while (!seen[trail[index] >> 1]) --index;
      pivot = trail[index--];
      seen[pivot >> 1] = 0;
      --paths;
      conflict = reason[pivot >> 1];
    } while (paths > 0);
    learned[0] = pivot ^ 1;
    back = 0;
    int best = 1;
    for (int k = 1; k < static_cast<int>(learned.size()); ++k) {
      const int v = learned[k] >> 1;
      seen[v] = 0;
      if (level[v] > back) {
        back = level[v];
        best = k;
      }
    }
    if (learned.size() > 1) std::swap(learned[1], learned[best]);
    std::vector<int> levels;
    levels.reserve(learned.size());
    levels.push_back(static_cast<int>(limits.size()));
    for (size_t k = 1; k < learned.size(); ++k) levels.push_back(level[learned[k] >> 1]);
    std::sort(levels.begin(), levels.end());
    lbd = static_cast<int>(std::unique(levels.begin(), levels.end()) - levels.begin());
    increment /= decay;
    return learned;
  }

  void reduce_database()
  {
    for (int lit : trail)
      if (reason[lit >> 1] >= 0) arena[reason[lit >> 1] + 3] |= flag_locked;
    for (int ref : clause_refs) {
      const int flags = arena[ref + 3];
      if ((flags & flag_learned) && !(flags & flag_deleted) && !(flags & flag_locked) &&
          arena[ref + 1] > 2 && arena[ref] > 2 && arena[ref + 2] < conflicts - 2000) {
        arena[ref + 3] |= flag_deleted;
      }
    }
    for (int lit : trail)
      if (reason[lit >> 1] >= 0) arena[reason[lit >> 1] + 3] &= ~flag_locked;
  }

  fj_sat_t(int n, std::vector<std::vector<int>> input, const std::vector<int8_t>& seed, uint64_t rng)
    : watches(2 * n), value(n, -1), phase(n, 0), level(n), reason(n, -1), position(n, -1),
      seen(n), activity(n), tie(n)
  {
    decay = 0.94 + 0.01 * (rng % 5);
    for (int v = 0; v < n; ++v) {
      rng ^= rng >> 12;
      rng ^= rng << 25;
      rng ^= rng >> 27;
      tie[v] = rng * 2685821657736338717ULL;
      if (v < static_cast<int>(seed.size())) phase[v] = seed[v];
    }
    clause_refs.reserve(input.size() + 10000);
    size_t total_lits = 0;
    for (const auto& clause : input) total_lits += clause.size();
    arena.reserve((total_lits + clause_header * input.size()) * 3 / 2);
    for (auto& clause : input) {
      for (int lit : clause) activity[lit >> 1] += 0.001;
      add(std::move(clause));
    }
    // Encoding nodes have no independent model meaning; conflicts may still promote them later.
    for (int v = static_cast<int>(seed.size()); v < n; ++v) activity[v] = 0.0;
    for (int v = 0; v < n; ++v) insert(v);
  }

  template <typename Stop>
  int solve(Stop stop, uint64_t seed)
  {
    for (int ref : clause_refs) {
      const int n_lits = clause_size(ref);
      if (n_lits == 0 || (n_lits == 1 && !enqueue(lits_of(ref)[0], ref))) return -1;
    }
    int restart_limit = 64 + 64 * (seed % 4);
    int since_restart = 0;
    while (true) {
      if ((++steps & 127) == 0 && stop()) return 0;
      const int conflict = propagate(stop);
      if (conflict == -2) return 0;
      if (conflict >= 0) {
        ++conflicts;
        ++since_restart;
        if (limits.empty()) return -1;
        int back = 0;
        int lbd = 0;
        auto learned = analyze(conflict, back, lbd);
        const int asserting = learned[0];
        backtrack(back);
        const int id = add(std::move(learned), true, lbd);
        if (!enqueue(asserting, id)) return -1;
        if (conflicts % 4000 == 0) reduce_database();
        if (since_restart >= restart_limit) {
          backtrack(0);
          since_restart = 0;
          restart_limit = restart_limit < 4096 ? restart_limit * 3 / 2 : 64 + 64 * (seed % 4);
        }
      } else {
        int decision = -1;
        while (!heap.empty()) {
          const int candidate = pop();
          if (value[candidate] < 0) {
            decision = candidate;
            break;
          }
        }
        if (decision < 0) return 1;
        limits.push_back(static_cast<int>(trail.size()));
        enqueue(2 * decision + phase[decision], -1);
      }
    }
  }
};

struct fj_sat_bve_t {
  struct extension_t {
    int variable;
    std::vector<std::vector<int>> clauses;
  };
  std::vector<extension_t> extension;
  std::vector<int> compact_to_original;
  int model_variables{0};

  static bool canonicalize(std::vector<int>& clause)
  {
    std::sort(clause.begin(), clause.end());
    size_t out = 0;
    for (int lit : clause) {
      if (out && clause[out - 1] == lit) continue;
      if (out && (clause[out - 1] ^ 1) == lit) return false;
      clause[out++] = lit;
    }
    clause.resize(out);
    return true;
  }

  template <typename Stop>
  bool reduce(std::vector<std::vector<int>>& input, int n, int original_variables, Stop& stop)
  {
    struct clause_t { std::vector<int> lits; bool deleted{false}; };
    std::vector<clause_t> clauses;
    clauses.reserve(input.size() * 2);
    for (auto& clause : input) {
      if (canonicalize(clause)) clauses.push_back({std::move(clause), false});
    }
    std::vector<std::vector<int>> occurrence(2 * n);
    for (int id = 0; id < static_cast<int>(clauses.size()); ++id)
      for (int lit : clauses[id].lits) occurrence[lit].push_back(id);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const size_t degree_a = occurrence[2 * a].size() + occurrence[2 * a + 1].size();
      const size_t degree_b = occurrence[2 * b].size() + occurrence[2 * b + 1].size();
      return degree_a != degree_b ? degree_a < degree_b : a < b;
    });
    std::vector<int> positive, negative;
    std::vector<std::vector<int>> resolvents;
    for (int v : order) {
      if ((v & 255) == 0 && stop()) return false;
      positive.clear();
      negative.clear();
      for (int id : occurrence[2 * v + 1]) if (!clauses[id].deleted) positive.push_back(id);
      for (int id : occurrence[2 * v]) if (!clauses[id].deleted) negative.push_back(id);
      if (positive.empty() || negative.empty() || positive.size() > 32 || negative.size() > 32)
        continue;

      size_t n_res = 0;
      bool reject  = false;
      for (int p : positive) {
        for (int q : negative) {
          if (resolvents.size() <= n_res) resolvents.emplace_back();
          std::vector<int>& resolvent = resolvents[n_res];
          resolvent.clear();
          for (int lit : clauses[p].lits) if ((lit >> 1) != v) resolvent.push_back(lit);
          for (int lit : clauses[q].lits) if ((lit >> 1) != v) resolvent.push_back(lit);
          if (!canonicalize(resolvent)) continue;
          ++n_res;
          if (n_res > 64 || n_res > positive.size() + negative.size()) {
            reject = true;
            break;
          }
        }
        if (reject) break;
      }
      if (reject) continue;

      extension_t saved{v, {}};
      saved.clauses.reserve(positive.size() + negative.size());
      for (int id : positive) {
        saved.clauses.push_back(std::move(clauses[id].lits));
        clauses[id].deleted = true;
      }
      for (int id : negative) {
        saved.clauses.push_back(std::move(clauses[id].lits));
        clauses[id].deleted = true;
      }
      extension.push_back(std::move(saved));
      for (size_t r = 0; r < n_res; ++r) {
        const int id = static_cast<int>(clauses.size());
        clauses.push_back({resolvents[r], false});
        for (int lit : clauses[id].lits) occurrence[lit].push_back(id);
      }
    }

    std::vector<uint8_t> used(n);
    for (const auto& clause : clauses) if (!clause.deleted)
      for (int lit : clause.lits) used[lit >> 1] = 1;
    for (int v = 0; v < original_variables; ++v) if (used[v]) compact_to_original.push_back(v);
    model_variables = static_cast<int>(compact_to_original.size());
    for (int v = original_variables; v < n; ++v) if (used[v]) compact_to_original.push_back(v);
    std::vector<int> remap(n, -1);
    for (int v = 0; v < static_cast<int>(compact_to_original.size()); ++v)
      remap[compact_to_original[v]] = v;
    input.clear();
    for (auto& clause : clauses) if (!clause.deleted) {
      for (int& lit : clause.lits) lit = 2 * remap[lit >> 1] + (lit & 1);
      input.push_back(std::move(clause.lits));
    }
    return true;
  }

  bool recover(const std::vector<int8_t>& compact,
               std::vector<int8_t>& value,
               const std::vector<int8_t>& seed) const
  {
    value = seed;
    for (size_t i = 0; i < compact_to_original.size(); ++i)
      value[compact_to_original[i]] = compact[i];
    for (auto it = extension.rbegin(); it != extension.rend(); ++it) {
      int required = -1;
      for (const auto& clause : it->clauses) {
        bool satisfied = false;
        for (int lit : clause)
          if ((lit >> 1) != it->variable && value[lit >> 1] == (lit & 1)) satisfied = true;
        if (satisfied) continue;
        for (int lit : clause) if ((lit >> 1) == it->variable) {
          if (required >= 0 && required != (lit & 1)) return false;
          required = lit & 1;
        }
      }
      value[it->variable] = static_cast<int8_t>(required >= 0 ? required : seed[it->variable]);
    }
    return true;
  }
};
}  // namespace

template <typename coef_t>
int fj_bin_sat_search(const fj_bin_problem_t<coef_t>& pb,
                      std::vector<int8_t>& assignment,
                      uint64_t seed,
                      const std::function<bool()>& stop,
                      int64_t& steps)
{
  std::vector<std::vector<int>> cnf;
  int variables = pb.n_variables;
  std::vector<std::pair<int, int64_t>> terms;
  std::vector<int64_t> subset_weight;
  for (int r = 0; r < pb.n_constraints; ++r) {
    if ((r & 127) == 0 && stop()) return 0;
    terms.clear();
    for (int p = pb.offsets[r]; p < pb.offsets[r + 1]; ++p)
      terms.emplace_back(pb.variables[p], pb.coefficients[p]);
    std::sort(terms.begin(), terms.end());
    size_t out = 0;
    for (size_t k = 0; k < terms.size();) {
      const int v = terms[k].first;
      int64_t coefficient = 0;
      do coefficient += terms[k++].second; while (k < terms.size() && terms[k].first == v);
      if (coefficient) terms[out++] = {v, coefficient};
    }
    terms.resize(out);
    int64_t rhs = pb.bound[r];
    int64_t total = 0;
    for (auto [v, coefficient] : terms) {
      rhs -= std::min<int64_t>(0, coefficient);
      total += std::abs(coefficient);
    }
    if (rhs < 0) return -1;
    if (total <= rhs) continue;

    // Short rows are best represented by their prime clauses: each minimal overweight subset
    // forbids exactly one combination, without introducing branching-only Tseitin variables.
    const size_t row_begin = cnf.size();
    bool directly_encoded = terms.size() <= 12;
    if (directly_encoded) {
      const int count = 1 << terms.size();
      subset_weight.assign(count, 0);
      for (int mask = 1; mask < count; ++mask) {
        const int bit = __builtin_ctz(static_cast<unsigned>(mask));
        subset_weight[mask] = subset_weight[mask & (mask - 1)] + std::abs(terms[bit].second);
        if (subset_weight[mask] <= rhs) continue;
        bool minimal = true;
        for (int k = 0; k < static_cast<int>(terms.size()) && minimal; ++k) {
          if ((mask & (1 << k)) && subset_weight[mask] - std::abs(terms[k].second) > rhs)
            minimal = false;
        }
        if (!minimal) continue;
        std::vector<int> clause;
        for (int k = 0; k < static_cast<int>(terms.size()); ++k) {
          if (mask & (1 << k))
            clause.push_back(2 * terms[k].first + (terms[k].second < 0));
        }
        cnf.push_back(std::move(clause));
        if (cnf.size() - row_begin > 32) {
          directly_encoded = false;
          break;
        }
      }
    }

    if (!directly_encoded) {
      cnf.resize(row_begin);
      std::vector<int64_t> suffix(terms.size() + 1, 0);
      for (int k = static_cast<int>(terms.size()) - 1; k >= 0; --k)
        suffix[k] = suffix[k + 1] + std::abs(terms[k].second);
      std::map<std::pair<int, int64_t>, int> memo;
      bool overflow = false;
      auto node = [&](auto&& self, int i, int64_t remaining) -> int {
        if (remaining < 0) return -1;
        if (suffix[i] <= remaining) return -2;
        const auto key = std::make_pair(i, remaining);
        if (const auto found = memo.find(key); found != memo.end()) return found->second;
        if (memo.size() > 20000 || variables >= 1000000 || terms.size() > 256) {
          overflow = true;
          return -1;
        }
        const int low = self(self, i + 1, remaining);
        const int high = self(self, i + 1, remaining - std::abs(terms[i].second));
        if (overflow) return -1;
        if (low == high) {
          memo.emplace(key, low);
          return low;
        }
        const int q = 2 * variables++ + 1;
        const int z = 2 * terms[i].first + (terms[i].second > 0);
        if (low != -2) {
          std::vector<int> clause{q ^ 1};
          if (low >= 0) clause.push_back(low);
          cnf.push_back(std::move(clause));
        }
        if (high != -2) {
          std::vector<int> clause{q ^ 1, z ^ 1};
          if (high >= 0) clause.push_back(high);
          cnf.push_back(std::move(clause));
        }
        memo.emplace(key, q);
        return q;
      };
      const int root = node(node, 0, rhs);
      if (overflow) return -2;
      if (root == -1) return -1;
      if (root >= 0) cnf.push_back({root});
    }
    if (cnf.size() > 2000000) return -2;
  }
  fj_sat_bve_t bve;
  if (!bve.reduce(cnf, variables, pb.n_variables, stop)) return 0;
  std::vector<int8_t> compact_seed;
  compact_seed.reserve(bve.model_variables);
  for (int k = 0; k < bve.model_variables; ++k)
    compact_seed.push_back(assignment[bve.compact_to_original[k]]);

  fj_sat_t sat(static_cast<int>(bve.compact_to_original.size()),
               std::move(cnf),
               compact_seed,
               seed);
  const int result = sat.solve(stop, seed);
  steps            = sat.steps;
  if (result == 1) {
    std::vector<int8_t> full_seed(variables, 0), full_value;
    std::copy(assignment.begin(), assignment.end(), full_seed.begin());
    if (!bve.recover(sat.value, full_value, full_seed)) return 0;
    for (int r = 0; r < pb.n_constraints; ++r) {
      int64_t lhs = 0;
      for (int p = pb.offsets[r]; p < pb.offsets[r + 1]; ++p)
        lhs += int64_t(pb.coefficients[p]) * full_value[pb.variables[p]];
      if (lhs > pb.bound[r]) return 0;
    }
    std::copy_n(full_value.begin(), pb.n_variables, assignment.begin());
  }
  return result;
}

template int fj_bin_sat_search<int8_t>(const fj_bin_problem_t<int8_t>&,
                                       std::vector<int8_t>&,
                                       uint64_t,
                                       const std::function<bool()>&,
                                       int64_t&);

template int fj_bin_sat_search<int16_t>(const fj_bin_problem_t<int16_t>&,
                                        std::vector<int8_t>&,
                                        uint64_t,
                                        const std::function<bool()>&,
                                        int64_t&);

}  // namespace cuopt::mathematical_optimization::mip
