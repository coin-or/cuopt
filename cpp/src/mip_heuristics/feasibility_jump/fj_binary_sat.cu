/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fj_cpu_binary.cuh"

#include <utilities/macros.cuh>
#include <utilities/pcgenerator.hpp>
#include <utilities/splitmix64.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

#define LIT(var, value) (((var) << 1) | (value))
#define LIT_VAR(lit)    ((lit) >> 1)
#define LIT_VALUE(lit)  ((lit) & 1)
#define LIT_NEG(lit)    ((lit) ^ 1)

namespace cuopt::mathematical_optimization::mip {

namespace {

static constexpr int encoding_stop_poll_period       = 128;
static constexpr size_t direct_encoding_term_limit   = 12;
static constexpr size_t direct_encoding_clause_limit = 32;
static constexpr size_t bdd_state_limit              = 20000;
static constexpr int sat_variable_limit              = 1000000;
static constexpr size_t bdd_term_limit               = 256;
static constexpr size_t sat_clause_limit             = 2000000;
static constexpr int bdd_false_terminal              = -1;
static constexpr int bdd_true_terminal               = -2;

using pb_term_t = std::pair<int, int64_t>;

enum class pb_row_encoding_result_t { encoded, infeasible, declined };

// Short rows are best represented by their prime clauses: each minimal overweight subset
// forbids exactly one combination, without introducing branching-only Tseitin variables.
static bool try_encode_pb_row_direct(const std::vector<pb_term_t>& terms,
                                     int64_t rhs,
                                     std::vector<int64_t>& subset_weight,
                                     std::vector<std::vector<int>>& cnf)
{
  if (terms.size() > direct_encoding_term_limit) return false;
  const size_t row_begin = cnf.size();
  const int count        = 1 << terms.size();
  subset_weight.assign(count, 0);
  for (int mask = 1; mask < count; ++mask) {
    const int bit       = std::countr_zero((unsigned)mask);
    subset_weight[mask] = subset_weight[mask & (mask - 1)] + std::abs(terms[bit].second);
    if (subset_weight[mask] <= rhs) continue;
    bool minimal = true;
    for (size_t k = 0; k < terms.size() && minimal; ++k) {
      if ((mask & (1 << k)) && subset_weight[mask] - std::abs(terms[k].second) > rhs)
        minimal = false;
    }
    if (!minimal) continue;
    std::vector<int> clause;
    for (size_t k = 0; k < terms.size(); ++k) {
      if (mask & (1 << k)) clause.push_back(LIT(terms[k].first, terms[k].second < 0));
    }
    cnf.push_back(std::move(clause));
    if (cnf.size() - row_begin > direct_encoding_clause_limit) {
      cnf.resize(row_begin);
      return false;
    }
  }
  return true;
}

static int encode_pb_row_bdd_node(const std::vector<pb_term_t>& terms,
                                  const std::vector<int64_t>& suffix,
                                  std::map<std::pair<int, int64_t>, int>& memo,
                                  int& variables,
                                  std::vector<std::vector<int>>& cnf,
                                  bool& overflow,
                                  int i,
                                  int64_t remaining)
{
  if (remaining < 0) return bdd_false_terminal;
  if (suffix[i] <= remaining) return bdd_true_terminal;
  const auto key = std::make_pair(i, remaining);
  if (const auto found = memo.find(key); found != memo.end()) return found->second;
  if (memo.size() > bdd_state_limit || variables >= sat_variable_limit ||
      terms.size() > bdd_term_limit) {
    overflow = true;
    return bdd_false_terminal;
  }
  const int low =
    encode_pb_row_bdd_node(terms, suffix, memo, variables, cnf, overflow, i + 1, remaining);
  const int high = encode_pb_row_bdd_node(
    terms, suffix, memo, variables, cnf, overflow, i + 1, remaining - std::abs(terms[i].second));
  if (overflow) return bdd_false_terminal;
  if (low == high) {
    memo.emplace(key, low);
    return low;
  }
  const int q = LIT(variables++, 1);
  const int z = LIT(terms[i].first, terms[i].second > 0);
  if (low != bdd_true_terminal) {
    std::vector<int> clause{LIT_NEG(q)};
    if (low >= 0) clause.push_back(low);
    cnf.push_back(std::move(clause));
  }
  if (high != bdd_true_terminal) {
    std::vector<int> clause{LIT_NEG(q), LIT_NEG(z)};
    if (high >= 0) clause.push_back(high);
    cnf.push_back(std::move(clause));
  }
  memo.emplace(key, q);
  return q;
}

static pb_row_encoding_result_t encode_pb_row_bdd(const std::vector<pb_term_t>& terms,
                                                  int64_t rhs,
                                                  int& variables,
                                                  std::vector<std::vector<int>>& cnf)
{
  std::vector<int64_t> suffix(terms.size() + 1, 0);
  for (int k = (int)terms.size() - 1; k >= 0; --k)
    suffix[k] = suffix[k + 1] + std::abs(terms[k].second);
  std::map<std::pair<int, int64_t>, int> memo;
  bool overflow  = false;
  const int root = encode_pb_row_bdd_node(terms, suffix, memo, variables, cnf, overflow, 0, rhs);
  if (overflow) return pb_row_encoding_result_t::declined;
  if (root == bdd_false_terminal) return pb_row_encoding_result_t::infeasible;
  if (root >= 0) cnf.push_back({root});
  return pb_row_encoding_result_t::encoded;
}

struct sat_variable_order_t {
  static constexpr int not_in_heap                   = -1;
  static constexpr double initial_activity_increment = 1.0;
  static constexpr double activity_rescale_threshold = 1e100;
  static constexpr double activity_rescale_factor    = 1e-100;
  static constexpr double initial_clause_activity    = 0.001;

  sat_variable_order_t(int n, uint64_t rng_seed) : position(n, not_in_heap), activity(n), tie(n)
  {
    cuopt::splitmix64_t tie_rng(rng_seed);
    for (int v = 0; v < n; ++v)
      tie[v] = tie_rng.next_u64();
  }

  bool empty() const { return heap.empty(); }

  void add_initial_activity(int v) { activity[v] += initial_clause_activity; }

  void clear_activity(int v) { activity[v] = 0.0; }

  void build_heap()
  {
    for (size_t v = 0; v < activity.size(); ++v)
      insert(v);
  }

  void insert(int v)
  {
    if (position[v] != not_in_heap) return;
    position[v] = (int)heap.size();
    heap.push_back(v);
    heap_up(position[v]);
  }

  int pop_best()
  {
    const int result = heap.front();
    const int last   = heap.back();
    heap.pop_back();
    position[result] = not_in_heap;
    if (!heap.empty()) {
      int p = 0;
      while (2 * p + 1 < (int)heap.size()) {
        int child = 2 * p + 1;
        if (child + 1 < (int)heap.size() && higher(heap[child + 1], heap[child])) ++child;
        if (!higher(heap[child], last)) break;
        heap[p]           = heap[child];
        position[heap[p]] = p;
        p                 = child;
      }
      heap[p]        = last;
      position[last] = p;
    }
    return result;
  }

  void bump(int v)
  {
    activity[v] += increment;
    if (activity[v] > activity_rescale_threshold) {
      for (double& score : activity)
        score *= activity_rescale_factor;
      increment *= activity_rescale_factor;
    }
    if (position[v] != not_in_heap) heap_up(position[v]);
  }

  void decay_activity(double factor) { increment /= factor; }

 private:
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
      heap[p]           = heap[parent];
      position[heap[p]] = p;
      p                 = parent;
    }
    heap[p]     = v;
    position[v] = p;
  }

  std::vector<int> heap, position;
  std::vector<double> activity;
  std::vector<uint64_t> tie;
  double increment{initial_activity_increment};
};

// Compact CDCL solver used only for large, objective-free Boolean feasibility models. Literals are
// encoded as 2 * variable + satisfying value.
struct sat_t {
  enum clause_field_t : int {
    clause_size_field,
    clause_lbd_field,
    clause_last_conflict_field,
    clause_flags_field,
    clause_header_size,
  };
  enum clause_flag_t : int {
    flag_none    = 0,
    flag_learned = 1 << 0,
    flag_deleted = 1 << 1,
    flag_locked  = 1 << 2,
  };
  static constexpr int no_reason                       = -1;
  static constexpr int no_literal                      = -1;
  static constexpr int no_variable                     = -1;
  static constexpr int no_conflict                     = -1;
  static constexpr int propagation_stopped             = -2;
  static constexpr int8_t unassigned                   = -1;
  static constexpr double activity_decay_min           = 0.94;
  static constexpr double activity_decay_step          = 0.01;
  static constexpr int activity_decay_choices          = 5;
  static constexpr int restart_interval_unit           = 64;
  static constexpr int restart_interval_choices        = 4;
  static constexpr int restart_growth_limit            = 4096;
  static constexpr int restart_growth_numerator        = 3;
  static constexpr int restart_growth_denominator      = 2;
  static constexpr size_t propagation_stop_poll_period = 256;
  static constexpr int64_t solve_stop_poll_period      = 128;
  static constexpr int protected_clause_lbd            = 2;
  static constexpr int protected_clause_size           = 2;
  static constexpr int clause_stale_conflicts          = 2000;
  static constexpr int database_reduction_interval     = 4000;
  static constexpr size_t learned_clause_reserve       = 10000;
  struct watch_t {
    int id;
    int blocker;
  };
  std::vector<int> arena;
  std::vector<int> clause_refs;
  std::vector<std::vector<watch_t>> watches;
  std::vector<int8_t> value, phase;
  std::vector<int> level, reason, trail, limits;
  std::vector<uint8_t> seen;
  sat_variable_order_t variable_order;
  size_t head{0};
  double decay{activity_decay_min};
  int conflicts{0};
  int restart_initial_limit{restart_interval_unit};
  int64_t steps{0};

  int* lits_of(int ref) { return arena.data() + ref + clause_header_size; }
  const int* lits_of(int ref) const { return arena.data() + ref + clause_header_size; }
  int clause_size(int ref) const { return arena[ref + clause_size_field]; }
  int literal_value(int lit) const
  {
    const int state = value[LIT_VAR(lit)];
    return state == unassigned ? unassigned : state == LIT_VALUE(lit);
  }
  bool enqueue(int lit, int why)
  {
    const int v         = LIT_VAR(lit);
    const int requested = LIT_VALUE(lit);
    if (value[v] != unassigned) return value[v] == requested;
    value[v] = phase[v] = requested;
    level[v]            = (int)limits.size();
    reason[v]           = why;
    trail.push_back(lit);
    return true;
  }
  int add(std::vector<int> lits, bool learned = false, int lbd = 0)
  {
    const int ref = (int)arena.size();
    arena.push_back((int)lits.size());
    arena.push_back(lbd);
    arena.push_back(conflicts);
    arena.push_back(learned ? flag_learned : flag_none);
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
      const int false_lit  = LIT_NEG(trail[head++]);
      auto& watched        = watches[false_lit];
      const int8_t* values = value.data();
      int* arena_base      = arena.data();
      size_t out           = 0;
      for (size_t k = 0; k < watched.size(); ++k) {
        const watch_t w         = watched[k];
        const int blocker_value = values[LIT_VAR(w.blocker)];
        if (blocker_value != unassigned && blocker_value == LIT_VALUE(w.blocker)) {
          watched[out++] = w;
          continue;
        }
        const int ref = w.id;
        int* header   = arena_base + ref;
        if (header[clause_flags_field] & flag_deleted) continue;
        int* lits        = header + clause_header_size;
        const int n_lits = header[clause_size_field];
        if (lits[0] == false_lit) std::swap(lits[0], lits[1]);
        if (literal_value(lits[0]) == 1) {
          watched[out++] = {ref, lits[0]};
          continue;
        }
        bool moved = false;
        for (int j = 2; j < n_lits; ++j) {
          if (literal_value(lits[j]) == 0) continue;
          std::swap(lits[1], lits[j]);
          watches[lits[1]].push_back({ref, lits[0]});
          moved = true;
          break;
        }
        if (moved) continue;
        watched[out++] = {ref, lits[0]};
        if (!enqueue(lits[0], ref)) {
          while (++k < watched.size())
            watched[out++] = watched[k];
          watched.resize(out);
          return ref;
        }
      }
      watched.resize(out);
      if (head % propagation_stop_poll_period == 0 && stop()) return propagation_stopped;
    }
    return no_conflict;
  }
  void backtrack(int target)
  {
    if ((int)limits.size() <= target) return;
    const size_t keep = limits[target];
    for (size_t k = trail.size(); k > keep;) {
      const int v = LIT_VAR(trail[--k]);
      value[v]    = unassigned;
      reason[v]   = no_reason;
      variable_order.insert(v);
    }
    trail.resize(keep);
    head = std::min(head, keep);
    limits.resize(target);
  }
  std::vector<int> analyze(int conflict, int& back, int& lbd)
  {
    std::vector<int> learned(1, no_literal);
    int paths = 0;
    int pivot = no_variable;
    int index = (int)trail.size() - 1;
    do {
      arena[conflict + clause_last_conflict_field] = conflicts;
      const int* clause_lits                       = lits_of(conflict);
      for (int q = 0; q < clause_size(conflict); ++q) {
        const int lit = clause_lits[q];
        const int v   = LIT_VAR(lit);
        if ((pivot != no_variable && v == LIT_VAR(pivot)) || seen[v] || level[v] == 0) continue;
        seen[v] = 1;
        variable_order.bump(v);
        if (level[v] == (int)limits.size())
          ++paths;
        else
          learned.push_back(lit);
      }
      while (!seen[LIT_VAR(trail[index])])
        --index;
      pivot                = trail[index--];
      seen[LIT_VAR(pivot)] = 0;
      --paths;
      conflict = reason[LIT_VAR(pivot)];
      cuopt_assert(paths == 0 || conflict != no_reason, "active conflict path has no reason");
    } while (paths > 0);
    learned[0]  = LIT_NEG(pivot);
    back        = 0;
    size_t best = 1;
    for (size_t k = 1; k < learned.size(); ++k) {
      const int v = LIT_VAR(learned[k]);
      seen[v]     = 0;
      if (level[v] > back) {
        back = level[v];
        best = k;
      }
    }
    if (learned.size() > 1) std::swap(learned[1], learned[best]);
    std::vector<int> levels;
    levels.reserve(learned.size());
    levels.push_back((int)limits.size());
    for (size_t k = 1; k < learned.size(); ++k)
      levels.push_back(level[LIT_VAR(learned[k])]);
    std::sort(levels.begin(), levels.end());
    lbd = static_cast<int>(std::unique(levels.begin(), levels.end()) - levels.begin());
    variable_order.decay_activity(decay);
    return learned;
  }

  void reduce_database()
  {
    for (int lit : trail)
      if (reason[LIT_VAR(lit)] != no_reason)
        arena[reason[LIT_VAR(lit)] + clause_flags_field] |= flag_locked;
    for (int ref : clause_refs) {
      const int flags = arena[ref + clause_flags_field];
      if ((flags & flag_learned) && !(flags & flag_deleted) && !(flags & flag_locked) &&
          arena[ref + clause_lbd_field] > protected_clause_lbd &&
          arena[ref + clause_size_field] > protected_clause_size &&
          arena[ref + clause_last_conflict_field] < conflicts - clause_stale_conflicts) {
        arena[ref + clause_flags_field] |= flag_deleted;
      }
    }
    for (int lit : trail)
      if (reason[LIT_VAR(lit)] != no_reason)
        arena[reason[LIT_VAR(lit)] + clause_flags_field] &= ~flag_locked;
  }

  sat_t(int n,
        std::vector<std::vector<int>> input,
        const std::vector<int8_t>& seed,
        uint64_t rng_seed)
    : watches(2 * n),
      value(n, unassigned),
      phase(n, 0),
      level(n),
      reason(n, no_reason),
      seen(n),
      variable_order(n, rng_seed)
  {
    cuopt::pcgenerator_t policy_rng(rng_seed);
    decay =
      activity_decay_min + activity_decay_step * policy_rng.uniform(0, activity_decay_choices);
    restart_initial_limit =
      restart_interval_unit * policy_rng.uniform(1, restart_interval_choices + 1);

    for (int v = 0; v < n; ++v)
      if (v < (int)seed.size()) phase[v] = seed[v];

    clause_refs.reserve(input.size() + learned_clause_reserve);
    size_t total_lits = 0;
    for (const auto& clause : input)
      total_lits += clause.size();
    arena.reserve((total_lits + clause_header_size * input.size()) * 3 / 2);
    for (auto& clause : input) {
      for (int lit : clause)
        variable_order.add_initial_activity(LIT_VAR(lit));
      add(std::move(clause));
    }
    // Encoding nodes have no independent model meaning; conflicts may still promote them later.
    for (int v = (int)seed.size(); v < n; ++v)
      variable_order.clear_activity(v);
    variable_order.build_heap();
  }

  template <typename Stop>
  fj_binary_sat_result_t solve(Stop stop)
  {
    for (int ref : clause_refs) {
      const int n_lits = clause_size(ref);
      if (n_lits == 0 || (n_lits == 1 && !enqueue(lits_of(ref)[0], ref)))
        return fj_binary_sat_result_t::infeasible;
    }
    int restart_limit = restart_initial_limit;
    int since_restart = 0;
    while (true) {
      if (++steps % solve_stop_poll_period == 0 && stop()) return fj_binary_sat_result_t::stopped;
      const int conflict = propagate(stop);
      if (conflict == propagation_stopped) return fj_binary_sat_result_t::stopped;
      if (conflict != no_conflict) {
        ++conflicts;
        ++since_restart;
        if (limits.empty()) return fj_binary_sat_result_t::infeasible;
        int back            = 0;
        int lbd             = 0;
        auto learned        = analyze(conflict, back, lbd);
        const int asserting = learned[0];
        backtrack(back);
        const int id                                   = add(std::move(learned), true, lbd);
        [[maybe_unused]] const bool asserting_enqueued = enqueue(asserting, id);
        cuopt_assert(asserting_enqueued, "learned clause is not asserting after backtrack");
        if (conflicts % database_reduction_interval == 0) reduce_database();
        if (since_restart >= restart_limit) {
          backtrack(0);
          since_restart = 0;
          restart_limit = restart_limit < restart_growth_limit
                            ? restart_limit * restart_growth_numerator / restart_growth_denominator
                            : restart_initial_limit;
        }
      } else {
        int decision = no_variable;
        while (!variable_order.empty()) {
          const int candidate = variable_order.pop_best();
          if (value[candidate] == unassigned) {
            decision = candidate;
            break;
          }
        }
        if (decision == no_variable) return fj_binary_sat_result_t::feasible;
        limits.push_back((int)trail.size());
        [[maybe_unused]] const bool decision_enqueued =
          enqueue(LIT(decision, phase[decision]), no_reason);
        cuopt_assert(decision_enqueued, "decision variable is already assigned");
      }
    }
  }
};

struct sat_bve_t {
  static constexpr int stop_poll_period    = 256;
  static constexpr size_t occurrence_limit = 32;
  static constexpr size_t resolvent_limit  = 64;
  static constexpr int no_required_value   = -1;

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
      // duplicate
      if (out && clause[out - 1] == lit) continue;
      // tautology, drop
      if (out && LIT_NEG(clause[out - 1]) == lit) return false;
      clause[out++] = lit;
    }
    clause.resize(out);
    return true;
  }

  template <typename Stop>
  bool presolve(std::vector<std::vector<int>>& input, int n, int original_variables, Stop& stop)
  {
    struct clause_t {
      std::vector<int> lits;
      bool deleted{false};
    };
    std::vector<clause_t> clauses;
    clauses.reserve(input.size() * 2);
    for (auto& clause : input) {
      if (canonicalize(clause)) clauses.push_back({std::move(clause), false});
    }
    std::vector<std::vector<int>> occurrence(2 * n);
    for (size_t id = 0; id < clauses.size(); ++id)
      for (int lit : clauses[id].lits)
        occurrence[lit].push_back(id);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const size_t degree_a = occurrence[LIT(a, 0)].size() + occurrence[LIT(a, 1)].size();
      const size_t degree_b = occurrence[LIT(b, 0)].size() + occurrence[LIT(b, 1)].size();
      return degree_a != degree_b ? degree_a < degree_b : a < b;
    });
    std::vector<int> positive, negative;
    std::vector<std::vector<int>> resolvents;
    for (int v : order) {
      if (v % stop_poll_period == 0 && stop()) return false;
      positive.clear();
      negative.clear();
      for (int id : occurrence[LIT(v, 1)])
        if (!clauses[id].deleted) positive.push_back(id);
      for (int id : occurrence[LIT(v, 0)])
        if (!clauses[id].deleted) negative.push_back(id);
      if (positive.empty() || negative.empty() || positive.size() > occurrence_limit ||
          negative.size() > occurrence_limit)
        continue;

      size_t n_res = 0;
      bool reject  = false;
      for (int p : positive) {
        for (int q : negative) {
          if (resolvents.size() <= n_res) resolvents.emplace_back();
          std::vector<int>& resolvent = resolvents[n_res];
          resolvent.clear();
          for (int lit : clauses[p].lits)
            if (LIT_VAR(lit) != v) resolvent.push_back(lit);
          for (int lit : clauses[q].lits)
            if (LIT_VAR(lit) != v) resolvent.push_back(lit);
          if (!canonicalize(resolvent)) continue;
          ++n_res;
          if (n_res > resolvent_limit || n_res > positive.size() + negative.size()) {
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
        const int id = (int)clauses.size();
        clauses.push_back({resolvents[r], false});
        for (int lit : clauses[id].lits)
          occurrence[lit].push_back(id);
      }
    }

    std::vector<uint8_t> used(n);
    for (const auto& clause : clauses)
      if (!clause.deleted)
        for (int lit : clause.lits)
          used[LIT_VAR(lit)] = 1;
    for (int v = 0; v < original_variables; ++v)
      if (used[v]) compact_to_original.push_back(v);

    model_variables = (int)compact_to_original.size();
    for (int v = original_variables; v < n; ++v)
      if (used[v]) compact_to_original.push_back(v);

    std::vector<int> remap(n, -1);
    for (size_t v = 0; v < compact_to_original.size(); ++v)
      remap[compact_to_original[v]] = v;
    input.clear();

    for (auto& clause : clauses)
      if (!clause.deleted) {
        for (int& lit : clause.lits)
          lit = LIT(remap[LIT_VAR(lit)], LIT_VALUE(lit));
        input.push_back(std::move(clause.lits));
      }
    return true;
  }

  void postsolve(const std::vector<int8_t>& compact,
                 std::vector<int8_t>& value,
                 const std::vector<int8_t>& seed) const
  {
    value = seed;
    for (size_t i = 0; i < compact_to_original.size(); ++i)
      value[compact_to_original[i]] = compact[i];
    for (auto it = extension.rbegin(); it != extension.rend(); ++it) {
      int required = no_required_value;
      for (const auto& clause : it->clauses) {
        bool satisfied = false;
        for (int lit : clause)
          if (LIT_VAR(lit) != it->variable && value[LIT_VAR(lit)] == LIT_VALUE(lit))
            satisfied = true;
        if (satisfied) continue;
        for (int lit : clause)
          if (LIT_VAR(lit) == it->variable) {
            cuopt_assert(required == no_required_value || required == LIT_VALUE(lit),
                         "BVE recovery requires conflicting values");
            required = LIT_VALUE(lit);
          }
      }
      value[it->variable] = required != no_required_value ? required : seed[it->variable];
    }
  }
};
}  // namespace

template <typename coef_t>
fj_binary_sat_result_t fj_bin_sat_search(const fj_bin_problem_t<coef_t>& pb,
                                         std::vector<int8_t>& assignment,
                                         uint64_t seed,
                                         const std::function<bool()>& stop,
                                         int64_t& steps)
{
  std::vector<std::vector<int>> cnf;
  int variables = pb.n_variables;
  std::vector<pb_term_t> terms;
  std::vector<int64_t> subset_weight;
  for (int r = 0; r < pb.n_constraints; ++r) {
    if (r % encoding_stop_poll_period == 0 && stop()) return fj_binary_sat_result_t::stopped;
    terms.clear();
    for (int p = pb.offsets[r]; p < pb.offsets[r + 1]; ++p)
      terms.emplace_back(pb.variables[p], pb.coefficients[p]);
    std::sort(terms.begin(), terms.end());
    size_t out = 0;
    for (size_t k = 0; k < terms.size();) {
      const int v         = terms[k].first;
      int64_t coefficient = 0;
      do
        coefficient += terms[k++].second;
      while (k < terms.size() && terms[k].first == v);
      if (coefficient) terms[out++] = {v, coefficient};
    }
    terms.resize(out);
    int64_t rhs   = pb.bound[r];
    int64_t total = 0;
    for (auto [v, coefficient] : terms) {
      rhs -= std::min<int64_t>(0, coefficient);
      total += std::abs(coefficient);
    }
    if (rhs < 0) return fj_binary_sat_result_t::infeasible;
    if (total <= rhs) continue;

    if (!try_encode_pb_row_direct(terms, rhs, subset_weight, cnf)) {
      const auto encoding = encode_pb_row_bdd(terms, rhs, variables, cnf);
      if (encoding == pb_row_encoding_result_t::declined) return fj_binary_sat_result_t::declined;
      if (encoding == pb_row_encoding_result_t::infeasible)
        return fj_binary_sat_result_t::infeasible;
    }
    if (cnf.size() > sat_clause_limit) return fj_binary_sat_result_t::declined;
  }
  sat_bve_t bve;
  if (!bve.presolve(cnf, variables, pb.n_variables, stop)) return fj_binary_sat_result_t::stopped;
  std::vector<int8_t> compact_seed;
  compact_seed.reserve(bve.model_variables);
  for (int k = 0; k < bve.model_variables; ++k)
    compact_seed.push_back(assignment[bve.compact_to_original[k]]);

  sat_t sat((int)bve.compact_to_original.size(), std::move(cnf), compact_seed, seed);
  const auto result = sat.solve(stop);
  steps             = sat.steps;
  if (result == fj_binary_sat_result_t::feasible) {
    std::vector<int8_t> full_seed(variables, 0), full_value;
    std::copy(assignment.begin(), assignment.end(), full_seed.begin());
    bve.postsolve(sat.value, full_value, full_seed);
    for (int r = 0; r < pb.n_constraints; ++r) {
      [[maybe_unused]] int64_t lhs = 0;
      for (int p = pb.offsets[r]; p < pb.offsets[r + 1]; ++p)
        lhs += int64_t(pb.coefficients[p]) * full_value[pb.variables[p]];
      cuopt_assert(lhs <= pb.bound[r], "SAT assignment violates original constraint");
    }
    std::copy_n(full_value.begin(), pb.n_variables, assignment.begin());
  }
  return result;
}

template fj_binary_sat_result_t fj_bin_sat_search<int8_t>(const fj_bin_problem_t<int8_t>&,
                                                          std::vector<int8_t>&,
                                                          uint64_t,
                                                          const std::function<bool()>&,
                                                          int64_t&);

template fj_binary_sat_result_t fj_bin_sat_search<int16_t>(const fj_bin_problem_t<int16_t>&,
                                                           std::vector<int8_t>&,
                                                           uint64_t,
                                                           const std::function<bool()>&,
                                                           int64_t&);

}  // namespace cuopt::mathematical_optimization::mip
