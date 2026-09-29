/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include "../../../experiments/hive_lns/interface.hpp"

namespace cuopt::hive_lns {
// Improvement only: every trajectory starts from a feasible population copy.
inline void run_lns(const model_t& model,
                    const snapshot_fn& snapshot,
                    const submit_fn& submit,
                    const stop_fn& stop,
                    uint64_t seed)
{
  const size_t n = model.lower.size();
  if (!n) return;
  std::mt19937_64 rng(seed);

  // Precompute column-row structure with normalized coefficients using L1 norm
  std::vector<std::vector<std::pair<int, double>>> column_rows(n);
  std::vector<double> row_l1_norm(model.row_lower.size(), 1e-10);

  // Compute L1 norm for each row for proper normalization
  for (size_t r = 0; r < model.row_lower.size(); ++r) {
    for (int p = model.offsets[r]; p < model.offsets[r + 1]; ++p) {
      row_l1_norm[r] += std::abs(model.coefficients[p]);
    }
  }

  for (size_t r = 0; r < model.row_lower.size(); ++r) {
    for (int p = model.offsets[r]; p < model.offsets[r + 1]; ++p) {
      const int col      = model.columns[p];
      const double coeff = model.coefficients[p];
      // Normalize coefficient by L1 norm of the row
      column_rows[col].emplace_back(r, coeff / row_l1_norm[r]);
    }
  }

  std::vector<int> movable;
  for (size_t j = 0; j < n; ++j) {
    if (model.lower[j] < model.upper[j]) movable.push_back(j);
  }
  if (movable.empty()) return;

  // Simplified multi-armed bandit for repair operator selection
  struct BanditArm {
    int pulls     = 0;
    double reward = 0.0;
    double ucb() const
    {
      if (pulls == 0) return 1e6;  // Large but finite value
      const double exploitation = reward / std::max(1, pulls);
      const double exploration =
        std::sqrt(2.0 * std::log(static_cast<double>(std::max(1, pulls))) / std::max(1, pulls));
      return exploitation + exploration;
    }
  };

  enum class RepairOperator { CPUFJ, SUBMIP, BP };
  BanditArm bandit_arms[3];  // Indexed by RepairOperator

  const size_t calibrated_ruin_size = 15;  // Conservative base ruin size
  const int failure_threshold       = 3;   // Trigger escalation after 3 consecutive failures
  const size_t max_ruin_size =
    std::min(calibrated_ruin_size * 4, movable.size());  // Max 4x escalation

  // Bound-propagation cost per branch-and-bound node scales with row density
  double avg_row_nnz = 1.0;
  if (!model.row_lower.empty())
    avg_row_nnz = static_cast<double>(model.coefficients.size()) / model.row_lower.size();
  const double density_scale     = std::min(1.0, 30.0 / std::max(1.0, avg_row_nnz));
  const int base_node_budget     = std::max(100, static_cast<int>(1500 * density_scale));
  const int max_propagate_passes = std::max(5, static_cast<int>(40 * density_scale));

  // Track consecutive failures for VNS escalation
  int consecutive_failures = 0;

  // Persistent best-known solution across the whole run
  std::vector<double> best_known;
  double best_known_cost = std::numeric_limits<double>::infinity();

  while (!stop()) {
    const auto population = snapshot();
    for (const auto& sol : population) {
      const double c = model.cost(sol);
      if (c < best_known_cost) {
        best_known_cost = c;
        best_known      = sol;
      }
    }
    if (best_known.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    // Elitist most of the time: build on the best solution ever seen so
    // that any local improvement is likely to be a genuine new global
    // best. Occasionally diversify from a fresh population member to
    // avoid pure hill-climbing stagnation.
    std::vector<double> current;
    if (!population.empty() &&
        static_cast<int>(rng() % 100) < 10) {  // Reduced diversification rate
      current = population[static_cast<size_t>(rng()) % population.size()];
    } else {
      current = best_known;
    }
    if (!model.normalize_seed(current)) continue;
    double current_cost = model.cost(current);

    // Seed-selection scores are O(nnz) to build; recompute them only when the
    // reference solution actually changes (on improvement) or periodically,
    // instead of on every inner iteration. This is the dominant cost driver
    // for large instances and was previously paid 50x per outer restart.
    std::vector<std::pair<double, int>> variable_scores;
    size_t top_count          = 1;
    const auto refresh_scores = [&]() {
      variable_scores.clear();
      variable_scores.reserve(movable.size());
      for (int j : movable) {
        const double val         = current[j];
        const double obj_contrib = std::abs(model.objective[j] * val);

        // Add constraint activity measure
        double activity = 0.0;
        for (const auto& [r, norm_coeff] : column_rows[j]) {
          activity += std::abs(norm_coeff * val);
        }

        // Combined scoring
        double combined_score = 0.5 * obj_contrib + 0.5 * activity;

        // Bonus for fractional integer variables
        if (model.integer[j] && std::abs(val - std::round(val)) > model.integrality_tolerance) {
          combined_score *= 1.5;
        }

        variable_scores.emplace_back(combined_score, j);
      }
      // Partition (not fully sort) the top 20% to the front: O(n) instead of
      // O(n log n), and selection below only needs membership in that set.
      top_count = std::max(size_t{1}, variable_scores.size() / 5);
      if (!variable_scores.empty()) {
        std::nth_element(variable_scores.begin(),
                         variable_scores.begin() + std::min(top_count, variable_scores.size() - 1),
                         variable_scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
      }
    };
    refresh_scores();
    int iterations_since_refresh = 0;
    const int refresh_period     = 30;

    // Bound each outer restart's inner loop by wall clock, not just an
    // iteration count. On instances with very dense variable-constraint
    // graphs a single ruin-repair episode can cost far more than on sparse
    // ones; a fixed iteration cap left the algorithm running the *entire*
    // remaining time budget on one outer iteration without ever calling
    // snapshot() again, starving population refresh and stop()-responsiveness.
    // Re-checking snapshot()/best_known every couple of seconds keeps the
    // algorithm responsive across instance sizes.
    const int max_inner_iterations = 400;  // Cheaper per-iteration cost now allows many more tries.
    const auto inner_loop_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    for (int iteration = 0; iteration < max_inner_iterations && !stop() &&
                            std::chrono::steady_clock::now() < inner_loop_deadline;
         ++iteration) {
      // VNS escalation: increase ruin size after consecutive failures
      size_t current_ruin_size = calibrated_ruin_size;
      if (consecutive_failures >= failure_threshold) {
        // Linear escalation for stability
        current_ruin_size = std::min(
          max_ruin_size, current_ruin_size + (consecutive_failures - failure_threshold + 1) * 5);
      }

      const size_t ruin_size = std::min(current_ruin_size, movable.size());

      if (++iterations_since_refresh >= refresh_period) {
        refresh_scores();
        iterations_since_refresh = 0;
      }

      // Select seed from top performers
      int seed_var;
      if (!variable_scores.empty()) {
        // Select from top 20% with bias toward higher scores
        const size_t selection_idx =
          std::min(static_cast<size_t>(rng()) % top_count, top_count - 1);
        seed_var = variable_scores[selection_idx].second;
      } else {
        seed_var = movable[static_cast<size_t>(rng()) % movable.size()];
      }

      std::vector<int> ruined;
      ruined.push_back(seed_var);

      std::vector<bool> chosen(n, false);
      chosen[ruined.front()] = true;

      // Simplified similarity calculation
      while (ruined.size() < ruin_size) {
        const int last = ruined.back();

        // Collect candidates from shared constraints
        std::vector<std::pair<double, int>> candidates;
        std::vector<int> candidate_vars;

        // Collect all variables connected through shared constraints
        for (const auto& [r, norm_coeff] : column_rows[last]) {
          for (int p = model.offsets[r]; p < model.offsets[r + 1]; ++p) {
            const int j = model.columns[p];
            if (!chosen[j] && model.lower[j] < model.upper[j]) { candidate_vars.push_back(j); }
          }
        }

        // Remove duplicates and limit size
        std::sort(candidate_vars.begin(), candidate_vars.end());
        candidate_vars.erase(std::unique(candidate_vars.begin(), candidate_vars.end()),
                             candidate_vars.end());

        if (candidate_vars.size() > 100) candidate_vars.resize(100);

        // Calculate simplified similarity scores
        for (int j : candidate_vars) {
          const auto& left_col  = column_rows[last];
          const auto& right_col = column_rows[j];

          // Simple shared constraint count similarity
          size_t a = 0, b = 0, shared_constraints = 0;
          while (a < left_col.size() && b < right_col.size()) {
            if (left_col[a].first < right_col[b].first) {
              ++a;
              continue;
            }
            if (right_col[b].first < left_col[a].first) {
              ++b;
              continue;
            }
            ++shared_constraints;
            ++a;
            ++b;
          }

          // Value difference similarity
          double value_diff       = std::abs(current[last] - current[j]);
          double value_similarity = std::exp(-value_diff);

          // Combined similarity (favoring shared constraints)
          double similarity_score = 0.7 * shared_constraints + 0.3 * value_similarity;

          candidates.emplace_back(similarity_score, j);
        }

        if (candidates.empty()) break;

        // Sort candidates by similarity (higher is better)
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
          return a.first > b.first;
        });

        // Select best candidate
        int selected_candidate = -1;
        if (!candidates.empty()) {
          if (static_cast<int>(rng() % 100) < 80 && candidates[0].first > 1e-6) {
            selected_candidate = candidates[0].second;
          } else {
            const size_t selection_range = std::min(size_t{3}, candidates.size());
            if (selection_range > 0) {
              const size_t idx   = static_cast<size_t>(rng()) % selection_range;
              selected_candidate = candidates[idx].second;
            }
          }
        }

        // Fallback to random selection if needed
        if (selected_candidate < 0) {
          const size_t offset = static_cast<size_t>(rng()) % movable.size();
          for (size_t k = 0; k < movable.size(); ++k) {
            const int j = movable[(offset + k) % movable.size()];
            if (!chosen[j]) {
              selected_candidate = j;
              break;
            }
          }
        }

        if (selected_candidate < 0) break;
        ruined.push_back(selected_candidate);
        chosen[selected_candidate] = true;
      }

      // Simplified domain expansion
      auto lower = current, upper = current;
      std::vector<int> rows;
      bool valid_domain = true;

      // Domain expansion, scaled to each variable's own range so that
      // continuous variables with wide domains are not stuck with a tiny
      // fixed window while tightly-bounded variables are not over-relaxed.
      for (int j : ruined) {
        const double domain_width = model.upper[j] - model.lower[j];
        double expansion;
        if (model.integer[j]) {
          expansion = 2.0;
        } else if (std::isfinite(domain_width) && domain_width < 1e12) {
          expansion = std::max(2.0, 0.2 * domain_width);
        } else {
          expansion = std::max(2.0, 0.1 * std::abs(current[j]) + 1.0);
        }

        const double val = current[j];
        lower[j]         = std::max(model.lower[j], val - expansion);
        upper[j]         = std::min(model.upper[j], val + expansion);

        if (model.integer[j]) {
          lower[j] = std::ceil(lower[j]);
          upper[j] = std::floor(upper[j]);
        }

        if (lower[j] > upper[j]) {
          valid_domain = false;
          break;
        }

        // Collect affected rows
        for (const auto& entry : column_rows[j]) {
          rows.push_back(entry.first);
        }
      }

      if (!valid_domain || rows.empty()) continue;
      std::sort(rows.begin(), rows.end());
      rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

      // Tighten every touched row to a fixpoint. Record only actual bound
      // mutations so branch-and-propagate can undo them without copying the
      // full model-sized bound vectors at every node.
      struct bound_change_t {
        int variable;
        double old_lo;
        double old_hi;
      };
      const auto propagate =
        [&](std::vector<double>& lo, std::vector<double>& hi, std::vector<bound_change_t>& trail) {
          bool changed = true;
          int passes   = 0;
          while (changed && passes++ < max_propagate_passes) {
            changed = false;
            for (int r : rows) {
              double min_activity = 0.0, max_activity = 0.0;
              for (int p = model.offsets[r]; p < model.offsets[r + 1]; ++p) {
                const int k    = model.columns[p];
                const double a = model.coefficients[p];
                min_activity += a * (a >= 0.0 ? lo[k] : hi[k]);
                max_activity += a * (a >= 0.0 ? hi[k] : lo[k]);
              }
              if (min_activity > model.row_upper[r] + model.row_tolerances[r] ||
                  max_activity < model.row_lower[r] - model.row_tolerances[r])
                return false;
              for (int p = model.offsets[r]; p < model.offsets[r + 1]; ++p) {
                const int k    = model.columns[p];
                const double a = model.coefficients[p];
                if (!chosen[k] || a == 0.0 || lo[k] == hi[k]) continue;
                const double other_min = min_activity - a * (a >= 0.0 ? lo[k] : hi[k]);
                const double other_max = max_activity - a * (a >= 0.0 ? hi[k] : lo[k]);
                double new_lo          = (model.row_lower[r] - other_max) / a;
                double new_hi          = (model.row_upper[r] - other_min) / a;
                if (a < 0.0) std::swap(new_lo, new_hi);
                if (model.integer[k]) {
                  new_lo = std::ceil(new_lo - model.integrality_tolerance);
                  new_hi = std::floor(new_hi + model.integrality_tolerance);
                }
                new_lo = std::max(lo[k], new_lo);
                new_hi = std::min(hi[k], new_hi);
                if (new_lo > new_hi) return false;
                if (new_lo > lo[k] || new_hi < hi[k]) {
                  trail.push_back({k, lo[k], hi[k]});
                  lo[k]   = new_lo;
                  hi[k]   = new_hi;
                  changed = true;
                }
              }
            }
          }
          return !changed;
        };

      // Simplified bandit selection of repair operator
      RepairOperator selected_operator = RepairOperator::BP;  // Default to BP
      if (ruined.size() <= 25) {
        // Use bandit selection for small neighborhoods
        double best_ucb = -1e6;
        for (int op = 0; op < 3; ++op) {
          double ucb = bandit_arms[op].ucb();
          if (ucb > best_ucb) {
            best_ucb          = ucb;
            selected_operator = static_cast<RepairOperator>(op);
          }
        }
      }

      bool improvement_found = false;

      // Execute selected repair operator
      if (selected_operator == RepairOperator::CPUFJ) {
        // Use CPUFJ repair for medium-sized neighborhoods
        if (ruined.size() <= 30) {
          repair_request_t request;
          request.start              = current;
          request.lower              = lower;
          request.upper              = upper;
          request.time_limit_seconds = 0.15;
          request.seed               = static_cast<uint64_t>(rng());
          request.max_iterations     = 3000;
          request.max_nodes          = 200;

          auto result = model.repair.cpufj(request);
          if (result.feasible && result.objective < current_cost) {
            current           = result.assignment;
            current_cost      = result.objective;
            improvement_found = true;

            // Update bandit reward
            bandit_arms[0].pulls++;
            bandit_arms[0].reward += (current_cost < best_known_cost) ? 2.0 : 1.0;

            if (result.objective < best_known_cost) {
              submit(result.assignment);
              best_known      = result.assignment;
              best_known_cost = result.objective;
            }
          } else {
            // Update bandit with small penalty
            bandit_arms[0].pulls++;
            bandit_arms[0].reward -= 0.1;
          }
        } else {
          // Fall back to BP for large neighborhoods
          selected_operator = RepairOperator::BP;
        }
      }

      if (selected_operator == RepairOperator::SUBMIP) {
        // Use SubMIP repair for small neighborhoods
        if (ruined.size() <= 20) {
          repair_request_t request;
          request.start              = current;
          request.lower              = lower;
          request.upper              = upper;
          request.time_limit_seconds = 0.2;
          request.seed               = static_cast<uint64_t>(rng());
          request.max_iterations     = 5000;
          request.max_nodes          = 500;

          auto result = model.repair.submip(request);
          if (result.feasible && result.objective < current_cost) {
            current           = result.assignment;
            current_cost      = result.objective;
            improvement_found = true;

            // Update bandit reward
            bandit_arms[1].pulls++;
            bandit_arms[1].reward += (current_cost < best_known_cost) ? 3.0 : 1.5;

            if (result.objective < best_known_cost) {
              submit(result.assignment);
              best_known      = result.assignment;
              best_known_cost = result.objective;
            }
          } else {
            // Update bandit with small penalty
            bandit_arms[1].pulls++;
            bandit_arms[1].reward -= 0.15;
          }
        } else {
          // Fall back to BP for large neighborhoods
          selected_operator = RepairOperator::BP;
        }
      }

      // A single propagate() call costs O(passes * touched-rows * row-nnz);
      // the per-node wall-clock deadline below only bounds cost *between*
      // propagate() calls, not cost accrued *inside* one call. On instances
      // with a few very high-degree columns (e.g. dense linking rows), a
      // ruin set can pull in an enormous touched-row set and make even one
      // propagate() call pathologically slow. Bound that directly by
      // skipping BP outright for oversized touched-row sets; CPUFJ/SUBMIP
      // remain available since their cost does not scale with this set.
      if (selected_operator == RepairOperator::BP && rows.size() > 4000) {
        bandit_arms[2].pulls++;
        bandit_arms[2].reward -= 0.05;
      } else if (selected_operator == RepairOperator::BP) {
        // Use Branch-and-Propagate (default BP) repair
        int nodes = 0;
        // Adjust node budget inversely with ruin size
        const int node_budget = static_cast<int>(base_node_budget * calibrated_ruin_size /
                                                 std::max(current_ruin_size, size_t{1}));

        // The full-size bounds are allocated once for this ruin episode. DFS
        // nodes mutate them in place and restore only changed ruined variables.
        std::vector<double>& lo = lower;
        std::vector<double>& hi = upper;
        std::vector<bound_change_t> trail;
        trail.reserve(ruined.size() * 4);

        // Objective contribution of all non-ruined variables is fixed for the
        // entire DFS episode (their lo==hi==current value). Precompute it once
        // by summing over the non-ruined variables, then only add the
        // optimistic contribution of the ruined variables at each node:
        // O(|ruined|) per node instead of O(n).
        double fixed_objective = 0.0;
        for (size_t j = 0; j < n; ++j)
          fixed_objective += model.objective[j] * current[j];
        for (int j : ruined)
          fixed_objective -= model.objective[j] * current[j];
        const auto rollback = [&](size_t mark) {
          while (trail.size() > mark) {
            const bound_change_t change = trail.back();
            trail.pop_back();
            lo[change.variable] = change.old_lo;
            hi[change.variable] = change.old_hi;
          }
        };

        // A per-episode wall-clock deadline caps the *actual* cost of this
        // ruin-repair attempt regardless of how expensive each propagate()
        // call turns out to be (dense variable-constraint graphs can make a
        // single node far more costly than the node-count budget assumes).
        const auto search_deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
        std::function<void()> search;
        search = [&]() {
          const size_t node_mark = trail.size();
          if (nodes++ >= node_budget || stop() ||
              std::chrono::steady_clock::now() >= search_deadline || !propagate(lo, hi, trail)) {
            rollback(node_mark);
            return;
          }

          // Pruning based on objective bound: fixed part plus the optimistic
          // contribution of only the ruined variables (O(|ruined|)).
          double objective_bound = fixed_objective;
          for (int j : ruined)
            objective_bound += model.objective[j] * (model.objective[j] >= 0.0 ? lo[j] : hi[j]);
          if (objective_bound >= current_cost - 1e-9) {
            rollback(node_mark);
            return;
          }

          int branch             = -1;
          double smallest_domain = std::numeric_limits<double>::infinity();
          for (int j : ruined) {
            if (hi[j] <= lo[j]) continue;
            const double domain = model.integer[j] ? hi[j] - lo[j] + 1.0 : 3.0;
            if (domain < smallest_domain) {
              smallest_domain = domain;
              branch          = j;
            }
          }
          if (branch < 0) {
            // Recheck improving leaves with the same tolerances as publication;
            // propagation arithmetic alone does not certify the full assignment.
            {
              // Leaf cost: fixed part (unruined variables) plus the exact
              // contribution of the now-fixed ruined variables. O(|ruined|)
              // instead of an O(n) full re-scan.
              double cost = fixed_objective;
              for (int j : ruined)
                cost += model.objective[j] * lo[j];
              if (cost < current_cost - 1e-9 && model.feasible(lo)) {
                current           = lo;
                current_cost      = cost;
                improvement_found = true;

                // Update bandit reward for BP
                bandit_arms[2].pulls++;
                bandit_arms[2].reward += (current_cost < best_known_cost) ? 1.5 : 0.5;

                if (cost < best_known_cost) {
                  submit(lo);
                  best_known      = lo;
                  best_known_cost = cost;
                }
              }
            }
            rollback(node_mark);
            return;
          }

          std::vector<double> values;
          if (model.integer[branch]) {
            for (double v = lo[branch]; v <= hi[branch] + 0.5; v += 1.0)
              values.push_back(v);
          } else {
            values = {lo[branch], std::clamp(current[branch], lo[branch], hi[branch]), hi[branch]};
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
          }
          if (model.objective[branch] < 0.0) std::reverse(values.begin(), values.end());
          for (double value : values) {
            if (nodes >= node_budget || stop()) break;
            const size_t child_mark = trail.size();
            trail.push_back({branch, lo[branch], hi[branch]});
            lo[branch] = hi[branch] = value;
            search();
            rollback(child_mark);
          }
          rollback(node_mark);
        };
        search();

        if (!improvement_found) {
          // No improvement found within the node/time budget for this episode.
          bandit_arms[2].pulls++;
          bandit_arms[2].reward -= 0.05;
        }
      }

      // Update failure counter based on whether improvement was found
      if (improvement_found) {
        // A backend can return tolerance-feasible values. Recheck its normalized
        // copy before the next iteration uses it for exact fixings.
        if (!model.normalize_seed(current)) break;
        current_cost         = model.cost(current);
        consecutive_failures = 0;
        // The reference solution moved: keep the seed-selection scores fresh.
        refresh_scores();
        iterations_since_refresh = 0;
      } else {
        consecutive_failures++;
      }
    }
  }
}
}  // namespace cuopt::hive_lns
