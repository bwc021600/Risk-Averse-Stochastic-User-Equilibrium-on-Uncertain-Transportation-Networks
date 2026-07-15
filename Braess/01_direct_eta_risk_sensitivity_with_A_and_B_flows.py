from __future__ import annotations
import argparse
import json
import math
import platform
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import matplotlib

def _running_in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False
if not _running_in_jupyter():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as opt
from scipy.optimize import linprog
Q_DEMAND = 80.0
THETA = 0.6
R0 = 1.0
BPR_BETA = 0.15
BPR_GAMMA = 4.0
ALPHA_MAIN = 0.9
ETA_MAIN = 0.5
LINKS = ('AB', 'AC', 'BC', 'BD', 'CD')
LINK_LABELS = {'AB': 'A→B', 'AC': 'A→C', 'BC': 'B→C', 'BD': 'B→D', 'CD': 'C→D'}
BASE_LINK_PARAMS = {'AB': (3.0, 130.0, 0.0), 'AC': (7.5, 85.0, 0.0), 'BC': (0.8, 120.0, 0.0), 'BD': (14.0, 65.0, 0.0), 'CD': (8.0, 110.0, 0.0)}
SCENARIOS = ('Normal', 'Upper incident', 'Lower incident', 'Shortcut incident', 'System-wide')
SCENARIO_PROBS = np.array([0.55, 0.18, 0.17, 0.06, 0.04], dtype=float)

@dataclass(frozen=True)
class LinkScenarioParams:
    t0: float
    capacity: float
    delay: float

@dataclass(frozen=True)
class PathSet:
    name: str
    path_names: tuple[str, ...]
    incidence: np.ndarray

    @property
    def n_paths(self) -> int:
        return len(self.path_names)

    def validate(self) -> None:
        if self.incidence.shape != (self.n_paths, len(LINKS)):
            raise ValueError(f'Invalid incidence shape for {self.name}: {self.incidence.shape}.')
        if not np.all((self.incidence == 0) | (self.incidence == 1)):
            raise ValueError('Path-link incidence must be binary.')
        if np.any(self.incidence.sum(axis=1) <= 0):
            raise ValueError('Every path must use at least one link.')

@dataclass(frozen=True)
class BraessInstance:
    name: str
    scenario_names: tuple[str, ...]
    probabilities: np.ndarray
    link_params: dict[str, dict[str, LinkScenarioParams]]

@dataclass(frozen=True)
class SolverConfig:
    q: float = Q_DEMAND
    theta: float = THETA
    r0: float = R0
    alpha: float = ALPHA_MAIN
    eta: float = ETA_MAIN
    bpr_beta: float = BPR_BETA
    bpr_gamma: float = BPR_GAMMA

@dataclass(frozen=True)
class EquilibriumResult:
    instance: str
    path_set: str
    model: str
    overlap_model: str
    beta_ps: float
    flows: np.ndarray
    perceived_costs: np.ndarray
    raw_costs: np.ndarray
    overlap_penalty: np.ndarray
    probabilities: np.ndarray
    pi_od: float | None
    link_flows: np.ndarray
    iterations: int
    residual: float
    active_paths: tuple[str, ...]
    cpu_seconds: float

def base_path_set() -> PathSet:
    incidence = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 1]], dtype=float)
    return PathSet(name='Base-3 paths', path_names=('P1: A→B→D', 'P2: A→C→D', 'P3: A→B→C→D'), incidence=incidence)

def cloned_p1_path_set() -> PathSet:
    incidence = np.array([[1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 1]], dtype=float)
    return PathSet(name='Clone-4 paths', path_names=('P1: A→B→D', 'P1′ clone: A→B→D', 'P2: A→C→D', 'P3: A→B→C→D'), incidence=incidence)

def build_common_tail_instance() -> BraessInstance:
    severity_schedule = {'Normal': (1.0, 1.0, 0.0), 'Upper incident': (1.04, 0.92, 1.5), 'Lower incident': (1.08, 0.84, 3.5), 'Shortcut incident': (1.12, 0.76, 6.0), 'System-wide': (1.18, 0.65, 10.0)}
    params: dict[str, dict[str, LinkScenarioParams]] = {}
    for scenario in SCENARIOS:
        t_mult, cap_mult, delay_add = severity_schedule[scenario]
        params[scenario] = {}
        for link, (t0, capacity, delay) in BASE_LINK_PARAMS.items():
            params[scenario][link] = LinkScenarioParams(t0=t0 * t_mult, capacity=capacity * cap_mult, delay=delay + delay_add)
    return BraessInstance(name='Common-tail Braess', scenario_names=SCENARIOS, probabilities=SCENARIO_PROBS.copy(), link_params=params)

def build_route_local_instance() -> BraessInstance:
    params: dict[str, dict[str, LinkScenarioParams]] = {}
    for scenario in SCENARIOS:
        params[scenario] = {}
        for link, (t0, capacity, delay) in BASE_LINK_PARAMS.items():
            t_mult, cap_mult, delay_add = (1.0, 1.0, 0.0)
            if scenario == 'Upper incident':
                if link == 'BD':
                    t_mult, cap_mult, delay_add = (1.18, 0.55, 16.0)
                elif link == 'AB':
                    t_mult, cap_mult, delay_add = (1.08, 0.78, 4.0)
                else:
                    t_mult, cap_mult, delay_add = (1.02, 0.96, 0.3)
            elif scenario == 'Lower incident':
                if link == 'AC':
                    t_mult, cap_mult, delay_add = (1.14, 0.62, 10.0)
                elif link == 'CD':
                    t_mult, cap_mult, delay_add = (1.08, 0.75, 4.0)
                else:
                    t_mult, cap_mult, delay_add = (1.02, 0.96, 0.3)
            elif scenario == 'Shortcut incident':
                if link == 'BC':
                    t_mult, cap_mult, delay_add = (1.25, 0.35, 24.0)
                else:
                    t_mult, cap_mult, delay_add = (1.01, 0.98, 0.2)
            elif scenario == 'System-wide':
                t_mult, cap_mult, delay_add = (1.22, 0.58, 10.0)
            params[scenario][link] = LinkScenarioParams(t0=t0 * t_mult, capacity=capacity * cap_mult, delay=delay + delay_add)
    return BraessInstance(name='Route-local tail Braess', scenario_names=SCENARIOS, probabilities=SCENARIO_PROBS.copy(), link_params=params)

def validate_instance(instance: BraessInstance) -> None:
    if not np.isclose(instance.probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError('Scenario probabilities must sum to one.')
    if np.any(instance.probabilities < 0):
        raise ValueError('Scenario probabilities must be nonnegative.')
    for scenario in instance.scenario_names:
        if scenario not in instance.link_params:
            raise ValueError(f'Missing scenario {scenario}.')
        for link in LINKS:
            prm = instance.link_params[scenario][link]
            if prm.t0 <= 0 or prm.capacity <= 0 or prm.delay < 0:
                raise ValueError(f'Invalid parameters for scenario={scenario}, link={link}: {prm}.')

def link_flows_from_path_flows(path_set: PathSet, path_flows: Sequence[float]) -> np.ndarray:
    f = np.asarray(path_flows, dtype=float)
    return path_set.incidence.T @ f

def link_times(instance: BraessInstance, path_set: PathSet, link_flows: Sequence[float], scenario: str, cfg: SolverConfig) -> np.ndarray:
    x = np.asarray(link_flows, dtype=float)
    times = []
    for j, link in enumerate(LINKS):
        prm = instance.link_params[scenario][link]
        times.append(prm.t0 * (1.0 + cfg.bpr_beta * (x[j] / prm.capacity) ** cfg.bpr_gamma) + prm.delay)
    return np.asarray(times, dtype=float)

def path_times_by_scenario(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> np.ndarray:
    x = link_flows_from_path_flows(path_set, path_flows)
    return np.vstack([path_set.incidence @ link_times(instance, path_set, x, scenario, cfg) for scenario in instance.scenario_names])

def link_potential_integral(x: float, prm: LinkScenarioParams, cfg: SolverConfig) -> float:
    return prm.t0 * (x + cfg.bpr_beta * x ** (cfg.bpr_gamma + 1.0) / ((cfg.bpr_gamma + 1.0) * prm.capacity ** cfg.bpr_gamma)) + prm.delay * x

def scenario_potential(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], scenario: str, cfg: SolverConfig) -> float:
    x = link_flows_from_path_flows(path_set, path_flows)
    return float(sum((link_potential_integral(x[j], instance.link_params[scenario][link], cfg) for j, link in enumerate(LINKS))))

def potentials_by_scenario(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> np.ndarray:
    return np.asarray([scenario_potential(instance, path_set, path_flows, s, cfg) for s in instance.scenario_names], dtype=float)

def entropy(path_flows: Sequence[float], cfg: SolverConfig) -> float:
    f = np.maximum(np.asarray(path_flows, dtype=float), 0.0)
    return float(np.sum((f + cfg.r0) * np.log((f + cfg.r0) / cfg.r0) - f))

def eta_from_lambda(alpha: float, lam: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie in (0,1).')
    if not 0.0 <= lam <= alpha:
        raise ValueError('lambda must satisfy 0 <= lambda <= alpha.')
    denom = 1.0 + alpha - 2.0 * lam
    if denom <= 0.0:
        raise ValueError('Invalid alpha/lambda denominator.')
    return (alpha - lam) / denom

def finite_cvar_selector(values: Sequence[float], probs: Sequence[float], alpha: float) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie in (0,1).')
    v = np.asarray(values, dtype=float)
    p = np.asarray(probs, dtype=float)
    if v.shape != p.shape:
        raise ValueError('values and probs must have the same shape.')
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-12):
        raise ValueError('probabilities must be nonnegative and sum to one.')
    order = np.argsort(v)
    cumulative = np.cumsum(p[order])
    boundary_pos = int(np.searchsorted(cumulative, alpha, side='left'))
    boundary_pos = min(boundary_pos, len(v) - 1)
    chi = np.zeros_like(v, dtype=float)
    if boundary_pos + 1 < len(v):
        chi[order[boundary_pos + 1:]] = 1.0
    boundary_index = order[boundary_pos]
    chi[boundary_index] = (cumulative[boundary_pos] - alpha) / p[boundary_index]
    return np.clip(chi, 0.0, 1.0)

def cvar_value(values: Sequence[float], probs: Sequence[float], alpha: float) -> float:
    v = np.asarray(values, dtype=float)
    p = np.asarray(probs, dtype=float)
    chi = finite_cvar_selector(v, p, alpha)
    return float(np.dot(p * chi, v) / (1.0 - alpha))

def mean_cvar_value_and_tilt(values: Sequence[float], probs: Sequence[float], alpha: float, eta: float) -> tuple[float, np.ndarray, np.ndarray]:
    if not 0.0 <= eta <= 1.0:
        raise ValueError('eta must lie in [0,1].')
    v = np.asarray(values, dtype=float)
    p = np.asarray(probs, dtype=float)
    chi = finite_cvar_selector(v, p, alpha)
    mean = float(np.dot(p, v))
    cvar = float(np.dot(p * chi, v) / (1.0 - alpha))
    tilted = p * (1.0 - eta + eta * chi / (1.0 - alpha))
    tilted = tilted / tilted.sum()
    return ((1.0 - eta) * mean + eta * cvar, chi, tilted)

def nominal_link_lengths(instance: BraessInstance) -> np.ndarray:
    return np.asarray([instance.link_params['Normal'][link].t0 for link in LINKS], dtype=float)

def path_size_factors(instance: BraessInstance, path_set: PathSet) -> np.ndarray:
    lengths = nominal_link_lengths(instance)
    incidence = path_set.incidence.astype(float)
    path_lengths = incidence @ lengths
    link_counts = incidence.sum(axis=0)
    link_counts_safe = np.where(link_counts > 0.0, link_counts, 1.0)
    factors = []
    for k in range(path_set.n_paths):
        if path_lengths[k] <= 0.0:
            raise ValueError('Path length must be positive.')
        ps = float(np.sum(incidence[k, :] * (lengths / path_lengths[k]) / link_counts_safe))
        factors.append(max(ps, 1e-12))
    return np.asarray(factors, dtype=float)

def overlap_penalty(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, beta_ps: float=0.0, overlap_model: str='none') -> np.ndarray:
    if overlap_model == 'none' or beta_ps == 0.0:
        return np.zeros(path_set.n_paths, dtype=float)
    if overlap_model != 'path-size':
        raise ValueError(f'Unknown overlap_model={overlap_model}.')
    ps = path_size_factors(instance, path_set)
    return -(beta_ps / cfg.theta) * np.log(ps)

def risk_neutral_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> np.ndarray:
    return instance.probabilities @ path_times_by_scenario(instance, path_set, path_flows, cfg)

def model_a_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = path_times_by_scenario(instance, path_set, path_flows, cfg)
    costs, selectors, tilts = ([], [], [])
    for k in range(path_set.n_paths):
        ce, chi, tilted = mean_cvar_value_and_tilt(pt[:, k], instance.probabilities, cfg.alpha, cfg.eta)
        costs.append(ce)
        selectors.append(chi)
        tilts.append(tilted)
    return (np.asarray(costs), np.asarray(selectors), np.asarray(tilts))

def model_b_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = potentials_by_scenario(instance, path_set, path_flows, cfg)
    _, chi, tilted = mean_cvar_value_and_tilt(vals, instance.probabilities, cfg.alpha, cfg.eta)
    costs = tilted @ path_times_by_scenario(instance, path_set, path_flows, cfg)
    return (np.asarray(costs), chi, tilted)

def add_overlap(raw_costs: np.ndarray, instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, beta_ps: float, overlap_model: str) -> tuple[np.ndarray, np.ndarray]:
    penalty = overlap_penalty(instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    return (np.asarray(raw_costs, dtype=float) + penalty, penalty)

def solve_endogenous_pi(costs: Sequence[float], cfg: SolverConfig) -> float:
    c = np.asarray(costs, dtype=float)
    if not np.all(np.isfinite(c)):
        raise ValueError('costs must be finite.')
    lower = float(np.min(c))
    upper = lower + math.log1p(cfg.q / cfg.r0) / cfg.theta
    for _ in range(250):
        mid = 0.5 * (lower + upper)
        flows = cfg.r0 * np.maximum(np.expm1(cfg.theta * (mid - c)), 0.0)
        if flows.sum() < cfg.q:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)

def truncated_flows_from_costs(costs: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, float]:
    c = np.asarray(costs, dtype=float)
    pi = solve_endogenous_pi(c, cfg)
    flows = cfg.r0 * np.maximum(np.expm1(cfg.theta * (pi - c)), 0.0)
    if abs(flows.sum() - cfg.q) > 1e-07:
        raise RuntimeError(f'Demand conservation failed: {flows.sum() - cfg.q:.3e}')
    return (flows, flows / cfg.q, pi)

def ordinary_logit_flows_from_costs(costs: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(costs, dtype=float)
    z = -cfg.theta * (c - np.min(c))
    w = np.exp(z)
    probs = w / w.sum()
    return (cfg.q * probs, probs)

def solve_fixed_point(instance: BraessInstance, path_set: PathSet, model: str, raw_cost_function: Callable[[np.ndarray], np.ndarray], cfg: SolverConfig, truncated: bool=True, overlap_model: str='none', beta_ps: float=0.0, relaxation: float=0.2, tol: float=1e-09, max_iter: int=80000, initial_flows: np.ndarray | None=None) -> EquilibriumResult:
    if initial_flows is None:
        flows = np.full(path_set.n_paths, cfg.q / path_set.n_paths, dtype=float)
    else:
        flows = np.maximum(np.asarray(initial_flows, dtype=float), 0.0)
        if flows.sum() <= 0:
            flows[:] = cfg.q / path_set.n_paths
        else:
            flows *= cfg.q / flows.sum()
    pi: float | None = None
    residual = np.inf
    start = time.perf_counter()
    target_flows = flows.copy()
    for iteration in range(1, max_iter + 1):
        raw = raw_cost_function(flows)
        perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
        if truncated:
            target_flows, _, pi = truncated_flows_from_costs(perceived, cfg)
        else:
            target_flows, _ = ordinary_logit_flows_from_costs(perceived, cfg)
            pi = None
        residual = float(np.max(np.abs(target_flows - flows)))
        if residual <= tol:
            flows = target_flows
            break
        flows = (1.0 - relaxation) * flows + relaxation * target_flows
    else:
        raise RuntimeError(f'Fixed point failed for {path_set.name}, {model}, beta_ps={beta_ps}; residual={residual:.3e}.')
    raw = raw_cost_function(flows)
    perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    if truncated:
        flows, probs, pi = truncated_flows_from_costs(perceived, cfg)
    else:
        flows, probs = ordinary_logit_flows_from_costs(perceived, cfg)
        pi = None
    elapsed = time.perf_counter() - start
    link_flows = link_flows_from_path_flows(path_set, flows)
    active = tuple((path_set.path_names[i] for i, val in enumerate(flows) if val > 1e-06))
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model=model, overlap_model=overlap_model, beta_ps=beta_ps, flows=flows, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=probs, pi_od=pi, link_flows=link_flows, iterations=iteration, residual=float(np.max(np.abs(flows - target_flows))), active_paths=active, cpu_seconds=elapsed)

def solve_core_models(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, overlap_model: str='none', beta_ps: float=0.0) -> dict[str, EquilibriumResult]:
    return {'Ordinary logit SUE': solve_fixed_point(instance, path_set, 'Ordinary logit SUE', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg), cfg, truncated=False, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2), 'Risk-neutral truncated TSUE': solve_fixed_point(instance, path_set, 'Risk-neutral truncated TSUE', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg), cfg, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2), 'Model A mean-CVaR TSUE': solve_fixed_point(instance, path_set, 'Model A mean-CVaR TSUE', lambda f: model_a_raw_costs(instance, path_set, f, cfg)[0], cfg, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.1), 'Model B common-state TSUE': solve_model_b_for_params_labeled(instance, path_set, cfg, cfg.alpha, cfg.eta, overlap_model=overlap_model, beta_ps=beta_ps)}

def vi_gap_model_a(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig, overlap_model: str='none', beta_ps: float=0.0) -> float:
    f = np.asarray(path_flows, dtype=float)
    raw = model_a_raw_costs(instance, path_set, f, cfg)[0]
    perceived, _ = add_overlap(raw, instance, path_set, cfg, beta_ps, overlap_model)
    mapping = perceived + 1.0 / cfg.theta * np.log1p(f / cfg.r0)
    gap = float(np.dot(mapping, f) - cfg.q * np.min(mapping))
    return gap / (1.0 + abs(float(np.dot(mapping, f))))

def model_a_b_metrics(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig) -> dict[str, float | str]:
    results = solve_core_models(instance, path_set, cfg, overlap_model='none', beta_ps=0.0)
    f_a = results['Model A mean-CVaR TSUE'].flows
    f_b = results['Model B common-state TSUE'].flows
    _, _, tilts_a_at_b = model_a_raw_costs(instance, path_set, f_b, cfg)
    _, _, tilt_b_at_b = model_b_raw_costs(instance, path_set, f_b, cfg)
    d_f = float(np.sum(np.abs(f_a - f_b)) / cfg.q)
    d_p = float(max((np.sum(np.abs(tilts_a_at_b[k] - tilt_b_at_b)) for k in range(path_set.n_paths))))
    gap_a = vi_gap_model_a(instance, path_set, f_b, cfg)
    active_diff = len(results['Model A mean-CVaR TSUE'].active_paths) - len(results['Model B common-state TSUE'].active_paths)
    interpretation = 'Exact' if d_f < 1e-05 and d_p < 1e-05 and (gap_a < 1e-05) else 'Not exact'
    return {'Instance': instance.name, 'PathSet': path_set.name, 'D_p_AB': d_p, 'D_f_AB': d_f, 'Gap_A_at_fB': gap_a, 'Active_A_minus_Active_B': active_diff, 'Interpretation': interpretation}

def b1b2_values_for_ru(instance: BraessInstance, path_set: PathSet, f: Sequence[float], cfg: SolverConfig, entropy_inside: bool) -> np.ndarray:
    values = potentials_by_scenario(instance, path_set, f, cfg)
    if entropy_inside:
        values = values + entropy(f, cfg) / cfg.theta
    return values

def objective_b1b2_x(x: np.ndarray, instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alpha: float, lam: float, entropy_inside: bool) -> float:
    n = path_set.n_paths
    f = np.asarray(x[:n], dtype=float)
    gamma = float(x[n])
    u = np.asarray(x[n + 1:], dtype=float)
    eta = eta_from_lambda(alpha, lam)
    values = b1b2_values_for_ru(instance, path_set, f, cfg, entropy_inside)
    obj = (1.0 - eta) * float(np.dot(instance.probabilities, values))
    obj += eta * gamma + eta / (1.0 - alpha) * float(np.dot(instance.probabilities, u))
    if not entropy_inside:
        obj += entropy(f, cfg) / cfg.theta
    return obj

def initial_b1b2_x(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alpha: float, lam: float, entropy_inside: bool, f0: Sequence[float] | None=None) -> np.ndarray:
    if f0 is None:
        f = np.full(path_set.n_paths, cfg.q / path_set.n_paths, dtype=float)
    else:
        f = np.maximum(np.asarray(f0, dtype=float), 0.0)
        if f.sum() <= 0.0:
            f[:] = cfg.q / path_set.n_paths
        else:
            f *= cfg.q / f.sum()
    values = b1b2_values_for_ru(instance, path_set, f, cfg, entropy_inside)
    gamma = float(np.dot(instance.probabilities, values))
    u = np.maximum(values - gamma, 0.0)
    return np.concatenate([f, [gamma], u])

def solve_b1b2_convex_sp(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alpha: float, lam: float, entropy_inside: bool, x0: np.ndarray | None=None) -> tuple[np.ndarray, opt.OptimizeResult]:
    n = path_set.n_paths
    s_count = len(instance.scenario_names)
    if x0 is None:
        x0 = initial_b1b2_x(instance, path_set, cfg, alpha, lam, entropy_inside)
    bounds = opt.Bounds([0.0] * n + [-np.inf] + [0.0] * s_count, [np.inf] * n + [np.inf] + [np.inf] * s_count)
    cons: list[dict[str, object]] = [{'type': 'eq', 'fun': lambda x: float(np.sum(x[:n]) - cfg.q)}]

    def excess(x: np.ndarray, s_idx: int) -> float:
        f = x[:n]
        gamma = x[n]
        values = b1b2_values_for_ru(instance, path_set, f, cfg, entropy_inside)
        return float(x[n + 1 + s_idx] - (values[s_idx] - gamma))
    for s_idx in range(s_count):
        cons.append({'type': 'ineq', 'fun': lambda x, s_idx=s_idx: excess(x, s_idx)})
    res = opt.minimize(lambda x: objective_b1b2_x(x, instance, path_set, cfg, alpha, lam, entropy_inside), x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-11, 'maxiter': 3000, 'disp': False})
    if not res.success:
        fallback = initial_b1b2_x(instance, path_set, cfg, alpha, lam, entropy_inside)
        res2 = opt.minimize(lambda x: objective_b1b2_x(x, instance, path_set, cfg, alpha, lam, entropy_inside), fallback, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-11, 'maxiter': 5000, 'disp': False})
        if res2.success or res2.fun < res.fun:
            res = res2
    if not res.success:
        raise RuntimeError(f"B{('1' if entropy_inside else '2')} SP failed for alpha={alpha}, lambda={lam}: {res.message}")
    f = np.maximum(np.asarray(res.x[:n], dtype=float), 0.0)
    if abs(f.sum() - cfg.q) > 1e-07:
        f *= cfg.q / f.sum()
    return (f, res)

def b1b2_grid_dataframe(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alphas: Sequence[float], lams: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    x0_b1: np.ndarray | None = None
    x0_b2: np.ndarray | None = None
    for alpha in alphas:
        for lam in lams:
            if lam > alpha:
                continue
            eta = eta_from_lambda(alpha, lam)
            f_b1, res_b1 = solve_b1b2_convex_sp(instance, path_set, cfg, alpha, lam, True, x0_b1)
            f_b2, res_b2 = solve_b1b2_convex_sp(instance, path_set, cfg, alpha, lam, False, x0_b2)
            x0_b1 = res_b1.x
            x0_b2 = res_b2.x
            for form, f, res in (('B1', f_b1, res_b1), ('B2', f_b2, res_b2)):
                cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=alpha, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
                costs = model_b_raw_costs(instance, path_set, f, cfg_case)[0]
                _, probs, pi = truncated_flows_from_costs(costs, cfg_case)
                row: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Formulation': form, 'alpha': alpha, 'lambda': lam, 'eta': eta, 'Objective': float(res.fun), 'SolverIterations': int(getattr(res, 'nit', -1)), 'pi_od': pi, 'L1FlowDiff_B1_B2': float(np.sum(np.abs(f_b1 - f_b2)))}
                for k, path_name in enumerate(path_set.path_names):
                    row[f'f{k + 1}'] = f[k]
                    row[f'P{k + 1}'] = f[k] / cfg.q
                    row[f'g{k + 1}'] = costs[k]
                    row[f'Path{k + 1}'] = path_name
                rows.append(row)
    return pd.DataFrame(rows)

def solve_model_b_convex_sp_direct_eta(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alpha: float, eta: float, x0: np.ndarray | None=None, overlap_model: str='none', beta_ps: float=0.0) -> tuple[np.ndarray, opt.OptimizeResult]:
    n = path_set.n_paths
    s_count = len(instance.scenario_names)
    cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=alpha, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
    deterministic_penalty = overlap_penalty(instance, path_set, cfg_case, beta_ps=beta_ps, overlap_model=overlap_model)
    if x0 is None:
        f0 = np.full(n, cfg.q / n, dtype=float)
        vals0 = potentials_by_scenario(instance, path_set, f0, cfg_case)
        gamma0 = float(np.dot(instance.probabilities, vals0))
        u0 = np.maximum(vals0 - gamma0, 0.0)
        x0 = np.concatenate([f0, [gamma0], u0])
    bounds = opt.Bounds([0.0] * n + [-np.inf] + [0.0] * s_count, [np.inf] * n + [np.inf] + [np.inf] * s_count)
    cons: list[dict[str, object]] = [{'type': 'eq', 'fun': lambda x: float(np.sum(x[:n]) - cfg.q)}]

    def excess(x: np.ndarray, s_idx: int) -> float:
        f = x[:n]
        gamma = x[n]
        vals = potentials_by_scenario(instance, path_set, f, cfg_case)
        return float(x[n + 1 + s_idx] - (vals[s_idx] - gamma))
    for s_idx in range(s_count):
        cons.append({'type': 'ineq', 'fun': lambda x, s_idx=s_idx: excess(x, s_idx)})

    def objective(x: np.ndarray) -> float:
        f = np.asarray(x[:n], dtype=float)
        gamma = float(x[n])
        u = np.asarray(x[n + 1:], dtype=float)
        vals = potentials_by_scenario(instance, path_set, f, cfg_case)
        obj = (1.0 - eta) * float(np.dot(instance.probabilities, vals))
        obj += eta * gamma + eta / (1.0 - alpha) * float(np.dot(instance.probabilities, u))
        obj += entropy(f, cfg_case) / cfg_case.theta
        obj += float(np.dot(deterministic_penalty, f))
        return obj
    res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-11, 'maxiter': 4000, 'disp': False})
    if not res.success:
        try:
            rn = solve_fixed_point(instance, path_set, 'RN start', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg_case), cfg_case, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2, tol=1e-08, max_iter=20000).flows
            vals0 = potentials_by_scenario(instance, path_set, rn, cfg_case)
            gamma0 = float(np.dot(instance.probabilities, vals0))
            u0 = np.maximum(vals0 - gamma0, 0.0)
            fallback = np.concatenate([rn, [gamma0], u0])
        except Exception:
            fallback = x0
        res2 = opt.minimize(objective, fallback, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-11, 'maxiter': 6000, 'disp': False})
        if res2.success or res2.fun < res.fun:
            res = res2
    if not res.success:
        raise RuntimeError(f'Direct-eta Model B SP failed for alpha={alpha}, eta={eta}: {res.message}')
    f = np.maximum(np.asarray(res.x[:n], dtype=float), 0.0)
    if abs(f.sum() - cfg.q) > 1e-07:
        f *= cfg.q / f.sum()
    return (f, res)

def solve_model_b_for_params(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    start = time.perf_counter()
    f, res = solve_model_b_convex_sp_direct_eta(instance, path_set, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
    raw = model_b_raw_costs(instance, path_set, f, cfg)[0]
    perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    _, _, pi = truncated_flows_from_costs(perceived, cfg)
    residual = float('nan')
    elapsed = time.perf_counter() - start
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model B common-state TSUE', overlap_model=overlap_model, beta_ps=beta_ps, flows=f, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=f / cfg.q, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, f), iterations=int(getattr(res, 'nit', -1)), residual=residual, active_paths=tuple((path_set.path_names[i] for i, val in enumerate(f) if val > 1e-06)), cpu_seconds=elapsed)

def lambda_mapped_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alphas: Sequence[float], lams: Sequence[float]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        for lam in lams:
            if lam > alpha:
                continue
            eta = eta_from_lambda(alpha, lam)
            res = solve_model_b_for_params(instance, path_set, cfg, alpha, eta)
            row: dict[str, object] = {'Parameterization': 'lambda-mapped', 'alpha': alpha, 'lambda': lam, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha), 'pi_od': res.pi_od, 'ActiveCount': len(res.active_paths), 'Iterations': res.iterations}
            for k in range(path_set.n_paths):
                row[f'f{k + 1}'] = res.flows[k]
                row[f'P{k + 1}'] = res.probabilities[k]
                row[f'g{k + 1}'] = res.raw_costs[k]
            rows.append(row)
    return pd.DataFrame(rows)

def direct_eta_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alphas: Sequence[float], etas: Sequence[float]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        for eta in etas:
            res = solve_model_b_for_params(instance, path_set, cfg, alpha, eta)
            row: dict[str, object] = {'Parameterization': 'direct-eta', 'alpha': alpha, 'lambda': np.nan, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha), 'pi_od': res.pi_od, 'ActiveCount': len(res.active_paths), 'Iterations': res.iterations}
            for k in range(path_set.n_paths):
                row[f'f{k + 1}'] = res.flows[k]
                row[f'P{k + 1}'] = res.probabilities[k]
                row[f'g{k + 1}'] = res.raw_costs[k]
            rows.append(row)
    return pd.DataFrame(rows)

def fixed_lambda_vs_fixed_eta(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alphas: Sequence[float], fixed_lambda: float=0.4, alpha_ref: float=0.9) -> pd.DataFrame:
    eta_ref = eta_from_lambda(alpha_ref, fixed_lambda)
    rows = []
    for alpha in alphas:
        cases = [('fixed-lambda', fixed_lambda, eta_from_lambda(alpha, fixed_lambda) if fixed_lambda <= alpha else np.nan), ('fixed-eta', np.nan, eta_ref)]
        for label, lam, eta in cases:
            if np.isnan(eta):
                continue
            res = solve_model_b_for_params(instance, path_set, cfg, alpha, float(eta))
            row: dict[str, object] = {'Parameterization': label, 'alpha': alpha, 'lambda': lam, 'eta': float(eta), 'tail_hinge_coeff': float(eta) / (1.0 - alpha), 'pi_od': res.pi_od, 'ActiveCount': len(res.active_paths)}
            for k in range(path_set.n_paths):
                row[f'f{k + 1}'] = res.flows[k]
                row[f'P{k + 1}'] = res.probabilities[k]
            rows.append(row)
    return pd.DataFrame(rows)

def lambda_direct_identity_check(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, lambda_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in lambda_df.iterrows():
        alpha = float(row['alpha'])
        lam = float(row['lambda'])
        eta = float(row['eta'])
        res_eta = solve_model_b_for_params(instance, path_set, cfg, alpha, eta)
        f_lambda = np.array([row[f'f{k + 1}'] for k in range(path_set.n_paths)], dtype=float)
        rows.append({'alpha': alpha, 'lambda': lam, 'eta': eta, 'L1FlowDiff_lambda_vs_direct_eta': float(np.sum(np.abs(f_lambda - res_eta.flows)))})
    return pd.DataFrame(rows)

def run_model_a_warm_start_experiment(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig) -> pd.DataFrame:
    model_b_res = solve_core_models(instance, path_set, cfg)['Model B common-state TSUE']
    rows = []
    starts = [('Uniform', None), ('Model B warm start', model_b_res.flows)]
    for label, initial in starts:
        res = solve_fixed_point(instance, path_set, 'Model A mean-CVaR TSUE', lambda f: model_a_raw_costs(instance, path_set, f, cfg)[0], cfg, truncated=True, initial_flows=initial, relaxation=0.2, tol=1e-09, max_iter=80000)
        row: dict[str, object] = {'Initialization': label, 'Iterations': res.iterations, 'CPUSeconds': res.cpu_seconds, 'FinalResidual': res.residual, 'pi_od': res.pi_od, 'ActiveCount': len(res.active_paths), 'Gap_A': vi_gap_model_a(instance, path_set, res.flows, cfg)}
        for k in range(path_set.n_paths):
            row[f'f{k + 1}'] = res.flows[k]
            row[f'P{k + 1}'] = res.probabilities[k]
        rows.append(row)
    return pd.DataFrame(rows)

def plot_model_a_warm_start(warm_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(warm_df))
    bars = ax.bar(x, warm_df['Iterations'].values)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', (bar.get_x() + bar.get_width() / 2.0, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(warm_df['Initialization'].values)
    ax.set_ylabel('Model A fixed-point iterations')
    ax.set_title('Model B solution as a warm start for Model A')
    ax.grid(True, axis='y', alpha=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def aggregate_group_shares(path_set: PathSet, flows: np.ndarray) -> dict[str, float]:
    shares = flows / Q_DEMAND
    if path_set.name.startswith('Base'):
        return {'P1-corridor': float(shares[0]), 'P2': float(shares[1]), 'P3': float(shares[2])}
    if path_set.name.startswith('Clone'):
        return {'P1-corridor': float(shares[0] + shares[1]), 'P2': float(shares[2]), 'P3': float(shares[3])}
    raise ValueError(f'Unknown path_set={path_set.name}')

def run_iia_experiment(instance: BraessInstance, cfg: SolverConfig, beta_values: Sequence[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = base_path_set()
    clone = cloned_p1_path_set()
    rows = []
    detail_rows = []
    models = [('Ordinary logit SUE', False, 'RN'), ('Risk-neutral truncated TSUE', True, 'RN'), ('Model B common-state TSUE', True, 'ModelB')]
    for beta_ps in beta_values:
        overlap_model = 'none' if beta_ps == 0.0 else 'path-size'
        for model_name, truncated, cost_tag in models:

            def raw_cost_fun_for(pset: PathSet, tag: str) -> Callable[[np.ndarray], np.ndarray]:
                if tag == 'RN':
                    return lambda f: risk_neutral_raw_costs(instance, pset, f, cfg)
                if tag == 'ModelB':
                    return lambda f: model_b_raw_costs(instance, pset, f, cfg)[0]
                raise ValueError(tag)
            if cost_tag == 'ModelB':
                res_base = solve_model_b_for_params_labeled(instance, base, cfg, cfg.alpha, cfg.eta, overlap_model=overlap_model, beta_ps=beta_ps)
                res_clone = solve_model_b_for_params_labeled(instance, clone, cfg, cfg.alpha, cfg.eta, overlap_model=overlap_model, beta_ps=beta_ps)
            else:
                res_base = solve_fixed_point(instance, base, model_name, raw_cost_fun_for(base, cost_tag), cfg, truncated=truncated, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2)
                res_clone = solve_fixed_point(instance, clone, model_name, raw_cost_fun_for(clone, cost_tag), cfg, truncated=truncated, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2)
            g_base = aggregate_group_shares(base, res_base.flows)
            g_clone = aggregate_group_shares(clone, res_clone.flows)
            tv = 0.5 * sum((abs(g_clone[g] - g_base[g]) for g in g_base))
            clone_split = res_clone.flows[0] / max(res_clone.flows[0] + res_clone.flows[1], 1e-12)
            rows.append({'Model': model_name, 'RiskCostTag': cost_tag, 'beta_PS': beta_ps, 'OverlapModel': overlap_model, 'Base_P1_corridor': g_base['P1-corridor'], 'Base_P2': g_base['P2'], 'Base_P3': g_base['P3'], 'Clone_P1_corridor': g_clone['P1-corridor'], 'Clone_P2': g_clone['P2'], 'Clone_P3': g_clone['P3'], 'CloneDistortion_TV': tv, 'CloneSplit_P1_share_within_pair': clone_split, 'Base_ActiveCount': len(res_base.active_paths), 'Clone_ActiveCount': len(res_clone.active_paths), 'Base_pi': res_base.pi_od, 'Clone_pi': res_clone.pi_od})
            for res in (res_base, res_clone):
                detail = {'Model': res.model, 'PathSet': res.path_set, 'beta_PS': beta_ps, 'OverlapModel': overlap_model, 'pi_od': res.pi_od, 'ActiveCount': len(res.active_paths), 'Iterations': res.iterations}
                for k, name in enumerate((base if res.path_set.startswith('Base') else clone).path_names):
                    detail[f'Path{k + 1}'] = name
                    detail[f'f{k + 1}'] = res.flows[k]
                    detail[f'P{k + 1}'] = res.probabilities[k]
                    detail[f'RawCost{k + 1}'] = res.raw_costs[k]
                    detail[f'Penalty{k + 1}'] = res.overlap_penalty[k]
                    detail[f'PerceivedCost{k + 1}'] = res.perceived_costs[k]
                detail_rows.append(detail)
    return (pd.DataFrame(rows), pd.DataFrame(detail_rows))

def path_size_table(instance: BraessInstance, path_sets: Sequence[PathSet], beta_values: Sequence[float], cfg: SolverConfig) -> pd.DataFrame:
    rows = []
    for pset in path_sets:
        ps = path_size_factors(instance, pset)
        lengths = pset.incidence @ nominal_link_lengths(instance)
        for k, name in enumerate(pset.path_names):
            row: dict[str, object] = {'PathSet': pset.name, 'Path': name, 'NominalLength': lengths[k], 'PathSizeFactor': ps[k]}
            for beta in beta_values:
                penalty = overlap_penalty(instance, pset, cfg, beta_ps=beta, overlap_model='none' if beta == 0 else 'path-size')[k]
                row[f'Penalty_beta_{beta:g}'] = penalty
            rows.append(row)
    return pd.DataFrame(rows)

def results_to_dataframe(results: Iterable[EquilibriumResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row: dict[str, object] = {'Instance': r.instance, 'PathSet': r.path_set, 'Model': r.model, 'OverlapModel': r.overlap_model, 'beta_PS': r.beta_ps, 'pi_od': np.nan if r.pi_od is None else r.pi_od, 'ActiveCount': len(r.active_paths), 'Iterations': r.iterations, 'Residual': r.residual, 'CPUSeconds': r.cpu_seconds}
        for k, val in enumerate(r.flows, start=1):
            row[f'f{k}'] = val
            row[f'P{k}'] = r.probabilities[k - 1]
            row[f'raw_cost{k}'] = r.raw_costs[k - 1]
            row[f'penalty{k}'] = r.overlap_penalty[k - 1]
            row[f'perceived_cost{k}'] = r.perceived_costs[k - 1]
        for j, link in enumerate(LINKS):
            row[f'x_{link}'] = r.link_flows[j]
        rows.append(row)
    return pd.DataFrame(rows)

def scenario_params_to_dataframe(instance: BraessInstance) -> pd.DataFrame:
    rows = []
    for s_idx, scenario in enumerate(instance.scenario_names):
        for link in LINKS:
            prm = instance.link_params[scenario][link]
            rows.append({'Instance': instance.name, 'Scenario': scenario, 'Probability': instance.probabilities[s_idx], 'Link': LINK_LABELS[link], 't0': prm.t0, 'capacity': prm.capacity, 'Delta': prm.delay})
    return pd.DataFrame(rows)

def scenario_time_functions_to_dataframe(instance: BraessInstance) -> pd.DataFrame:
    rows = []
    for s_idx, scenario in enumerate(instance.scenario_names):
        for link in LINKS:
            prm = instance.link_params[scenario][link]
            rows.append({'Instance': instance.name, 'Scenario': scenario, 'Probability': instance.probabilities[s_idx], 'LinkCode': link, 'Link': LINK_LABELS[link], 't0': prm.t0, 'capacity': prm.capacity, 'Delta': prm.delay, 'Formula': f't_{link}^s(x_{link}) = {prm.t0:.2f}[1 + 0.15 (x_{link}/{prm.capacity:.2f})^4] + {prm.delay:.2f}'})
    return pd.DataFrame(rows)

def plot_network_diagram(output_path: Path) -> None:
    coords = {'A': (0.0, 0.0), 'B': (1.0, 0.75), 'C': (1.0, -0.75), 'D': (2.0, 0.0)}
    arrows = {'AB': ('A', 'B'), 'AC': ('A', 'C'), 'BC': ('B', 'C'), 'BD': ('B', 'D'), 'CD': ('C', 'D')}
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for node, (x, y) in coords.items():
        ax.scatter([x], [y], s=420)
        ax.text(x, y, node, ha='center', va='center', fontsize=12)
    for link, (u, v) in arrows.items():
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'shrinkA': 16, 'shrinkB': 16})
        ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.08, LINK_LABELS[link], ha='center')
    ax.set_title('Braess network and retained A→D paths')
    ax.axis('off')
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-1.1, 1.1)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_truncation_comparison(results: Sequence[EquilibriumResult], path_set: PathSet, output_path: Path, cfg: SolverConfig) -> None:
    labels = [r.model.replace(' truncated ', '\ntruncated ').replace(' mean-CVaR ', '\nmean-CVaR ') for r in results]
    flows = np.vstack([r.flows for r in results])
    costs = np.vstack([r.perceived_costs for r in results])
    pi_vals = [r.pi_od for r in results]
    x = np.arange(len(results))
    width = 0.22
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, sharex=True, figsize=(12.2, 6.4), gridspec_kw={'height_ratios': [1.4, 3.0], 'hspace': 0.06})
    for k in range(path_set.n_paths):
        bars = ax_bottom.bar(x + (k - (path_set.n_paths - 1) / 2) * width, flows[:, k], width, label=path_set.path_names[k])
        for bar in bars:
            value = bar.get_height()
            ax_bottom.annotate(f'{value:.1f}', (bar.get_x() + bar.get_width() / 2.0, value), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax_bottom.set_ylabel('Path flow')
    ax_bottom.set_ylim(0.0, cfg.q * 1.05)
    ax_bottom.grid(True, axis='y', alpha=0.25)
    ax_bottom.legend(loc='upper right', framealpha=0.95, fontsize=8)
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position('right')
    ax_top.set_ylabel('Perceived path cost')
    ax_top.grid(True, axis='y', alpha=0.25)
    for j in range(len(results)):
        xj = np.array([x[j] + (k - (path_set.n_paths - 1) / 2) * width for k in range(path_set.n_paths)])
        ax_top.plot(xj, costs[j, :], marker='o', linewidth=1.1)
        for k in range(path_set.n_paths):
            ax_top.annotate(f'{costs[j, k]:.1f}', (xj[k], costs[j, k]), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
        if pi_vals[j] is not None:
            pi = float(pi_vals[j])
            ax_top.hlines(pi, x[j] - 1.35 * width, x[j] + 1.35 * width, linestyles='--', linewidth=1.0)
            ax_top.annotate(f'pi={pi:.1f}', (x[j] + 1.35 * width, pi), xytext=(3, 4), textcoords='offset points', ha='left', va='bottom', fontsize=8)
    top_values = list(costs.ravel()) + [p for p in pi_vals if p is not None]
    cmin, cmax = (float(np.min(top_values)), float(np.max(top_values)))
    pad = 0.18 * (cmax - cmin + 1e-09)
    ax_top.set_ylim(cmin - pad, cmax + pad)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, fontsize=8)
    fig.suptitle(f'Braess truncation comparison, {path_set.name}\nalpha={cfg.alpha:g}, eta={cfg.eta:g}, theta={cfg.theta:g}, r0={cfg.r0:g}; dashed segment is endogenous pi', y=0.99)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.18, top=0.82)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_tail_tilt_laws(instance: BraessInstance, path_set: PathSet, f_eval: np.ndarray, cfg: SolverConfig, output_path: Path, legend_mode: str='outside_right') -> pd.DataFrame:
    _, _, tilts_a = model_a_raw_costs(instance, path_set, f_eval, cfg)
    _, _, tilt_b = model_b_raw_costs(instance, path_set, f_eval, cfg)
    data: dict[str, np.ndarray] = {'Nominal p': instance.probabilities}
    for k in range(path_set.n_paths):
        data[f'Model A P{k + 1}'] = tilts_a[k]
    data['Model B common'] = tilt_b
    df = pd.DataFrame(data, index=instance.scenario_names)
    x = np.arange(len(instance.scenario_names))
    width = min(0.14, 0.78 / len(df.columns))
    if legend_mode == 'outside_right':
        fig, ax = plt.subplots(figsize=(12.8, 4.8))
    elif legend_mode == 'outside_top':
        fig, ax = plt.subplots(figsize=(11.8, 5.3))
    else:
        fig, ax = plt.subplots(figsize=(11.2, 4.8))
    offset_center = (len(df.columns) - 1) / 2
    for j, column in enumerate(df.columns):
        bars = ax.bar(x + (j - offset_center) * width, df[column].values, width, label=column)
        for bar in bars:
            value = bar.get_height()
            if value > 0.035:
                ax.annotate(f'{value:.2f}', (bar.get_x() + bar.get_width() / 2.0, value), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(instance.scenario_names, rotation=12, ha='right')
    ax.set_ylabel('Scenario probability used in perceived costs')
    ax.set_ylim(0.0, max(0.65, float(df.values.max()) * 1.18))
    ax.grid(True, axis='y', alpha=0.25)
    ax.set_title('Nominal and tail-tilted scenario laws on route-local Braess\nModel A path-specific laws vs Model B common law')
    if legend_mode == 'outside_right':
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.95, fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.8, bottom=0.2, top=0.84)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    elif legend_mode == 'outside_top':
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.16), ncol=min(3, len(df.columns)), framealpha=0.95, fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.7)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        ax.legend(loc='upper right', framealpha=0.95, fontsize=8)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return df

def plot_b1b2_grid(grid_df: pd.DataFrame, formulation: str, path_set: PathSet, output_path: Path) -> None:
    sub = grid_df[grid_df['Formulation'] == formulation].copy().reset_index(drop=True)
    labels = [f'alpha={a:.2f}' + '\n' + f'lambda={l:.1f}' for a, l in zip(sub['alpha'], sub['lambda'])]
    probs = sub[[f'P{k + 1}' for k in range(path_set.n_paths)]].to_numpy(dtype=float)
    costs = sub[[f'g{k + 1}' for k in range(path_set.n_paths)]].to_numpy(dtype=float)
    pi_vals = sub['pi_od'].to_numpy(dtype=float)
    x = np.arange(len(labels))
    width = min(0.22, 0.78 / path_set.n_paths)
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, sharex=True, figsize=(13.3, 6.4), gridspec_kw={'height_ratios': [1.4, 3.0], 'hspace': 0.05})
    offset_center = (path_set.n_paths - 1) / 2
    for k in range(path_set.n_paths):
        bars = ax_bottom.bar(x + (k - offset_center) * width, probs[:, k], width, label=f'P{k + 1}')
        for bar in bars:
            h = bar.get_height()
            ax_bottom.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2.0, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax_bottom.set_ylabel('Choice probability')
    ax_bottom.set_ylim(0.0, 1.0)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels)
    ax_bottom.grid(True, axis='y', alpha=0.25)
    ax_bottom.legend(loc='upper right', framealpha=0.95, fontsize=8)
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position('right')
    ax_top.set_ylabel('Induced marginal cost g_k (min)')
    ax_top.grid(True, axis='y', alpha=0.2)
    for j in range(len(labels)):
        xj = np.array([x[j] + (k - offset_center) * width for k in range(path_set.n_paths)])
        ax_top.plot(xj, costs[j, :], marker='o', linewidth=1.0)
        for k in range(path_set.n_paths):
            ax_top.annotate(f'{costs[j, k]:.1f}', (xj[k], costs[j, k]), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
        pi = pi_vals[j]
        ax_top.hlines(pi, x[j] - 1.35 * width, x[j] + 1.35 * width, linestyles='--', linewidth=1.0)
    values = np.concatenate([costs.ravel(), pi_vals.ravel()])
    cmin, cmax = (float(np.min(values)), float(np.max(values)))
    ax_top.set_ylim(cmin - 0.12 * (cmax - cmin + 1e-09), cmax + 0.12 * (cmax - cmin + 1e-09))
    fig.suptitle(f'Braess Approach {formulation}: entropy-placement check', y=0.99)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.16, top=0.82, hspace=0.05)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_eta_lambda_mapping(output_path: Path) -> pd.DataFrame:
    alphas = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
    rows = []
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for alpha in alphas:
        lams = np.linspace(0.0, alpha, 200)
        etas = np.array([eta_from_lambda(float(alpha), float(lam)) for lam in lams])
        ax.plot(lams, etas, label=f'alpha={alpha:.2f}')
        for lam in [0.2, 0.4, 0.6, 0.8]:
            if lam <= alpha:
                eta = eta_from_lambda(float(alpha), float(lam))
                rows.append({'alpha': float(alpha), 'lambda': lam, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha), 'eta_upper_bound': alpha / (1.0 + alpha), 'hinge_upper_bound': alpha / (1.0 - alpha ** 2)})
    ax.set_xlabel('reporting index lambda')
    ax.set_ylabel('effective CVaR weight eta_alpha,lambda')
    ax.set_title('Optional bounded map from (alpha,lambda) to effective eta')
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.95)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows)

def plot_fixed_lambda_vs_fixed_eta(df: pd.DataFrame, path_set: PathSet, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for param in df['Parameterization'].unique():
        sub = df[df['Parameterization'] == param].sort_values('alpha')
        ax.plot(sub['alpha'], sub['P1'], marker='o', label=f'{param}: P1')
        ax.plot(sub['alpha'], sub['P2'], marker='s', label=f'{param}: P2')
        if 'P3' in sub.columns:
            ax.plot(sub['alpha'], sub['P3'], marker='^', label=f'{param}: P3')
    ax.set_xlabel('alpha')
    ax.set_ylabel('Path share')
    ax.set_title('Fixed lambda changes both tail location and effective eta; fixed eta isolates tail location')
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.95, fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_iia_aggregate_shares(iia_df: pd.DataFrame, output_path: Path, model: str='Model B common-state TSUE') -> None:
    sub = iia_df[iia_df['Model'] == model].sort_values('beta_PS')
    labels = [f'beta_PS={b:g}' for b in sub['beta_PS']]
    x = np.arange(len(sub))
    width = 0.12
    fields = [('Base_P1_corridor', 'Base P1-corridor'), ('Clone_P1_corridor', 'Clone P1-corridor'), ('Base_P2', 'Base P2'), ('Clone_P2', 'Clone P2'), ('Base_P3', 'Base P3'), ('Clone_P3', 'Clone P3')]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    offset_center = (len(fields) - 1) / 2
    for j, (field, label) in enumerate(fields):
        bars = ax.bar(x + (j - offset_center) * width, sub[field].values, width, label=label)
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2.0, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Aggregate group share')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(framealpha=0.95, fontsize=8, ncol=2)
    ax.set_title('IIA duplicate-path diagnostic: base paths vs cloned P1 candidate set')
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_iia_distortion(iia_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for model in iia_df['Model'].unique():
        sub = iia_df[iia_df['Model'] == model].sort_values('beta_PS')
        ax.plot(sub['beta_PS'], sub['CloneDistortion_TV'], marker='o', label=model)
    ax.set_xlabel('path-size correction strength beta_PS')
    ax.set_ylabel('Duplicate-path distortion\n(total variation of aggregate shares)')
    ax.set_title('Path-overlap correction reduces sensitivity to a cloned alternative')
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.95, fontsize=8)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_path_size_factors(ps_df: pd.DataFrame, output_path: Path) -> None:
    sub = ps_df.copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = [f"{r.PathSet}\n{r.Path.split(':')[0]}" for _, r in sub.iterrows()]
    x = np.arange(len(labels))
    bars = ax.bar(x, sub['PathSizeFactor'].values)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2.0, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Path-size factor')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis='y', alpha=0.25)
    ax.set_title('Path-size factors used for deterministic overlap correction')
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def selected_tail_scenario_name(values: Sequence[float], probs: Sequence[float], alpha: float, scenario_names: Sequence[str]) -> str:
    chi = finite_cvar_selector(values, probs, alpha)
    parts = []
    for name, val in zip(scenario_names, chi):
        if val > 1e-08:
            parts.append(f'{name}({val:.2f})')
    return '; '.join(parts) if parts else 'None'

def classify_risk_case(eta: float) -> str:
    if abs(eta) <= 1e-12:
        return 'Expectation-only'
    if abs(eta - 1.0) <= 1e-12:
        return 'CVaR-only'
    return 'Mean-CVaR'

def system_risk_metrics(instance: BraessInstance, path_set: PathSet, flows: Sequence[float], alpha: float, eta: float, cfg: SolverConfig) -> dict[str, float | str]:
    cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=alpha, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
    vals = potentials_by_scenario(instance, path_set, flows, cfg_case)
    mean_val = float(np.dot(instance.probabilities, vals))
    cvar_val = cvar_value(vals, instance.probabilities, alpha)
    ce_val = (1.0 - eta) * mean_val + eta * cvar_val
    ent_val = entropy(flows, cfg_case) / cfg_case.theta
    return {'MeanPotential': mean_val, 'CVaRPotential': cvar_val, 'CEPotential': ce_val, 'EntropyOverTheta': ent_val, 'ModelBObjectiveAtFlow': ce_val + ent_val, 'SystemTailSelector': selected_tail_scenario_name(vals, instance.probabilities, alpha, instance.scenario_names)}

def solve_model_a_for_params(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, initial_flows: np.ndarray | None=None) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    return solve_fixed_point(instance, path_set, 'Model A mean-CVaR TSUE' if eta not in (0.0, 1.0) else 'Model A expectation TSUE' if eta == 0.0 else 'Model A CVaR-only TSUE', lambda f: model_a_raw_costs(instance, path_set, f, cfg)[0], cfg, truncated=True, relaxation=0.2, tol=1e-09, max_iter=80000, initial_flows=initial_flows)

def solve_model_b_for_params_labeled(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> EquilibriumResult:
    res = solve_model_b_for_params(instance, path_set, cfg_base, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
    label = 'Model B common-state TSUE'
    if eta == 0.0:
        label = 'Model B expectation TSUE'
    elif eta == 1.0:
        label = 'Model B CVaR-only TSUE'
    return EquilibriumResult(instance=res.instance, path_set=res.path_set, model=label, overlap_model=res.overlap_model, beta_ps=res.beta_ps, flows=res.flows, perceived_costs=res.perceived_costs, raw_costs=res.raw_costs, overlap_penalty=res.overlap_penalty, probabilities=res.probabilities, pi_od=res.pi_od, link_flows=res.link_flows, iterations=res.iterations, residual=res.residual, active_paths=res.active_paths, cpu_seconds=res.cpu_seconds)

def direct_eta_risk_sensitivity_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, alphas: Sequence[float], mean_cvar_etas: Sequence[float], include_model_a: bool=True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eta_cases = [0.0] + [float(e) for e in mean_cvar_etas] + [1.0]
    eta_cases = sorted(set((round(e, 10) for e in eta_cases)))
    last_model_a: dict[tuple[float, float], np.ndarray] = {}
    for alpha in alphas:
        for eta in eta_cases:
            cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=alpha, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
            res_b = solve_model_b_for_params_labeled(instance, path_set, cfg, alpha, eta)
            metrics_b = system_risk_metrics(instance, path_set, res_b.flows, alpha, eta, cfg_case)
            row_b: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Model': 'Model B', 'RiskCase': classify_risk_case(eta), 'alpha': alpha, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha) if eta > 0 else 0.0, 'pi_od': res_b.pi_od, 'ActiveCount': len(res_b.active_paths), 'Iterations': res_b.iterations, 'Residual': res_b.residual, 'CPUSeconds': res_b.cpu_seconds}
            row_b.update(metrics_b)
            for k, name in enumerate(path_set.path_names):
                row_b[f'Path{k + 1}'] = name
                row_b[f'f{k + 1}'] = res_b.flows[k]
                row_b[f'P{k + 1}'] = res_b.probabilities[k]
                row_b[f'g{k + 1}'] = res_b.raw_costs[k]
            rows.append(row_b)
            if include_model_a:
                prev_eta_values = [e for e in eta_cases if e < eta]
                initial = None
                if prev_eta_values:
                    prev_key = (alpha, max(prev_eta_values))
                    initial = last_model_a.get(prev_key)
                res_a = solve_model_a_for_params(instance, path_set, cfg, alpha, eta, initial_flows=initial)
                last_model_a[alpha, eta] = res_a.flows
                metrics_a = system_risk_metrics(instance, path_set, res_a.flows, alpha, eta, cfg_case)
                row_a: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Model': 'Model A', 'RiskCase': classify_risk_case(eta), 'alpha': alpha, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha) if eta > 0 else 0.0, 'pi_od': res_a.pi_od, 'ActiveCount': len(res_a.active_paths), 'Iterations': res_a.iterations, 'Residual': res_a.residual, 'CPUSeconds': res_a.cpu_seconds, 'Gap_A': vi_gap_model_a_limit(instance, path_set, res_a.flows, cfg_case) if alpha_is_limit else vi_gap_model_a(instance, path_set, res_a.flows, cfg_case)}
                row_a.update(metrics_a)
                for k, name in enumerate(path_set.path_names):
                    row_a[f'Path{k + 1}'] = name
                    row_a[f'f{k + 1}'] = res_a.flows[k]
                    row_a[f'P{k + 1}'] = res_a.probabilities[k]
                    row_a[f'g{k + 1}'] = res_a.raw_costs[k]
                rows.append(row_a)
    return pd.DataFrame(rows)

def model_a_b_eta_diagnostics_from_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, risk_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, eta), sub in risk_df.groupby(['alpha', 'eta']):
        sub_a = sub[sub['Model'] == 'Model A']
        sub_b = sub[sub['Model'] == 'Model B']
        if sub_a.empty or sub_b.empty:
            continue
        f_a = np.array([float(sub_a.iloc[0][f'f{k + 1}']) for k in range(path_set.n_paths)])
        f_b = np.array([float(sub_b.iloc[0][f'f{k + 1}']) for k in range(path_set.n_paths)])
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=float(alpha), eta=float(eta), bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        if eta == 0.0:
            d_p = 0.0
        else:
            _, _, tilts_a_at_b = model_a_raw_costs(instance, path_set, f_b, cfg_case)
            _, _, tilt_b_at_b = model_b_raw_costs(instance, path_set, f_b, cfg_case)
            d_p = float(max((np.sum(np.abs(tilts_a_at_b[k] - tilt_b_at_b)) for k in range(path_set.n_paths))))
        d_f = float(np.sum(np.abs(f_a - f_b)) / cfg.q)
        gap_a = vi_gap_model_a(instance, path_set, f_b, cfg_case)
        rows.append({'Instance': instance.name, 'PathSet': path_set.name, 'alpha': float(alpha), 'eta': float(eta), 'RiskCase': classify_risk_case(float(eta)), 'D_p_AB': d_p, 'D_f_AB': d_f, 'Gap_A_at_fB': gap_a, 'Active_A': int(sub_a.iloc[0]['ActiveCount']), 'Active_B': int(sub_b.iloc[0]['ActiveCount']), 'Active_A_minus_Active_B': int(sub_a.iloc[0]['ActiveCount'] - sub_b.iloc[0]['ActiveCount'])})
    return pd.DataFrame(rows).sort_values(['alpha', 'eta']).reset_index(drop=True)

def compact_modelb_risk_table(risk_df: pd.DataFrame) -> pd.DataFrame:
    sub = risk_df[risk_df['Model'] == 'Model B'].copy()
    keep = ['RiskCase', 'alpha', 'eta', 'tail_hinge_coeff', 'P1', 'P2', 'P3', 'pi_od', 'ActiveCount', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    return sub[keep].sort_values(['alpha', 'eta']).reset_index(drop=True)

def run_all(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SolverConfig()
    base = base_path_set()
    clone = cloned_p1_path_set()
    for pset in (base, clone):
        pset.validate()
    common = build_common_tail_instance()
    route_local = build_route_local_instance()
    for inst in (common, route_local):
        validate_instance(inst)
    print('[1/8] plotting network', flush=True)
    plot_network_diagram(output_dir / 'braess_network_diagram.png')
    print('[2/8] core experiments', flush=True)
    core_results: list[EquilibriumResult] = []
    per: dict[str, dict[str, EquilibriumResult]] = {}
    for inst in (common, route_local):
        solved = solve_core_models(inst, base, cfg)
        per[inst.name] = solved
        core_results.extend(solved.values())
    results_df = results_to_dataframe(core_results)
    results_df.to_csv(output_dir / 'braess_equilibrium_summary.csv', index=False)
    params_df = pd.concat([scenario_params_to_dataframe(common), scenario_params_to_dataframe(route_local)], ignore_index=True)
    params_df.to_csv(output_dir / 'braess_scenario_parameters.csv', index=False)
    formula_df = scenario_time_functions_to_dataframe(route_local)
    formula_df.to_csv(output_dir / 'braess_scenario_time_functions.csv', index=False)
    route_local_order = [per[route_local.name]['Ordinary logit SUE'], per[route_local.name]['Risk-neutral truncated TSUE'], per[route_local.name]['Model A mean-CVaR TSUE'], per[route_local.name]['Model B common-state TSUE']]
    plot_truncation_comparison(route_local_order, base, output_dir / 'braess_truncation_comparison.png', cfg)
    f_b = per[route_local.name]['Model B common-state TSUE'].flows
    tilt_df = plot_tail_tilt_laws(route_local, base, f_b, cfg, output_dir / 'braess_tail_tilt_laws.png')
    tilt_df.to_csv(output_dir / 'braess_tail_tilt_laws.csv')
    ab_df = pd.DataFrame([model_a_b_metrics(common, base, cfg), model_a_b_metrics(route_local, base, cfg)])
    ab_df.to_csv(output_dir / 'braess_modelA_modelB_metrics.csv', index=False)
    warm_df = run_model_a_warm_start_experiment(route_local, base, cfg)
    warm_df.to_csv(output_dir / 'braess_modelA_warm_start.csv', index=False)
    plot_model_a_warm_start(warm_df, output_dir / 'braess_modelA_warm_start.png')
    print('[3/8] A/B metrics, tail tilt, and warm start complete', flush=True)
    pd.DataFrame().to_csv(output_dir / 'braess_B1_B2_grid.csv', index=False)
    print('[4/8] parameterization mapping', flush=True)
    map_table = plot_eta_lambda_mapping(output_dir / 'braess_eta_lambda_mapping.png')
    map_table.to_csv(output_dir / 'braess_eta_lambda_map_table.csv', index=False)
    pd.DataFrame().to_csv(output_dir / 'braess_lambda_mapped_grid.csv', index=False)
    pd.DataFrame().to_csv(output_dir / 'braess_direct_eta_grid.csv', index=False)
    pd.DataFrame().to_csv(output_dir / 'braess_fixed_lambda_vs_fixed_eta.csv', index=False)
    pd.DataFrame().to_csv(output_dir / 'braess_lambda_direct_eta_identity_check.csv', index=False)
    print('[5/9] direct-eta risk sensitivity and CVaR-only grid', flush=True)
    direct_eta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    risk_df = direct_eta_risk_sensitivity_grid(route_local, base, cfg, alphas=[0.9, 0.95], mean_cvar_etas=direct_eta_values, include_model_a=False)
    risk_df.to_csv(output_dir / 'braess_direct_eta_risk_sensitivity_all_models.csv', index=False)
    modelb_compact_df = compact_modelb_risk_table(risk_df)
    modelb_compact_df.to_csv(output_dir / 'braess_modelB_direct_eta_risk_table.csv', index=False)
    ab_eta_df = pd.DataFrame()
    ab_eta_df.to_csv(output_dir / 'braess_modelA_modelB_eta_grid_diagnostics.csv', index=False)
    display_risk_df = modelb_compact_df.copy()
    display_risk_df = display_risk_df[display_risk_df['eta'].isin([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])]
    print('[6/9] IIA/path-overlap diagnostic', flush=True)
    beta_values = [0.0, 0.5, 1.0, 2.0]
    ps_df = path_size_table(route_local, [base, clone], beta_values, cfg)
    ps_df.to_csv(output_dir / 'braess_iia_path_size_factors.csv', index=False)
    plot_path_size_factors(ps_df, output_dir / 'braess_iia_path_size_factors.png')
    iia_df, iia_detail_df = run_iia_experiment(route_local, cfg, beta_values)
    iia_df.to_csv(output_dir / 'braess_iia_clone_diagnostic.csv', index=False)
    iia_detail_df.to_csv(output_dir / 'braess_iia_equilibrium_details.csv', index=False)
    plot_iia_aggregate_shares(iia_df, output_dir / 'braess_iia_aggregate_shares_modelB.png', model='Model B common-state TSUE')
    plot_iia_distortion(iia_df, output_dir / 'braess_iia_distortion_vs_beta.png')
    print('[7/9] writing environment and README', flush=True)
    env = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__, 'q': cfg.q, 'theta': cfg.theta, 'r0': cfg.r0, 'alpha_main': cfg.alpha, 'eta_main': cfg.eta, 'bpr_beta': cfg.bpr_beta, 'bpr_gamma': cfg.bpr_gamma}
    (output_dir / 'computing_environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    readme = f'# Integrated Braess TSUE experiment outputs\n\nGenerated by `braess_integrated_experiment.py`.\n\nMain parameters: q={cfg.q}, theta={cfg.theta}, r0={cfg.r0}, alpha={cfg.alpha}, eta={cfg.eta}. v5 uses a shortcut-active route-local Braess calibration.\n\nImportant files:\n\n- `braess_equilibrium_summary.csv`: ordinary logit, RN truncated TSUE, Model A, and Model B on common-tail and route-local Braess.\n- `braess_modelA_modelB_metrics.csv`: exactness/divergence diagnostics.\n- `braess_modelA_warm_start.csv`: Model A convergence from uniform initialization versus Model B warm start.\n- `braess_B1_B2_grid.csv`: entropy-placement B1/B2 consistency check using the alpha/lambda grid.\n- `braess_lambda_mapped_grid.csv`, `braess_direct_eta_grid.csv`, `braess_fixed_lambda_vs_fixed_eta.csv`: parameterization comparisons.\n- `braess_direct_eta_risk_sensitivity_all_models.csv`: expectation-only, direct-eta mean-CVaR, and CVaR-only grid for alpha in {(0.9, 0.95)}; includes Model A and Model B.\n- `braess_modelB_direct_eta_risk_table.csv`: compact Model B risk-sensitivity table for manuscript reporting.\n- `braess_modelA_modelB_eta_grid_diagnostics.csv`: A/B divergence metrics over the direct-eta grid.\n- `braess_iia_path_size_factors.csv`: path-size factors and overlap penalties.\n- `braess_iia_clone_diagnostic.csv`: base-vs-cloned-candidate-set IIA diagnostic.\n- `braess_iia_equilibrium_details.csv`: detailed path flows/costs/penalties for every IIA case.\n\nIIA setup:\n\n- Base set: P1=A-B-D, P2=A-C-D, P3=A-B-C-D.\n- Clone set: P1, P1′=A-B-D, P2, P3.\n- beta_PS=0 is the no-overlap-correction / IIA baseline.\n- beta_PS in {{0.5,1.0,2.0}} adds path-size cost penalty: omega_k=-(beta_PS/theta) log(PS_k).\n'
    (output_dir / 'README.md').write_text(readme, encoding='utf-8')
    print('[8/9] run_all complete', flush=True)
    return {'output_dir': str(output_dir), 'summary_csv': str(output_dir / 'braess_equilibrium_summary.csv'), 'iia_csv': str(output_dir / 'braess_iia_clone_diagnostic.csv')}

def zip_outputs(output_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))

def _tail_law_from_selector(probs: np.ndarray, selector: np.ndarray, alpha: float) -> np.ndarray:
    return probs * selector / (1.0 - alpha)

def _tail_set_string(scenario_names: Sequence[str], q_tail: np.ndarray, tol: float=1e-08) -> str:
    parts = []
    for idx, mass in enumerate(q_tail):
        if mass > tol:
            parts.append(f'{scenario_names[idx]}({mass:.2f})')
    return '; '.join(parts) if parts else 'None'

def tail_divergence_at_fB(instance: BraessInstance, path_set: PathSet, f_b: Sequence[float], cfg: SolverConfig) -> dict[str, object]:
    _, selectors_a_at_b, tilts_a_at_b = model_a_raw_costs(instance, path_set, f_b, cfg)
    _, selector_b_at_b, tilt_b_at_b = model_b_raw_costs(instance, path_set, f_b, cfg)
    probs = instance.probabilities
    tail_a = np.vstack([_tail_law_from_selector(probs, selectors_a_at_b[k], cfg.alpha) for k in range(path_set.n_paths)])
    tail_b = _tail_law_from_selector(probs, selector_b_at_b, cfg.alpha)
    d_tail_by_path = np.asarray([np.sum(np.abs(tail_a[k] - tail_b)) for k in range(path_set.n_paths)], dtype=float)
    d_p_by_path = np.asarray([np.sum(np.abs(tilts_a_at_b[k] - tilt_b_at_b)) for k in range(path_set.n_paths)], dtype=float)
    worst_idx = int(np.argmax(d_tail_by_path))
    out: dict[str, object] = {'D_tail_AB': float(np.max(d_tail_by_path)), 'Mean_D_tail_AB': float(np.mean(d_tail_by_path)), 'D_p_AB': float(np.max(d_p_by_path)), 'Mean_D_p_AB': float(np.mean(d_p_by_path)), 'TailTV_max_AB': float(0.5 * np.max(d_p_by_path)), 'WorstTailPath': f'P{worst_idx + 1}', 'TailSet_B_at_fB': _tail_set_string(instance.scenario_names, tail_b)}
    for k in range(path_set.n_paths):
        out[f'D_tail_P{k + 1}'] = float(d_tail_by_path[k])
        out[f'D_p_P{k + 1}'] = float(d_p_by_path[k])
        out[f'TailSet_A_P{k + 1}_at_fB'] = _tail_set_string(instance.scenario_names, tail_a[k])
    return out

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-08, 1.0 - 1e-08))
    return math.log(p / (1.0 - p))

def _z_to_simplex_flows(z: Sequence[float], q: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    s1 = float(_sigmoid(z[0]))
    s2 = float(_sigmoid(z[1]))
    f1 = q * s1
    f2 = (q - f1) * s2
    f3 = q - f1 - f2
    return np.array([f1, f2, f3], dtype=float)

def _simplex_flows_to_z(f: Sequence[float], q: float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    f = np.maximum(f, 1e-07)
    f = q * f / f.sum()
    s1 = f[0] / q
    rem = max(q - f[0], 1e-07)
    s2 = f[1] / rem
    return np.array([_logit(s1), _logit(s2)], dtype=float)

def solve_model_a_fast(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, initial_flows: Sequence[float] | None=None) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=float(alpha), eta=float(eta), bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    if initial_flows is None:
        try:
            initial_flows = solve_model_b_for_params_labeled(instance, path_set, cfg_base, alpha, eta).flows
        except Exception:
            initial_flows = np.full(path_set.n_paths, cfg.q / path_set.n_paths, dtype=float)
    z0 = _simplex_flows_to_z(initial_flows, cfg.q)

    def residual_z(z: np.ndarray) -> np.ndarray:
        f = _z_to_simplex_flows(z, cfg.q)
        costs = model_a_raw_costs(instance, path_set, f, cfg)[0]
        target, _, _ = truncated_flows_from_costs(costs, cfg)
        return (target - f) / cfg.q
    start = time.perf_counter()
    res = opt.least_squares(residual_z, z0, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=300)
    flows = _z_to_simplex_flows(res.x, cfg.q)
    raw = model_a_raw_costs(instance, path_set, flows, cfg)[0]
    target, probs, pi = truncated_flows_from_costs(raw, cfg)
    flows = target
    raw = model_a_raw_costs(instance, path_set, flows, cfg)[0]
    flows, probs, pi = truncated_flows_from_costs(raw, cfg)
    elapsed = time.perf_counter() - start
    active = tuple((path_set.path_names[i] for i, val in enumerate(flows) if val > 1e-06))
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model A mean-CVaR TSUE' if eta not in (0.0, 1.0) else 'Model A expectation TSUE' if eta == 0.0 else 'Model A CVaR-only TSUE', overlap_model='none', beta_ps=0.0, flows=flows, perceived_costs=raw, raw_costs=raw, overlap_penalty=np.zeros_like(raw), probabilities=probs, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, flows), iterations=int(res.nfev), residual=float(np.max(np.abs(residual_z(res.x))) * cfg.q), active_paths=active, cpu_seconds=elapsed)

def _one_hot_tail_law(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v, dtype=float)
    out[int(np.argmax(v))] = 1.0
    return out

def _tail_law_string_from_values(scenario_names: Sequence[str], values: Sequence[float]) -> str:
    q_tail = _one_hot_tail_law(values)
    return _tail_set_string(scenario_names, q_tail)

def model_a_limit_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    pt = path_times_by_scenario(instance, path_set, path_flows, cfg)
    mean = instance.probabilities @ pt
    maxv = np.max(pt, axis=0)
    costs = (1.0 - cfg.eta) * mean + cfg.eta * maxv
    tail_laws = np.vstack([_one_hot_tail_law(pt[:, k]) for k in range(path_set.n_paths)])
    return (costs, tail_laws)

def model_b_limit_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = potentials_by_scenario(instance, path_set, path_flows, cfg)
    tail_law = _one_hot_tail_law(vals)
    tilted = (1.0 - cfg.eta) * instance.probabilities + cfg.eta * tail_law
    costs = tilted @ path_times_by_scenario(instance, path_set, path_flows, cfg)
    return (np.asarray(costs), tail_law, tilted)

def system_risk_metrics_limit(instance: BraessInstance, path_set: PathSet, flows: Sequence[float], eta: float, cfg: SolverConfig) -> dict[str, float | str]:
    vals = potentials_by_scenario(instance, path_set, flows, cfg)
    mean_val = float(np.dot(instance.probabilities, vals))
    max_val = float(np.max(vals))
    ce_val = (1.0 - eta) * mean_val + eta * max_val
    ent_val = entropy(flows, cfg) / cfg.theta
    return {'MeanPotential': mean_val, 'CVaRPotential': max_val, 'CEPotential': ce_val, 'EntropyOverTheta': ent_val, 'ModelBObjectiveAtFlow': ce_val + ent_val, 'SystemTailSelector': _tail_law_string_from_values(instance.scenario_names, vals)}

def vi_gap_model_a_limit(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> float:
    f = np.asarray(path_flows, dtype=float)
    raw = model_a_limit_raw_costs(instance, path_set, f, cfg)[0]
    mapping = raw + 1.0 / cfg.theta * np.log1p(f / cfg.r0)
    gap = float(np.dot(mapping, f) - cfg.q * np.min(mapping))
    return gap / (1.0 + abs(float(np.dot(mapping, f))))

def solve_model_a_limit_fast(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, eta: float, initial_flows: Sequence[float] | None=None) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=1.0 - ALPHA_LIMIT_EPS, eta=float(eta), bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    if initial_flows is None:
        try:
            initial_flows = solve_model_b_limit_for_params(instance, path_set, cfg_base, eta).flows
        except Exception:
            initial_flows = np.full(path_set.n_paths, cfg.q / path_set.n_paths, dtype=float)
    z0 = _simplex_flows_to_z(initial_flows, cfg.q)

    def residual_z(z: np.ndarray) -> np.ndarray:
        f = _z_to_simplex_flows(z, cfg.q)
        costs = model_a_limit_raw_costs(instance, path_set, f, cfg)[0]
        target, _, _ = truncated_flows_from_costs(costs, cfg)
        return (target - f) / cfg.q
    start = time.perf_counter()
    res = opt.least_squares(residual_z, z0, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=600)
    flows = _z_to_simplex_flows(res.x, cfg.q)
    raw = model_a_limit_raw_costs(instance, path_set, flows, cfg)[0]
    flows, probs, pi = truncated_flows_from_costs(raw, cfg)
    raw = model_a_limit_raw_costs(instance, path_set, flows, cfg)[0]
    flows, probs, pi = truncated_flows_from_costs(raw, cfg)
    elapsed = time.perf_counter() - start
    active = tuple((path_set.path_names[i] for i, val in enumerate(flows) if val > 1e-06))
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model A alpha-limit TSUE' if eta not in (0.0, 1.0) else 'Model A expectation TSUE' if eta == 0.0 else 'Model A max-tail TSUE', overlap_model='none', beta_ps=0.0, flows=flows, perceived_costs=raw, raw_costs=raw, overlap_penalty=np.zeros_like(raw), probabilities=probs, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, flows), iterations=int(res.nfev), residual=float(np.max(np.abs(residual_z(res.x))) * cfg.q), active_paths=active, cpu_seconds=elapsed)

def solve_model_b_limit_for_params(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=1.0 - ALPHA_LIMIT_EPS, eta=float(eta), bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    n = path_set.n_paths
    deterministic_penalty = overlap_penalty(instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)

    def objective_f(f_raw: np.ndarray) -> float:
        f = np.asarray(f_raw, dtype=float)
        if np.any(f < -1e-08):
            return 1e+30
        f = np.maximum(f, 0.0)
        if f.sum() <= 0:
            return 1e+30
        f = cfg.q * f / f.sum()
        vals = potentials_by_scenario(instance, path_set, f, cfg)
        ce = (1.0 - eta) * float(np.dot(instance.probabilities, vals)) + eta * float(np.max(vals))
        return float(ce + entropy(f, cfg) / cfg.theta + np.dot(deterministic_penalty, f))
    constraints = ({'type': 'eq', 'fun': lambda f: float(np.sum(f) - cfg.q)},)
    bounds = opt.Bounds(np.zeros(n), np.full(n, cfg.q))
    starts = [np.full(n, cfg.q / n, dtype=float)]
    try:
        rn = solve_fixed_point(instance, path_set, 'RN alpha-limit start', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg), cfg, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2, tol=1e-08, max_iter=20000).flows
        starts.append(rn)
    except Exception:
        pass
    for k in range(n):
        e = np.zeros(n, dtype=float)
        e[k] = cfg.q
        starts.append(e)
    starts += [np.array([0.5, 0.5, 0.0]) * cfg.q, np.array([0.5, 0.0, 0.5]) * cfg.q, np.array([0.0, 0.5, 0.5]) * cfg.q]
    best: opt.OptimizeResult | None = None
    start_time = time.perf_counter()
    for x0 in starts:
        res = opt.minimize(objective_f, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-11, 'maxiter': 3000, 'disp': False})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('Model B alpha-limit solve failed.')
    f = np.maximum(np.asarray(best.x, dtype=float), 0.0)
    if f.sum() <= 0.0:
        f[:] = cfg.q / n
    else:
        f *= cfg.q / f.sum()
    raw = model_b_limit_raw_costs(instance, path_set, f, cfg)[0]
    perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    _, _, pi = truncated_flows_from_costs(perceived, cfg)
    elapsed = time.perf_counter() - start_time
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model B alpha-limit TSUE' if eta not in (0.0, 1.0) else 'Model B expectation TSUE' if eta == 0.0 else 'Model B max-tail TSUE', overlap_model=overlap_model, beta_ps=beta_ps, flows=f, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=f / cfg.q, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, f), iterations=int(getattr(best, 'nit', -1)), residual=float('nan'), active_paths=tuple((path_set.path_names[i] for i, val in enumerate(f) if val > 1e-06)), cpu_seconds=elapsed)

def tail_divergence_limit_at_fB(instance: BraessInstance, path_set: PathSet, f_b: Sequence[float], cfg: SolverConfig) -> dict[str, object]:
    pt = path_times_by_scenario(instance, path_set, f_b, cfg)
    vals = potentials_by_scenario(instance, path_set, f_b, cfg)
    tail_a = np.vstack([_one_hot_tail_law(pt[:, k]) for k in range(path_set.n_paths)])
    tail_b = _one_hot_tail_law(vals)
    tilts_a = (1.0 - cfg.eta) * instance.probabilities[None, :] + cfg.eta * tail_a
    tilt_b = (1.0 - cfg.eta) * instance.probabilities + cfg.eta * tail_b
    d_tail_by_path = np.asarray([np.sum(np.abs(tail_a[k] - tail_b)) for k in range(path_set.n_paths)], dtype=float)
    d_p_by_path = np.asarray([np.sum(np.abs(tilts_a[k] - tilt_b)) for k in range(path_set.n_paths)], dtype=float)
    worst_idx = int(np.argmax(d_tail_by_path))
    out: dict[str, object] = {'D_tail_AB': float(np.max(d_tail_by_path)), 'Mean_D_tail_AB': float(np.mean(d_tail_by_path)), 'D_p_AB': float(np.max(d_p_by_path)), 'Mean_D_p_AB': float(np.mean(d_p_by_path)), 'TailTV_max_AB': float(0.5 * np.max(d_p_by_path)), 'WorstTailPath': f'P{worst_idx + 1}', 'TailSet_B_at_fB': _tail_set_string(instance.scenario_names, tail_b)}
    for k in range(path_set.n_paths):
        out[f'D_tail_P{k + 1}'] = float(d_tail_by_path[k])
        out[f'D_p_P{k + 1}'] = float(d_p_by_path[k])
        out[f'TailSet_A_P{k + 1}_at_fB'] = _tail_set_string(instance.scenario_names, tail_a[k])
    return out

def _solve_model_b_direct_objective_fallback(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> tuple[np.ndarray, opt.OptimizeResult]:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    n = path_set.n_paths
    deterministic_penalty = overlap_penalty(instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)

    def objective_f(f_raw: np.ndarray) -> float:
        f = np.asarray(f_raw, dtype=float)
        if np.any(f < -1e-07):
            return 1e+30
        f = np.maximum(f, 0.0)
        total = f.sum()
        if total <= 0.0:
            return 1e+30
        f = cfg.q * f / total
        vals = potentials_by_scenario(instance, path_set, f, cfg)
        ce, _, _ = mean_cvar_value_and_tilt(vals, instance.probabilities, alpha, eta)
        return float(ce + entropy(f, cfg) / cfg.theta + np.dot(deterministic_penalty, f))
    constraints = ({'type': 'eq', 'fun': lambda f: float(np.sum(f) - cfg.q)},)
    bounds = opt.Bounds(np.zeros(n), np.full(n, cfg.q))
    starts: list[np.ndarray] = [np.full(n, cfg.q / n, dtype=float)]
    try:
        rn = solve_fixed_point(instance, path_set, 'RN fallback start', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg), cfg, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2, tol=1e-08, max_iter=20000).flows
        starts.append(rn)
    except Exception:
        pass
    for k in range(n):
        e = np.zeros(n, dtype=float)
        e[k] = cfg.q
        starts.append(e)
    starts += [np.array([0.5, 0.5, 0.0]) * cfg.q, np.array([0.5, 0.0, 0.5]) * cfg.q, np.array([0.0, 0.5, 0.5]) * cfg.q]
    best: opt.OptimizeResult | None = None
    for x0 in starts:
        res = opt.minimize(objective_f, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-10, 'maxiter': 2500, 'disp': False})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('Fallback Model B direct objective solver did not run.')
    f = np.maximum(np.asarray(best.x, dtype=float), 0.0)
    if f.sum() <= 0.0:
        f[:] = cfg.q / n
    else:
        f *= cfg.q / f.sum()
    best.x = f
    return (f, best)

def solve_model_b_for_params(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    start = time.perf_counter()
    used_fallback = False
    try:
        f, res = solve_model_b_convex_sp_direct_eta(instance, path_set, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
    except Exception:
        used_fallback = True
        f, res = _solve_model_b_direct_objective_fallback(instance, path_set, cfg_base, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
    raw = model_b_raw_costs(instance, path_set, f, cfg)[0]
    perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    _, _, pi = truncated_flows_from_costs(perceived, cfg)
    elapsed = time.perf_counter() - start
    result = EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model B common-state TSUE', overlap_model=overlap_model, beta_ps=beta_ps, flows=f, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=f / cfg.q, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, f), iterations=int(getattr(res, 'nit', -1)), residual=float('nan') if not used_fallback else float(getattr(res, 'optimality', np.nan)), active_paths=tuple((path_set.path_names[i] for i, val in enumerate(f) if val > 1e-06)), cpu_seconds=elapsed)
    return result

def run_param_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, cases: Sequence[dict[str, float]], parameterization: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    last_a: dict[tuple[float, float], np.ndarray] = {}
    for case in cases:
        alpha = float(case['alpha'])
        eta = float(case['eta'])
        lam = case.get('lambda', np.nan)
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=alpha, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        res_b = solve_model_b_for_params_labeled(instance, path_set, cfg, alpha, eta)
        metrics_b = system_risk_metrics(instance, path_set, res_b.flows, alpha, eta, cfg_case)
        row_b: dict[str, object] = {'Parameterization': parameterization, 'Instance': instance.name, 'PathSet': path_set.name, 'Model': 'Model B', 'RiskCase': classify_risk_case(eta), 'alpha': alpha, 'lambda': lam, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha) if eta > 0 else 0.0, 'pi_od': res_b.pi_od, 'ActiveCount': len(res_b.active_paths), 'Iterations': res_b.iterations, 'Residual': res_b.residual, 'CPUSeconds': res_b.cpu_seconds}
        row_b.update(metrics_b)
        for k, name in enumerate(path_set.path_names):
            row_b[f'Path{k + 1}'] = name
            row_b[f'f{k + 1}'] = res_b.flows[k]
            row_b[f'P{k + 1}'] = res_b.probabilities[k]
            row_b[f'g{k + 1}'] = res_b.raw_costs[k]
        rows.append(row_b)
        init = last_a.get((alpha, eta), res_b.flows)
        res_a = solve_model_a_fast(instance, path_set, cfg, alpha, eta, initial_flows=init)
        last_a[alpha, eta] = res_a.flows
        metrics_a = system_risk_metrics(instance, path_set, res_a.flows, alpha, eta, cfg_case)
        row_a: dict[str, object] = {'Parameterization': parameterization, 'Instance': instance.name, 'PathSet': path_set.name, 'Model': 'Model A', 'RiskCase': classify_risk_case(eta), 'alpha': alpha, 'lambda': lam, 'eta': eta, 'tail_hinge_coeff': eta / (1.0 - alpha) if eta > 0 else 0.0, 'pi_od': res_a.pi_od, 'ActiveCount': len(res_a.active_paths), 'Iterations': res_a.iterations, 'Residual': res_a.residual, 'CPUSeconds': res_a.cpu_seconds, 'Gap_A': vi_gap_model_a(instance, path_set, res_a.flows, cfg_case)}
        row_a.update(metrics_a)
        for k, name in enumerate(path_set.path_names):
            row_a[f'Path{k + 1}'] = name
            row_a[f'f{k + 1}'] = res_a.flows[k]
            row_a[f'P{k + 1}'] = res_a.probabilities[k]
            row_a[f'g{k + 1}'] = res_a.raw_costs[k]
        rows.append(row_a)
    sort_cols = ['alpha', 'lambda', 'eta', 'Model'] if 'lambda' in pd.DataFrame(cases).columns else ['alpha', 'eta', 'Model']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def diagnostics_from_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, grid_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in grid_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keydict = dict(zip(group_cols, keys))
        sub_a = sub[sub['Model'] == 'Model A']
        sub_b = sub[sub['Model'] == 'Model B']
        if sub_a.empty or sub_b.empty:
            continue
        alpha = float(sub_b.iloc[0]['alpha'])
        eta = float(sub_b.iloc[0]['eta'])
        f_a = np.array([float(sub_a.iloc[0][f'f{k + 1}']) for k in range(path_set.n_paths)])
        f_b = np.array([float(sub_b.iloc[0][f'f{k + 1}']) for k in range(path_set.n_paths)])
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=alpha, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        row: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Parameterization': str(sub_b.iloc[0]['Parameterization']), 'alpha': alpha, 'eta': eta, 'RiskCase': classify_risk_case(eta), 'D_f_AB': float(np.sum(np.abs(f_a - f_b)) / cfg.q), 'Gap_A_at_fB': vi_gap_model_a_limit(instance, path_set, f_b, cfg_case) if alpha_is_limit else vi_gap_model_a(instance, path_set, f_b, cfg_case), 'Active_A': int(sub_a.iloc[0]['ActiveCount']), 'Active_B': int(sub_b.iloc[0]['ActiveCount']), 'Active_A_minus_Active_B': int(sub_a.iloc[0]['ActiveCount'] - sub_b.iloc[0]['ActiveCount'])}
        if 'lambda' in grid_df.columns:
            row['lambda'] = float(sub_b.iloc[0]['lambda']) if not pd.isna(sub_b.iloc[0]['lambda']) else np.nan
        if alpha_is_limit:
            row.update(tail_divergence_limit_at_fB(instance, path_set, f_b, cfg_case))
        else:
            row.update(tail_divergence_at_fB(instance, path_set, f_b, cfg_case))
        rows.append(row)
    sort_cols = ['alpha', 'lambda', 'eta'] if 'lambda' in grid_df.columns else ['alpha', 'eta']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def compact_modelb_ab_table(modelb_df: pd.DataFrame, diag_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    key_cols = ['alpha', 'lambda', 'eta'] if include_lambda else ['alpha', 'eta']
    diag_cols = key_cols + ['Instance', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    return modelb_df.merge(diag_df[diag_cols], on=key_cols, how='left').sort_values(key_cols).reset_index(drop=True)

def modelb_base_table_from_grid(grid_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    sub = grid_df[grid_df['Model'] == 'Model B'].copy()
    keep = ['RiskCase', 'alpha'] + (['lambda'] if include_lambda else []) + ['eta', 'tail_hinge_coeff', 'P1', 'P2', 'P3', 'pi_od', 'ActiveCount', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    return sub[keep].sort_values(['alpha'] + (['lambda'] if include_lambda else ['eta'])).reset_index(drop=True)

@dataclass(frozen=True)
class DROSettings:
    alpha: float = 0.95
    eta: float = 0.5
    train_counts: tuple[int, ...] = (6, 2, 2, 1, 0)
    test_probabilities: tuple[float, ...] = (0.48, 0.17, 0.16, 0.12, 0.07)
    radii: tuple[float, ...] = (0.05, 0.1, 0.2)

@dataclass(frozen=True)
class DROSolveResult:
    model: str
    rho: float
    flows: np.ndarray
    probabilities: np.ndarray
    active_count: int
    objective: float
    t_ru: float
    kappa: float
    iterations: int
    residual: float
    cpu_seconds: float
    worst_case_probabilities: np.ndarray | None
SCENARIO_VECTORS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.85, 0.85, 0.85]], dtype=float)

def instance_with_probabilities(instance: BraessInstance, probabilities: Sequence[float], name: str) -> BraessInstance:
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape[0] != len(instance.scenario_names):
        raise ValueError('Probability vector length does not match scenario count.')
    if np.any(probs < 0.0) or not np.isclose(probs.sum(), 1.0, atol=1e-10):
        raise ValueError('Probabilities must be nonnegative and sum to one.')
    return BraessInstance(name=name, scenario_names=instance.scenario_names, probabilities=probs, link_params=instance.link_params)

def build_reliable_bc_upgrade(instance: BraessInstance) -> BraessInstance:
    params: dict[str, dict[str, LinkScenarioParams]] = {}
    for scenario in instance.scenario_names:
        params[scenario] = {}
        for link, old in instance.link_params[scenario].items():
            if link != 'BC':
                params[scenario][link] = old
                continue
            if scenario == 'Normal':
                params[scenario][link] = LinkScenarioParams(t0=1.76, capacity=78.0, delay=4.0)
            elif scenario == 'Shortcut incident':
                params[scenario][link] = LinkScenarioParams(t0=1.2, capacity=110.0, delay=3.0)
            elif scenario == 'System-wide':
                params[scenario][link] = LinkScenarioParams(t0=1.4, capacity=100.0, delay=5.0)
            else:
                params[scenario][link] = LinkScenarioParams(t0=1.2, capacity=95.0, delay=1.5)
    return BraessInstance(name='Reliable B-C upgrade', scenario_names=instance.scenario_names, probabilities=instance.probabilities.copy(), link_params=params)

def scenario_distance_matrix() -> np.ndarray:
    return np.linalg.norm(SCENARIO_VECTORS[:, None, :] - SCENARIO_VECTORS[None, :, :], axis=2)

def training_probabilities(settings: DROSettings) -> np.ndarray:
    counts = np.asarray(settings.train_counts, dtype=float)
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError('Training counts must be nonnegative and not all zero.')
    return counts / counts.sum()

def format_time_function(link_key: str, params: LinkScenarioParams) -> str:
    label = LINK_LABELS[link_key].replace('→', '')
    base = f't_{label}(x) = {params.t0:.2f}[1 + 0.15(x/{params.capacity:.2f})^4]'
    if abs(params.delay) > 1e-10:
        base += f' + {params.delay:.2f}'
    return base

def solve_weighted_finite_dro(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, center_probabilities: Sequence[float], rho: float, distances: np.ndarray, alpha: float, eta: float) -> DROSolveResult:
    n = path_set.n_paths
    scenario_count = len(instance.scenario_names)
    center = np.asarray(center_probabilities, dtype=float)
    observed = np.where(center > 1e-12)[0]
    weights = center[observed]
    obs_count = len(observed)
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    train_instance = instance_with_probabilities(instance, center, f'{instance.name} empirical center')
    empirical = solve_model_b_for_params_labeled(train_instance, path_set, cfg, alpha, eta)
    f0 = empirical.flows
    L0 = potentials_by_scenario(instance, path_set, f0, cfg)
    t0 = float(np.dot(center, L0))
    u0 = np.maximum(L0 - t0, 0.0)
    kappa0 = 200.0
    g0 = (1.0 - eta) * L0 + eta / (1.0 - alpha) * u0
    s0 = np.array([np.max(g0 - kappa0 * distances[:, idx]) for idx in observed], dtype=float)
    x0 = np.concatenate([f0, [t0, kappa0], s0, u0])
    L_vertices: list[float] = []
    for k in range(n):
        f_vertex = np.zeros(n, dtype=float)
        f_vertex[k] = cfg.q
        L_vertices.extend(potentials_by_scenario(instance, path_set, f_vertex, cfg))
    t_upper = 2.0 * max(L_vertices) + 100.0
    lower = np.array([0.0] * n + [0.0, 0.0] + [0.0] * obs_count + [0.0] * scenario_count, dtype=float)
    upper = np.array([np.inf] * n + [t_upper, 10000.0] + [np.inf] * obs_count + [np.inf] * scenario_count, dtype=float)
    bounds = opt.Bounds(lower, upper)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        f = x[:n]
        t_ru = float(x[n])
        kappa = float(x[n + 1])
        s = x[n + 2:n + 2 + obs_count]
        u = x[n + 2 + obs_count:]
        return (f, t_ru, kappa, s, u)

    def objective(x: np.ndarray) -> float:
        f, t_ru, kappa, s, _u = unpack(x)
        return eta * t_ru + kappa * rho + float(np.dot(weights, s)) + entropy(f, cfg) / cfg.theta

    def eq_constraint(x: np.ndarray) -> np.ndarray:
        return np.array([np.sum(x[:n]) - cfg.q], dtype=float)

    def ineq_constraint(x: np.ndarray) -> np.ndarray:
        f, t_ru, kappa, s, u = unpack(x)
        L_vals = potentials_by_scenario(instance, path_set, f, cfg)
        robust_hinge = (1.0 - eta) * L_vals + eta / (1.0 - alpha) * u
        hinge_constraints = u - (L_vals - t_ru)
        support_constraints = s[:, None] - (robust_hinge[None, :] - kappa * distances[:, observed].T)
        return np.concatenate([hinge_constraints, support_constraints.ravel()])
    start = time.perf_counter()
    res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[{'type': 'eq', 'fun': eq_constraint}, {'type': 'ineq', 'fun': ineq_constraint}], options={'ftol': 1e-09, 'maxiter': 1200, 'disp': False})
    elapsed = time.perf_counter() - start
    if not res.success:
        raise RuntimeError(f'Finite-support DRO solve failed for {instance.name}, rho={rho}: {res.message}')
    f = np.maximum(res.x[:n], 0.0)
    if f.sum() <= 0:
        raise RuntimeError('DRO solver returned zero total flow.')
    f *= cfg.q / f.sum()
    t_ru = float(res.x[n])
    kappa = float(res.x[n + 1])
    residual = float(max(abs(eq_constraint(res.x)[0]), max(0.0, -np.min(ineq_constraint(res.x)))))
    worst_case = recover_worst_case_probabilities(instance, path_set, f, cfg, center, rho, distances, t_ru)
    return DROSolveResult(model='Wasserstein DRO Model B', rho=float(rho), flows=f, probabilities=f / cfg.q, active_count=int(np.sum(f > 1e-06)), objective=float(res.fun), t_ru=t_ru, kappa=kappa, iterations=int(getattr(res, 'nit', -1)), residual=residual, cpu_seconds=elapsed, worst_case_probabilities=worst_case)

def recover_worst_case_probabilities(instance: BraessInstance, path_set: PathSet, flows: Sequence[float], cfg: SolverConfig, center_probabilities: Sequence[float], rho: float, distances: np.ndarray, t_ru: float) -> np.ndarray | None:
    center = np.asarray(center_probabilities, dtype=float)
    observed = np.where(center > 1e-12)[0]
    weights = center[observed]
    L_vals = potentials_by_scenario(instance, path_set, flows, cfg)
    g_vals = (1.0 - cfg.eta) * L_vals + cfg.eta / (1.0 - cfg.alpha) * np.maximum(L_vals - t_ru, 0.0)
    obs_count = len(observed)
    scenario_count = len(L_vals)
    c = -np.tile(g_vals, obs_count)
    A_eq = np.zeros((obs_count, obs_count * scenario_count), dtype=float)
    for row, _idx in enumerate(observed):
        A_eq[row, row * scenario_count:(row + 1) * scenario_count] = 1.0
    b_eq = weights.copy()
    A_ub = np.zeros((1, obs_count * scenario_count), dtype=float)
    for row, idx in enumerate(observed):
        A_ub[0, row * scenario_count:(row + 1) * scenario_count] = distances[:, idx]
    result = linprog(c, A_ub=A_ub, b_ub=np.array([rho]), A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method='highs')
    if not result.success:
        return None
    plan = result.x.reshape(obs_count, scenario_count)
    return plan.sum(axis=0)

def empirical_model_solution(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, center_probabilities: Sequence[float], alpha: float, eta: float, label: str) -> DROSolveResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    empirical_instance = instance_with_probabilities(instance, center_probabilities, f'{instance.name} empirical')
    start = time.perf_counter()
    res = solve_model_b_for_params_labeled(empirical_instance, path_set, cfg, alpha, eta)
    elapsed = time.perf_counter() - start
    return DROSolveResult(model=label, rho=0.0, flows=res.flows, probabilities=res.flows / cfg.q, active_count=len(res.active_paths), objective=float('nan'), t_ru=float('nan'), kappa=0.0, iterations=res.iterations, residual=float(res.residual) if not math.isnan(res.residual) else float('nan'), cpu_seconds=elapsed, worst_case_probabilities=np.asarray(center_probabilities, dtype=float))

def test_metrics(instance: BraessInstance, path_set: PathSet, flows: Sequence[float], cfg: SolverConfig, test_probabilities: Sequence[float]) -> dict[str, float | str]:
    probs = np.asarray(test_probabilities, dtype=float)
    L_vals = potentials_by_scenario(instance, path_set, flows, cfg)
    mean_val = float(np.dot(probs, L_vals))
    cvar_val = float(cvar_value(L_vals, probs, cfg.alpha))
    ce_val = (1.0 - cfg.eta) * mean_val + cfg.eta * cvar_val
    order = np.argsort(L_vals)
    cumulative = np.cumsum(probs[order])
    var_idx = order[np.searchsorted(cumulative, cfg.alpha, side='left')]
    tail = selected_tail_scenario_name(L_vals, probs, cfg.alpha, instance.scenario_names)
    return {'MeanTest': mean_val, 'VaRTest': float(L_vals[var_idx]), 'CVaRTest': cvar_val, 'CETest': ce_val, 'TestTailSet': tail}

def solve_all_dro_decisions(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = DROSettings()
    cfg = SolverConfig(alpha=settings.alpha, eta=settings.eta)
    path_set = base_path_set()
    before = build_route_local_instance()
    after = build_reliable_bc_upgrade(before)
    center = training_probabilities(settings)
    distances = scenario_distance_matrix()
    test_probs = np.asarray(settings.test_probabilities, dtype=float)
    scenario_rows = []
    for name, train_p, test_p, vector in zip(before.scenario_names, center, test_probs, SCENARIO_VECTORS):
        scenario_rows.append({'Scenario': name, 'TrainingProbability': train_p, 'TestProbability': test_p, 'xi_upper': vector[0], 'xi_lower': vector[1], 'xi_shortcut': vector[2]})
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(output_dir / 'braess_dro_training_test_laws.csv', index=False)
    pd.DataFrame(distances, index=before.scenario_names, columns=before.scenario_names).to_csv(output_dir / 'braess_dro_scenario_distance_matrix.csv')
    models: list[tuple[str, float, DROSolveResult, DROSolveResult]] = []
    rn_before = empirical_model_solution(before, path_set, cfg, center, settings.alpha, 0.0, 'Risk-neutral TSUE')
    rn_after = empirical_model_solution(after, path_set, cfg, center, settings.alpha, 0.0, 'Risk-neutral TSUE')
    models.append(('Risk-neutral TSUE', 0.0, rn_before, rn_after))
    sp_before = empirical_model_solution(before, path_set, cfg, center, settings.alpha, settings.eta, 'Empirical Model B-SP')
    sp_after = empirical_model_solution(after, path_set, cfg, center, settings.alpha, settings.eta, 'Empirical Model B-SP')
    models.append(('Empirical Model B-SP', 0.0, sp_before, sp_after))
    for rho in settings.radii:
        dro_before = solve_weighted_finite_dro(before, path_set, cfg, center, rho, distances, settings.alpha, settings.eta)
        dro_after = solve_weighted_finite_dro(after, path_set, cfg, center, rho, distances, settings.alpha, settings.eta)
        models.append(('Wasserstein DRO Model B', rho, dro_before, dro_after))
    summary_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    worst_rows: list[dict[str, object]] = []
    for model_name, rho, before_res, after_res in models:
        cfg_eval = SolverConfig(alpha=settings.alpha, eta=settings.eta)
        before_metrics = test_metrics(before, path_set, before_res.flows, cfg_eval, test_probs)
        after_metrics = test_metrics(after, path_set, after_res.flows, cfg_eval, test_probs)
        summary_rows.append({'Model': model_name, 'rho': rho, 'MeanTestBefore': before_metrics['MeanTest'], 'MeanTestAfter': after_metrics['MeanTest'], 'DeltaMeanTest': after_metrics['MeanTest'] - before_metrics['MeanTest'], 'VaRTestBefore': before_metrics['VaRTest'], 'VaRTestAfter': after_metrics['VaRTest'], 'DeltaVaRTest': after_metrics['VaRTest'] - before_metrics['VaRTest'], 'CVaRTestBefore': before_metrics['CVaRTest'], 'CVaRTestAfter': after_metrics['CVaRTest'], 'DeltaCVaRTest': after_metrics['CVaRTest'] - before_metrics['CVaRTest'], 'CETestBefore': before_metrics['CETest'], 'CETestAfter': after_metrics['CETest'], 'DeltaCETest': after_metrics['CETest'] - before_metrics['CETest'], 'ActiveBefore': before_res.active_count, 'ActiveAfter': after_res.active_count, 'DeltaActive': after_res.active_count - before_res.active_count, 'P3Before': before_res.probabilities[2], 'P3After': after_res.probabilities[2], 'DeltaP3': after_res.probabilities[2] - before_res.probabilities[2], 'BeforeTailSet': before_metrics['TestTailSet'], 'AfterTailSet': after_metrics['TestTailSet'], 'DecisionByCE': 'Improve' if after_metrics['CETest'] < before_metrics['CETest'] else 'Do not improve', 'IterationsBefore': before_res.iterations, 'IterationsAfter': after_res.iterations, 'CPUSecondsBefore': before_res.cpu_seconds, 'CPUSecondsAfter': after_res.cpu_seconds})
        for network, result in (('Before', before_res), ('After', after_res)):
            row = {'Model': model_name, 'rho': rho, 'Network': network, 'ActiveCount': result.active_count, 'Objective': result.objective, 't_RU': result.t_ru, 'kappa': result.kappa}
            for i, path_name in enumerate(path_set.path_names):
                row[f'Path{i + 1}'] = path_name
                row[f'f{i + 1}'] = result.flows[i]
                row[f'P{i + 1}'] = result.probabilities[i]
            flow_rows.append(row)
            if result.worst_case_probabilities is not None:
                worst_row = {'Model': model_name, 'rho': rho, 'Network': network}
                for scen, prob in zip(before.scenario_names, result.worst_case_probabilities):
                    worst_row[scen] = prob
                worst_rows.append(worst_row)
    summary_df = pd.DataFrame(summary_rows)
    flows_df = pd.DataFrame(flow_rows)
    worst_df = pd.DataFrame(worst_rows)
    summary_df.to_csv(output_dir / 'braess_dro_reliable_bc_decision_summary.csv', index=False)
    flows_df.to_csv(output_dir / 'braess_dro_reliable_bc_flows.csv', index=False)
    worst_df.to_csv(output_dir / 'braess_dro_worst_case_probabilities.csv', index=False)
    return (summary_df, flows_df, worst_df)

def run_all_v16(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir)
    cfg = SolverConfig()
    base_paths = base_path_set()
    route_local = build_route_local_instance()
    direct_cases = []
    for alpha in [0.9, 0.95]:
        for eta in [0.0] + [round(float(x), 1) for x in np.arange(0.1, 0.81, 0.1)] + [1.0]:
            direct_cases.append({'alpha': alpha, 'eta': eta})
    direct_grid = run_param_grid(route_local, base_paths, cfg, direct_cases, 'direct-eta')
    direct_grid.to_csv(output_dir / 'braess_direct_eta_risk_sensitivity_all_models.csv', index=False)
    direct_diag = diagnostics_from_grid(route_local, base_paths, cfg, direct_grid, ['alpha', 'eta'])
    direct_diag.to_csv(output_dir / 'braess_modelA_modelB_eta_grid_diagnostics.csv', index=False)
    direct_modelb = compact_modelb_ab_table(modelb_base_table_from_grid(direct_grid, include_lambda=False), direct_diag, include_lambda=False)
    direct_modelb.to_csv(output_dir / 'braess_modelB_direct_eta_with_AB_diagnostics.csv', index=False)
    direct_modelb.to_csv(output_dir / 'braess_modelB_direct_eta_risk_table.csv', index=False)
    lambda_cases = []
    for alpha in [0.9, 0.95]:
        for lam in [round(float(x), 1) for x in np.arange(0.0, 0.91, 0.1)]:
            if lam <= alpha + 1e-12:
                lambda_cases.append({'alpha': alpha, 'lambda': lam, 'eta': eta_from_lambda(alpha, lam)})
    lambda_grid = run_param_grid(route_local, base_paths, cfg, lambda_cases, 'lambda-mapped')
    lambda_grid.to_csv(output_dir / 'braess_lambda_mapped_risk_sensitivity_all_models.csv', index=False)
    lambda_diag = diagnostics_from_grid(route_local, base_paths, cfg, lambda_grid, ['alpha', 'lambda', 'eta'])
    lambda_diag.to_csv(output_dir / 'braess_modelA_modelB_lambda_grid_diagnostics.csv', index=False)
    lambda_modelb = compact_modelb_ab_table(modelb_base_table_from_grid(lambda_grid, include_lambda=True), lambda_diag, include_lambda=True)
    lambda_modelb.to_csv(output_dir / 'braess_modelB_lambda_mapped_with_AB_diagnostics.csv', index=False)
    solve_all_dro_decisions(output_dir)
    print('[v16] regime-conditioned ambiguity experiment', flush=True)
    run_braess_regime_experiment(output_dir)
    with (output_dir / 'README.md').open('a', encoding='utf-8') as fh:
        fh.write('\nStandalone v16 outputs added:\n- Direct-eta and lambda-mapped Model B tables with Model A/B diagnostics.\n- Finite-support Wasserstein DRO reliable B-C upgrade decision tables.\n- Regime-conditioned ambiguity tables: fixed-frequency conditional DRO, joint-frequency DRO, and regime-informed benchmark.\n- This script is self-contained and imports no previous braess_integrated_experiment_v*.py files.\n')

@dataclass(frozen=True)
class RegimeLibrary:
    regimes: tuple[str, ...]
    p0: np.ndarray
    conditional_probabilities: dict[str, np.ndarray]
    rho: dict[str, float]
    scenario_indices: dict[str, np.ndarray]
    scenario_vectors: np.ndarray
    epsilon_p: float

def _scale_link_params(prm: LinkScenarioParams, t_mult: float=1.0, cap_mult: float=1.0, delay_add: float=0.0) -> LinkScenarioParams:
    return LinkScenarioParams(t0=prm.t0 * t_mult, capacity=prm.capacity * cap_mult, delay=prm.delay + delay_add)

def build_regime_conditioned_braess_instance() -> tuple[BraessInstance, RegimeLibrary]:
    route_local = build_route_local_instance()
    scenario_names = ('Normal-low', 'Normal-high', 'Upper incident', 'Lower incident', 'Shortcut incident', 'Flood-moderate', 'Flood-severe')
    params: dict[str, dict[str, LinkScenarioParams]] = {}
    params['Normal-low'] = dict(route_local.link_params['Normal'])
    params['Normal-high'] = {link: _scale_link_params(route_local.link_params['Normal'][link], t_mult=1.04, cap_mult=0.93, delay_add=0.55) for link in LINKS}
    for state in ('Upper incident', 'Lower incident', 'Shortcut incident'):
        params[state] = dict(route_local.link_params[state])
    normal = route_local.link_params['Normal']
    params['Flood-moderate'] = {link: _scale_link_params(normal[link], t_mult=1.15, cap_mult=0.72, delay_add=6.5) for link in LINKS}
    params['Flood-severe'] = dict(route_local.link_params['System-wide'])
    bc = params['Flood-severe']['BC']
    params['Flood-severe']['BC'] = LinkScenarioParams(t0=bc.t0 * 1.05, capacity=bc.capacity * 0.35, delay=bc.delay + 15.0)
    regimes = ('Normal', 'Localized incident', 'Flood')
    scenario_indices = {'Normal': np.array([0, 1], dtype=int), 'Localized incident': np.array([2, 3, 4], dtype=int), 'Flood': np.array([5, 6], dtype=int)}
    p0 = np.array([0.55, 0.41, 0.04], dtype=float)
    conditional_probabilities = {'Normal': np.array([0.7, 0.3], dtype=float), 'Localized incident': np.array([0.18, 0.17, 0.06], dtype=float) / 0.41, 'Flood': np.array([0.65, 0.35], dtype=float)}
    rho = {'Normal': 0.04, 'Localized incident': 0.1, 'Flood': 0.14}
    scenario_vectors = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.05, 0.05], [0.05, 0.75, 0.05], [0.05, 0.05, 0.75], [0.62, 0.62, 0.62], [1.0, 1.0, 1.0]], dtype=float)
    lib = RegimeLibrary(regimes=regimes, p0=p0, conditional_probabilities=conditional_probabilities, rho=rho, scenario_indices=scenario_indices, scenario_vectors=scenario_vectors, epsilon_p=0.12)
    probabilities = np.zeros(len(scenario_names), dtype=float)
    for w_idx, regime in enumerate(regimes):
        probabilities[scenario_indices[regime]] = p0[w_idx] * conditional_probabilities[regime]
    instance = BraessInstance(name='Regime-conditioned Braess', scenario_names=scenario_names, probabilities=probabilities, link_params=params)
    validate_instance(instance)
    return (instance, lib)

def conditional_distance_matrix(lib: RegimeLibrary, regime: str) -> np.ndarray:
    idx = lib.scenario_indices[regime]
    vectors = lib.scenario_vectors[idx]
    return np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)

def worst_case_frequency_vector(phi: Sequence[float], p0: Sequence[float], epsilon_p: float) -> tuple[float, np.ndarray, int]:
    phi_arr = np.asarray(phi, dtype=float)
    p = np.asarray(p0, dtype=float).copy()
    budget = max(float(epsilon_p), 0.0) / 2.0
    donors = list(np.argsort(phi_arr))
    receivers = list(np.argsort(-phi_arr))
    for receiver in receivers:
        if budget <= 1e-12:
            break
        for donor in donors:
            if budget <= 1e-12:
                break
            if donor == receiver or phi_arr[receiver] <= phi_arr[donor] + 1e-12:
                continue
            delta = min(budget, p[donor], 1.0 - p[receiver])
            if delta <= 1e-12:
                continue
            p[donor] -= delta
            p[receiver] += delta
            budget -= delta
    return (float(np.dot(p, phi_arr)), p, int(np.argmax(phi_arr)))

def _regime_unpack_exante(x: np.ndarray, n_paths: int, n_regimes: int, s_slices: dict[str, slice], n_scenarios: int) -> tuple[np.ndarray, float, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    f = x[:n_paths]
    t_ru = float(x[n_paths])
    kappa = x[n_paths + 1:n_paths + 1 + n_regimes]
    s_dict = {regime: x[sl] for regime, sl in s_slices.items()}
    u = x[-n_scenarios:]
    return (f, t_ru, kappa, s_dict, u)

def solve_regime_exante_dro(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, lib: RegimeLibrary, *, joint_frequency: bool=False) -> dict[str, object]:
    n_paths = path_set.n_paths
    n_regimes = len(lib.regimes)
    n_scenarios = len(instance.scenario_names)
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=cfg_base.alpha, eta=cfg_base.eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    empirical = solve_model_b_for_params_labeled(instance, path_set, cfg, cfg.alpha, cfg.eta)
    f0 = empirical.flows
    L0 = potentials_by_scenario(instance, path_set, f0, cfg)
    t0 = float(np.dot(instance.probabilities, L0))
    u0 = np.maximum(L0 - t0, 0.0)
    kappa0 = np.full(n_regimes, 200.0, dtype=float)
    s0_parts: list[np.ndarray] = []
    s_slices: dict[str, slice] = {}
    pos = n_paths + 1 + n_regimes
    for w_idx, regime in enumerate(lib.regimes):
        idx = lib.scenario_indices[regime]
        dist = conditional_distance_matrix(lib, regime)
        g0 = (1.0 - cfg.eta) * L0[idx] + cfg.eta / (1.0 - cfg.alpha) * u0[idx]
        s_w0 = np.array([np.max(g0 - kappa0[w_idx] * dist[:, i]) for i in range(len(idx))], dtype=float)
        s0_parts.append(s_w0)
        s_slices[regime] = slice(pos, pos + len(idx))
        pos += len(idx)
    x0 = np.concatenate([f0, [t0], kappa0, *s0_parts, u0])
    L_vertices: list[float] = []
    for k in range(n_paths):
        f_vertex = np.zeros(n_paths, dtype=float)
        f_vertex[k] = cfg.q
        L_vertices.extend(potentials_by_scenario(instance, path_set, f_vertex, cfg))
    t_upper = 2.0 * max(L_vertices) + 100.0
    n_s_vars = sum((len(lib.scenario_indices[regime]) for regime in lib.regimes))
    lower = np.concatenate([np.zeros(n_paths), np.array([0.0]), np.zeros(n_regimes), np.zeros(n_s_vars), np.zeros(n_scenarios)])
    upper = np.concatenate([np.full(n_paths, np.inf), np.array([t_upper]), np.full(n_regimes, 10000.0), np.full(n_s_vars, np.inf), np.full(n_scenarios, np.inf)])
    bounds = opt.Bounds(lower, upper)

    def phi_from_vars(x: np.ndarray) -> np.ndarray:
        _f, _t_ru, kappa, s_dict, _u = _regime_unpack_exante(x, n_paths, n_regimes, s_slices, n_scenarios)
        phi = np.zeros(n_regimes, dtype=float)
        for w_idx, regime in enumerate(lib.regimes):
            phi[w_idx] = kappa[w_idx] * lib.rho[regime] + float(np.dot(lib.conditional_probabilities[regime], s_dict[regime]))
        return phi

    def objective(x: np.ndarray) -> float:
        f, t_ru, _kappa, _s_dict, _u = _regime_unpack_exante(x, n_paths, n_regimes, s_slices, n_scenarios)
        phi = phi_from_vars(x)
        if joint_frequency:
            freq_value, _p_star, _worst_idx = worst_case_frequency_vector(phi, lib.p0, lib.epsilon_p)
        else:
            freq_value = float(np.dot(lib.p0, phi))
        return cfg.eta * t_ru + freq_value + entropy(f, cfg) / cfg.theta

    def eq_constraint(x: np.ndarray) -> np.ndarray:
        return np.array([np.sum(x[:n_paths]) - cfg.q], dtype=float)

    def ineq_constraint(x: np.ndarray) -> np.ndarray:
        f, t_ru, kappa, s_dict, u = _regime_unpack_exante(x, n_paths, n_regimes, s_slices, n_scenarios)
        L_vals = potentials_by_scenario(instance, path_set, f, cfg)
        constraints: list[np.ndarray] = [u - (L_vals - t_ru)]
        for w_idx, regime in enumerate(lib.regimes):
            idx = lib.scenario_indices[regime]
            dist = conditional_distance_matrix(lib, regime)
            g_vals = (1.0 - cfg.eta) * L_vals[idx] + cfg.eta / (1.0 - cfg.alpha) * u[idx]
            s_w = s_dict[regime]
            support = s_w[:, None] - (g_vals[None, :] - kappa[w_idx] * dist.T)
            constraints.append(support.ravel())
        return np.concatenate(constraints)
    start = time.perf_counter()
    result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[{'type': 'eq', 'fun': eq_constraint}, {'type': 'ineq', 'fun': ineq_constraint}], options={'ftol': 1e-09, 'maxiter': 2400, 'disp': False})
    cpu = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f'Regime ex ante DRO failed: {result.message}')
    f, t_ru, kappa, _s_dict, _u = _regime_unpack_exante(result.x, n_paths, n_regimes, s_slices, n_scenarios)
    f = np.maximum(f, 0.0)
    f *= cfg.q / f.sum()
    phi = phi_from_vars(result.x)
    if joint_frequency:
        _freq_value, p_star, worst_idx = worst_case_frequency_vector(phi, lib.p0, lib.epsilon_p)
    else:
        p_star = lib.p0.copy()
        worst_idx = int(np.argmax(phi))
    residual = float(max(abs(eq_constraint(result.x)[0]), max(0.0, -np.min(ineq_constraint(result.x)))))
    return {'model': 'Joint-frequency regime DRO' if joint_frequency else 'Fixed-frequency conditional DRO', 'flows': f, 'probabilities': f / cfg.q, 'objective': float(result.fun), 't_ru': t_ru, 'kappa': kappa, 'phi': phi, 'p_star': p_star, 'worst_regime': lib.regimes[worst_idx], 'iterations': int(getattr(result, 'nit', -1)), 'residual': residual, 'cpu_seconds': cpu, 'active_count': int(np.sum(f > 1e-06))}

def sub_instance_for_regime(instance: BraessInstance, lib: RegimeLibrary, regime: str) -> BraessInstance:
    idx = lib.scenario_indices[regime]
    names = tuple((instance.scenario_names[i] for i in idx))
    params = {name: instance.link_params[name] for name in names}
    return BraessInstance(name=f'{instance.name}: {regime}', scenario_names=names, probabilities=lib.conditional_probabilities[regime].copy(), link_params=params)

def solve_regime_informed_benchmark(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, lib: RegimeLibrary) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    weighted_objective = 0.0
    for w_idx, regime in enumerate(lib.regimes):
        sub = sub_instance_for_regime(instance, lib, regime)
        dist = conditional_distance_matrix(lib, regime)
        result = solve_weighted_finite_dro(sub, path_set, cfg, lib.conditional_probabilities[regime], lib.rho[regime], dist, cfg.alpha, cfg.eta)
        weighted_objective += lib.p0[w_idx] * result.objective
        rows.append({'Regime': regime, 'p0': lib.p0[w_idx], 'rho': lib.rho[regime], 'Objective': result.objective, 'WeightedObjective': lib.p0[w_idx] * result.objective, 'P1': result.probabilities[0], 'P2': result.probabilities[1], 'P3': result.probabilities[2], 'ActiveCount': result.active_count, 'Iterations': result.iterations, 'CPUSeconds': result.cpu_seconds})
    return {'model': 'Regime-informed benchmark', 'rows': pd.DataFrame(rows), 'objective': float(weighted_objective)}

def evaluate_mixture_ce(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, flows: Sequence[float], probabilities: Sequence[float] | None=None) -> dict[str, float]:
    probs = instance.probabilities if probabilities is None else np.asarray(probabilities, dtype=float)
    L_vals = potentials_by_scenario(instance, path_set, flows, cfg)
    mean_val = float(np.dot(probs, L_vals))
    cvar_val = float(cvar_value(L_vals, probs, cfg.alpha))
    ce_val = (1.0 - cfg.eta) * mean_val + cfg.eta * cvar_val
    return {'MeanTest': mean_val, 'CVaRTest': cvar_val, 'CETest': ce_val}

def evaluate_regime_informed_ce(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, lib: RegimeLibrary, informed_rows: pd.DataFrame) -> dict[str, float]:
    mean_total = 0.0
    cvar_total = 0.0
    ce_total = 0.0
    for w_idx, regime in enumerate(lib.regimes):
        idx = lib.scenario_indices[regime]
        row = informed_rows[informed_rows['Regime'] == regime].iloc[0]
        f = cfg.q * np.array([row['P1'], row['P2'], row['P3']], dtype=float)
        L_vals = potentials_by_scenario(instance, path_set, f, cfg)[idx]
        p_cond = lib.conditional_probabilities[regime]
        mean_w = float(np.dot(p_cond, L_vals))
        cvar_w = float(cvar_value(L_vals, p_cond, cfg.alpha))
        ce_w = (1.0 - cfg.eta) * mean_w + cfg.eta * cvar_w
        mean_total += lib.p0[w_idx] * mean_w
        cvar_total += lib.p0[w_idx] * cvar_w
        ce_total += lib.p0[w_idx] * ce_w
    return {'MeanTest': mean_total, 'CVaRTest': cvar_total, 'CETest': ce_total}

def plot_regime_path_shares(summary: pd.DataFrame, informed: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        if row['Model'] == 'Regime-informed benchmark':
            continue
        for path in ('P1', 'P2', 'P3'):
            rows.append({'Label': row['Model'], 'Path': path, 'Share': row[path]})
    for _, row in informed.iterrows():
        for path in ('P1', 'P2', 'P3'):
            rows.append({'Label': f"Informed: {row['Regime']}", 'Path': path, 'Share': row[path]})
    plot_df = pd.DataFrame(rows)
    labels = list(dict.fromkeys(plot_df['Label'].tolist()))
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for k, path in enumerate(('P1', 'P2', 'P3')):
        vals = [float(plot_df[(plot_df['Label'] == label) & (plot_df['Path'] == path)]['Share'].iloc[0]) for label in labels]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=path)
        for bar in bars:
            h = bar.get_height()
            if h > 0.025:
                ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2.0, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Path share')
    ax.set_title('Regime-conditioned Braess: ex ante flows vs regime-informed benchmark')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(framealpha=0.95, fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_braess_regime_experiment(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    instance, lib = build_regime_conditioned_braess_instance()
    path_set = base_path_set()
    cfg = SolverConfig(alpha=0.95, eta=0.5)
    fixed = solve_regime_exante_dro(instance, path_set, cfg, lib, joint_frequency=False)
    joint = solve_regime_exante_dro(instance, path_set, cfg, lib, joint_frequency=True)
    informed = solve_regime_informed_benchmark(instance, path_set, cfg, lib)
    empirical = solve_model_b_for_params_labeled(instance, path_set, cfg, cfg.alpha, cfg.eta)
    empirical_metrics = evaluate_mixture_ce(instance, path_set, cfg, empirical.flows)
    fixed_metrics = evaluate_mixture_ce(instance, path_set, cfg, fixed['flows'])
    joint_metrics = evaluate_mixture_ce(instance, path_set, cfg, joint['flows'])
    informed_metrics = evaluate_regime_informed_ce(instance, path_set, cfg, lib, informed['rows'])
    vri_fixed = float(fixed['objective'] - informed['objective'])
    vri_joint = float(joint['objective'] - informed['objective'])
    rows: list[dict[str, object]] = []
    for model_name, eps, result, metrics, vri in [('Empirical mixture SP', np.nan, None, empirical_metrics, np.nan), ('Fixed-frequency conditional DRO', np.nan, fixed, fixed_metrics, vri_fixed), ('Joint-frequency regime DRO', lib.epsilon_p, joint, joint_metrics, vri_joint)]:
        if result is None:
            p_star = lib.p0.copy()
            flows = empirical.flows
            robust_obj = np.nan
            worst_regime = '--'
            active_count = int(np.sum(flows > 1e-06))
            iterations = empirical.iterations
            residual = empirical.residual
            cpu_seconds = empirical.cpu_seconds
        else:
            p_star = result['p_star']
            flows = result['flows']
            robust_obj = result['objective']
            worst_regime = result['worst_regime']
            active_count = result['active_count']
            iterations = result['iterations']
            residual = result['residual']
            cpu_seconds = result['cpu_seconds']
        row = {'Model': model_name, 'epsilon_p': eps, 'p_star_Normal': p_star[0], 'p_star_Localized': p_star[1], 'p_star_Flood': p_star[2], 'P1': flows[0] / cfg.q, 'P2': flows[1] / cfg.q, 'P3': flows[2] / cfg.q, 'ActiveCount': active_count, 'WorstRegime': worst_regime, 'RobustObjective': robust_obj, 'MeanTest': metrics['MeanTest'], 'CVaRTest': metrics['CVaRTest'], 'CETest': metrics['CETest'], 'VRI_robust_obj': vri, 'Iterations': iterations, 'Residual': residual, 'CPUSeconds': cpu_seconds}
        if result is not None:
            for w_idx, regime in enumerate(lib.regimes):
                row[f'phi_{regime}'] = result['phi'][w_idx]
                row[f'kappa_{regime}'] = result['kappa'][w_idx]
        rows.append(row)
    rows.append({'Model': 'Regime-informed benchmark', 'epsilon_p': np.nan, 'p_star_Normal': lib.p0[0], 'p_star_Localized': lib.p0[1], 'p_star_Flood': lib.p0[2], 'P1': np.nan, 'P2': np.nan, 'P3': np.nan, 'ActiveCount': int(informed['rows']['ActiveCount'].sum()), 'WorstRegime': 'decomposed', 'RobustObjective': informed['objective'], 'MeanTest': informed_metrics['MeanTest'], 'CVaRTest': informed_metrics['CVaRTest'], 'CETest': informed_metrics['CETest'], 'VRI_robust_obj': np.nan, 'Iterations': int(informed['rows']['Iterations'].sum()), 'Residual': np.nan, 'CPUSeconds': float(informed['rows']['CPUSeconds'].sum())})
    summary = pd.DataFrame(rows)
    informed_rows = informed['rows']
    summary.to_csv(output_dir / 'braess_regime_conditioned_summary.csv', index=False)
    informed_rows.to_csv(output_dir / 'braess_regime_informed_by_regime.csv', index=False)
    scenario_rows: list[dict[str, object]] = []
    for s_idx, scenario in enumerate(instance.scenario_names):
        regime = next((w for w in lib.regimes if s_idx in set(lib.scenario_indices[w].tolist())))
        local_idx = list(lib.scenario_indices[regime]).index(s_idx)
        scenario_rows.append({'Scenario': scenario, 'Regime': regime, 'p0_regime': lib.p0[list(lib.regimes).index(regime)], 'conditional_probability': lib.conditional_probabilities[regime][local_idx], 'unconditional_probability': instance.probabilities[s_idx], 'rho_w': lib.rho[regime], 'xi1': lib.scenario_vectors[s_idx, 0], 'xi2': lib.scenario_vectors[s_idx, 1], 'xi3': lib.scenario_vectors[s_idx, 2]})
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(output_dir / 'braess_regime_scenario_library.csv', index=False)
    scenario_time_functions_to_dataframe(instance).to_csv(output_dir / 'braess_regime_scenario_time_functions.csv', index=False)
    plot_regime_path_shares(summary, informed_rows, output_dir / 'braess_regime_path_shares.png')
    readme_addition = '\nRegime-conditioned ambiguity outputs:\n- `braess_regime_scenario_library.csv`: regime probabilities, conditional probabilities, radii, and standardized support coordinates.\n- `braess_regime_conditioned_summary.csv`: empirical mixture SP, fixed-frequency conditional DRO, joint-frequency regime DRO, and regime-informed benchmark.\n- `braess_regime_informed_by_regime.csv`: separate regime-wise path shares and objectives.\n- `braess_regime_path_shares.png`: comparison of ex ante and regime-informed path shares.'
    with (output_dir / 'README.md').open('a', encoding='utf-8') as fh:
        fh.write(readme_addition)
    return {'summary_csv': str(output_dir / 'braess_regime_conditioned_summary.csv'), 'informed_csv': str(output_dir / 'braess_regime_informed_by_regime.csv'), 'scenario_csv': str(output_dir / 'braess_regime_scenario_library.csv'), 'figure': str(output_dir / 'braess_regime_path_shares.png')}
ALPHA_LIMIT_EPS = 1e-06
REQUESTED_ALPHA_GRID = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
REQUESTED_ETA_GRID = [round(float(x), 1) for x in np.arange(0.0, 1.01, 0.1)]

def alpha_for_computation(alpha_requested: float) -> float:
    alpha_requested = float(alpha_requested)
    if alpha_requested >= 1.0:
        return 1.0
    if not 0.0 < alpha_requested < 1.0:
        raise ValueError(f'alpha must lie in (0,1] for the requested grid; got {alpha_requested}')
    return alpha_requested

def build_direct_eta_wide_cases(alphas: Sequence[float] | None=None, etas: Sequence[float] | None=None) -> list[dict[str, float]]:
    if alphas is None:
        alphas = REQUESTED_ALPHA_GRID
    if etas is None:
        etas = REQUESTED_ETA_GRID
    cases: list[dict[str, float]] = []
    for alpha_requested in alphas:
        alpha_used = alpha_for_computation(float(alpha_requested))
        for eta in etas:
            cases.append({'alpha': float(alpha_requested), 'alpha_requested': float(alpha_requested), 'alpha_display': float(alpha_requested), 'alpha_used': float(alpha_used), 'alpha_is_limit': float(alpha_requested) >= 1.0, 'eta': float(eta)})
    return cases

def _one_hot_tail_law(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v, dtype=float)
    out[int(np.argmax(v))] = 1.0
    return out

def _tail_law_string_from_values(scenario_names: Sequence[str], values: Sequence[float]) -> str:
    q_tail = _one_hot_tail_law(values)
    return _tail_set_string(scenario_names, q_tail)

def model_a_limit_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    pt = path_times_by_scenario(instance, path_set, path_flows, cfg)
    mean = instance.probabilities @ pt
    maxv = np.max(pt, axis=0)
    costs = (1.0 - cfg.eta) * mean + cfg.eta * maxv
    tail_laws = np.vstack([_one_hot_tail_law(pt[:, k]) for k in range(path_set.n_paths)])
    return (costs, tail_laws)

def model_b_limit_raw_costs(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = potentials_by_scenario(instance, path_set, path_flows, cfg)
    tail_law = _one_hot_tail_law(vals)
    tilted = (1.0 - cfg.eta) * instance.probabilities + cfg.eta * tail_law
    costs = tilted @ path_times_by_scenario(instance, path_set, path_flows, cfg)
    return (np.asarray(costs), tail_law, tilted)

def system_risk_metrics_limit(instance: BraessInstance, path_set: PathSet, flows: Sequence[float], eta: float, cfg: SolverConfig) -> dict[str, float | str]:
    vals = potentials_by_scenario(instance, path_set, flows, cfg)
    mean_val = float(np.dot(instance.probabilities, vals))
    max_val = float(np.max(vals))
    ce_val = (1.0 - eta) * mean_val + eta * max_val
    ent_val = entropy(flows, cfg) / cfg.theta
    return {'MeanPotential': mean_val, 'CVaRPotential': max_val, 'CEPotential': ce_val, 'EntropyOverTheta': ent_val, 'ModelBObjectiveAtFlow': ce_val + ent_val, 'SystemTailSelector': _tail_law_string_from_values(instance.scenario_names, vals)}

def vi_gap_model_a_limit(instance: BraessInstance, path_set: PathSet, path_flows: Sequence[float], cfg: SolverConfig) -> float:
    f = np.asarray(path_flows, dtype=float)
    raw = model_a_limit_raw_costs(instance, path_set, f, cfg)[0]
    mapping = raw + 1.0 / cfg.theta * np.log1p(f / cfg.r0)
    gap = float(np.dot(mapping, f) - cfg.q * np.min(mapping))
    return gap / (1.0 + abs(float(np.dot(mapping, f))))

def solve_model_a_limit_fast(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, eta: float, initial_flows: Sequence[float] | None=None) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=1.0 - ALPHA_LIMIT_EPS, eta=float(eta), bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    if initial_flows is None:
        try:
            initial_flows = solve_model_b_limit_for_params(instance, path_set, cfg_base, eta).flows
        except Exception:
            initial_flows = np.full(path_set.n_paths, cfg.q / path_set.n_paths, dtype=float)
    z0 = _simplex_flows_to_z(initial_flows, cfg.q)

    def residual_z(z: np.ndarray) -> np.ndarray:
        f = _z_to_simplex_flows(z, cfg.q)
        costs = model_a_limit_raw_costs(instance, path_set, f, cfg)[0]
        target, _, _ = truncated_flows_from_costs(costs, cfg)
        return (target - f) / cfg.q
    start = time.perf_counter()
    res = opt.least_squares(residual_z, z0, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=600)
    flows = _z_to_simplex_flows(res.x, cfg.q)
    raw = model_a_limit_raw_costs(instance, path_set, flows, cfg)[0]
    flows, probs, pi = truncated_flows_from_costs(raw, cfg)
    raw = model_a_limit_raw_costs(instance, path_set, flows, cfg)[0]
    flows, probs, pi = truncated_flows_from_costs(raw, cfg)
    elapsed = time.perf_counter() - start
    active = tuple((path_set.path_names[i] for i, val in enumerate(flows) if val > 1e-06))
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model A alpha-limit TSUE' if eta not in (0.0, 1.0) else 'Model A expectation TSUE' if eta == 0.0 else 'Model A max-tail TSUE', overlap_model='none', beta_ps=0.0, flows=flows, perceived_costs=raw, raw_costs=raw, overlap_penalty=np.zeros_like(raw), probabilities=probs, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, flows), iterations=int(res.nfev), residual=float(np.max(np.abs(residual_z(res.x))) * cfg.q), active_paths=active, cpu_seconds=elapsed)

def solve_model_b_limit_for_params(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=1.0 - ALPHA_LIMIT_EPS, eta=float(eta), bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    n = path_set.n_paths
    deterministic_penalty = overlap_penalty(instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)

    def objective_f(f_raw: np.ndarray) -> float:
        f = np.asarray(f_raw, dtype=float)
        if np.any(f < -1e-08):
            return 1e+30
        f = np.maximum(f, 0.0)
        if f.sum() <= 0:
            return 1e+30
        f = cfg.q * f / f.sum()
        vals = potentials_by_scenario(instance, path_set, f, cfg)
        ce = (1.0 - eta) * float(np.dot(instance.probabilities, vals)) + eta * float(np.max(vals))
        return float(ce + entropy(f, cfg) / cfg.theta + np.dot(deterministic_penalty, f))
    constraints = ({'type': 'eq', 'fun': lambda f: float(np.sum(f) - cfg.q)},)
    bounds = opt.Bounds(np.zeros(n), np.full(n, cfg.q))
    starts = [np.full(n, cfg.q / n, dtype=float)]
    try:
        rn = solve_fixed_point(instance, path_set, 'RN alpha-limit start', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg), cfg, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2, tol=1e-08, max_iter=20000).flows
        starts.append(rn)
    except Exception:
        pass
    for k in range(n):
        e = np.zeros(n, dtype=float)
        e[k] = cfg.q
        starts.append(e)
    starts += [np.array([0.5, 0.5, 0.0]) * cfg.q, np.array([0.5, 0.0, 0.5]) * cfg.q, np.array([0.0, 0.5, 0.5]) * cfg.q]
    best: opt.OptimizeResult | None = None
    start_time = time.perf_counter()
    for x0 in starts:
        res = opt.minimize(objective_f, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-11, 'maxiter': 3000, 'disp': False})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('Model B alpha-limit solve failed.')
    f = np.maximum(np.asarray(best.x, dtype=float), 0.0)
    if f.sum() <= 0.0:
        f[:] = cfg.q / n
    else:
        f *= cfg.q / f.sum()
    raw = model_b_limit_raw_costs(instance, path_set, f, cfg)[0]
    perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    _, _, pi = truncated_flows_from_costs(perceived, cfg)
    elapsed = time.perf_counter() - start_time
    return EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model B alpha-limit TSUE' if eta not in (0.0, 1.0) else 'Model B expectation TSUE' if eta == 0.0 else 'Model B max-tail TSUE', overlap_model=overlap_model, beta_ps=beta_ps, flows=f, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=f / cfg.q, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, f), iterations=int(getattr(best, 'nit', -1)), residual=float('nan'), active_paths=tuple((path_set.path_names[i] for i, val in enumerate(f) if val > 1e-06)), cpu_seconds=elapsed)

def tail_divergence_limit_at_fB(instance: BraessInstance, path_set: PathSet, f_b: Sequence[float], cfg: SolverConfig) -> dict[str, object]:
    pt = path_times_by_scenario(instance, path_set, f_b, cfg)
    vals = potentials_by_scenario(instance, path_set, f_b, cfg)
    tail_a = np.vstack([_one_hot_tail_law(pt[:, k]) for k in range(path_set.n_paths)])
    tail_b = _one_hot_tail_law(vals)
    tilts_a = (1.0 - cfg.eta) * instance.probabilities[None, :] + cfg.eta * tail_a
    tilt_b = (1.0 - cfg.eta) * instance.probabilities + cfg.eta * tail_b
    d_tail_by_path = np.asarray([np.sum(np.abs(tail_a[k] - tail_b)) for k in range(path_set.n_paths)], dtype=float)
    d_p_by_path = np.asarray([np.sum(np.abs(tilts_a[k] - tilt_b)) for k in range(path_set.n_paths)], dtype=float)
    worst_idx = int(np.argmax(d_tail_by_path))
    out: dict[str, object] = {'D_tail_AB': float(np.max(d_tail_by_path)), 'Mean_D_tail_AB': float(np.mean(d_tail_by_path)), 'D_p_AB': float(np.max(d_p_by_path)), 'Mean_D_p_AB': float(np.mean(d_p_by_path)), 'TailTV_max_AB': float(0.5 * np.max(d_p_by_path)), 'WorstTailPath': f'P{worst_idx + 1}', 'TailSet_B_at_fB': _tail_set_string(instance.scenario_names, tail_b)}
    for k in range(path_set.n_paths):
        out[f'D_tail_P{k + 1}'] = float(d_tail_by_path[k])
        out[f'D_p_P{k + 1}'] = float(d_p_by_path[k])
        out[f'TailSet_A_P{k + 1}_at_fB'] = _tail_set_string(instance.scenario_names, tail_a[k])
    return out

def _solve_model_b_direct_objective_fallback(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> tuple[np.ndarray, opt.OptimizeResult]:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    n = path_set.n_paths
    deterministic_penalty = overlap_penalty(instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)

    def objective_f(f_raw: np.ndarray) -> float:
        f = np.asarray(f_raw, dtype=float)
        if np.any(f < -1e-07):
            return 1e+30
        f = np.maximum(f, 0.0)
        total = f.sum()
        if total <= 0.0:
            return 1e+30
        f = cfg.q * f / total
        vals = potentials_by_scenario(instance, path_set, f, cfg)
        ce, _, _ = mean_cvar_value_and_tilt(vals, instance.probabilities, alpha, eta)
        return float(ce + entropy(f, cfg) / cfg.theta + np.dot(deterministic_penalty, f))
    constraints = ({'type': 'eq', 'fun': lambda f: float(np.sum(f) - cfg.q)},)
    bounds = opt.Bounds(np.zeros(n), np.full(n, cfg.q))
    starts: list[np.ndarray] = [np.full(n, cfg.q / n, dtype=float)]
    try:
        rn = solve_fixed_point(instance, path_set, 'RN fallback start', lambda f: risk_neutral_raw_costs(instance, path_set, f, cfg), cfg, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.2, tol=1e-08, max_iter=20000).flows
        starts.append(rn)
    except Exception:
        pass
    for k in range(n):
        e = np.zeros(n, dtype=float)
        e[k] = cfg.q
        starts.append(e)
    starts += [np.array([0.5, 0.5, 0.0]) * cfg.q, np.array([0.5, 0.0, 0.5]) * cfg.q, np.array([0.0, 0.5, 0.5]) * cfg.q]
    best: opt.OptimizeResult | None = None
    for x0 in starts:
        res = opt.minimize(objective_f, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-10, 'maxiter': 2500, 'disp': False})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('Fallback Model B direct objective solver did not run.')
    f = np.maximum(np.asarray(best.x, dtype=float), 0.0)
    if f.sum() <= 0.0:
        f[:] = cfg.q / n
    else:
        f *= cfg.q / f.sum()
    best.x = f
    return (f, best)

def solve_model_b_for_params(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> EquilibriumResult:
    cfg = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    start = time.perf_counter()
    used_fallback = False
    try:
        f, res = solve_model_b_convex_sp_direct_eta(instance, path_set, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
    except Exception:
        used_fallback = True
        f, res = _solve_model_b_direct_objective_fallback(instance, path_set, cfg_base, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
    raw = model_b_raw_costs(instance, path_set, f, cfg)[0]
    perceived, penalty = add_overlap(raw, instance, path_set, cfg, beta_ps=beta_ps, overlap_model=overlap_model)
    _, _, pi = truncated_flows_from_costs(perceived, cfg)
    elapsed = time.perf_counter() - start
    result = EquilibriumResult(instance=instance.name, path_set=path_set.name, model='Model B common-state TSUE', overlap_model=overlap_model, beta_ps=beta_ps, flows=f, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=f / cfg.q, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, f), iterations=int(getattr(res, 'nit', -1)), residual=float('nan') if not used_fallback else float(getattr(res, 'optimality', np.nan)), active_paths=tuple((path_set.path_names[i] for i, val in enumerate(f) if val > 1e-06)), cpu_seconds=elapsed)
    return result

def run_param_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, cases: Sequence[dict[str, float]], parameterization: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    last_a: dict[tuple[float, float], np.ndarray] = {}
    case_columns = set().union(*(case.keys() for case in cases)) if cases else set()
    for case in cases:
        alpha_requested = float(case.get('alpha_requested', case.get('alpha', case.get('alpha_display', np.nan))))
        alpha_display = float(case.get('alpha_display', alpha_requested))
        alpha_used = float(case.get('alpha_used', alpha_for_computation(alpha_requested)))
        alpha_is_limit = bool(case.get('alpha_is_limit', alpha_requested >= 1.0))
        eta = float(case['eta'])
        lam = case.get('lambda', np.nan)
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=1.0 - ALPHA_LIMIT_EPS if alpha_is_limit else alpha_used, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        if alpha_is_limit:
            res_b = solve_model_b_limit_for_params(instance, path_set, cfg, eta)
            metrics_b = system_risk_metrics_limit(instance, path_set, res_b.flows, eta, cfg_case)
        else:
            res_b = solve_model_b_for_params_labeled(instance, path_set, cfg, alpha_used, eta)
            metrics_b = system_risk_metrics(instance, path_set, res_b.flows, alpha_used, eta, cfg_case)
        row_b: dict[str, object] = {'Parameterization': parameterization, 'Instance': instance.name, 'PathSet': path_set.name, 'Model': 'Model B', 'RiskCase': classify_risk_case(eta), 'alpha': alpha_display, 'alpha_requested': alpha_requested, 'alpha_used': alpha_used, 'alpha_is_limit': alpha_is_limit, 'lambda': lam, 'eta': eta, 'tail_hinge_coeff': np.inf if alpha_is_limit and eta > 0 else eta / (1.0 - alpha_used) if eta > 0 else 0.0, 'pi_od': res_b.pi_od, 'ActiveCount': len(res_b.active_paths), 'Iterations': res_b.iterations, 'Residual': res_b.residual, 'CPUSeconds': res_b.cpu_seconds}
        row_b.update(metrics_b)
        for k, name in enumerate(path_set.path_names):
            row_b[f'Path{k + 1}'] = name
            row_b[f'f{k + 1}'] = res_b.flows[k]
            row_b[f'P{k + 1}'] = res_b.probabilities[k]
            row_b[f'g{k + 1}'] = res_b.raw_costs[k]
        rows.append(row_b)
        init = last_a.get((alpha_used, eta), res_b.flows)
        if alpha_is_limit:
            res_a = solve_model_a_limit_fast(instance, path_set, cfg, eta, initial_flows=init)
            metrics_a = system_risk_metrics_limit(instance, path_set, res_a.flows, eta, cfg_case)
        else:
            res_a = solve_model_a_fast(instance, path_set, cfg, alpha_used, eta, initial_flows=init)
            metrics_a = system_risk_metrics(instance, path_set, res_a.flows, alpha_used, eta, cfg_case)
        last_a[alpha_used, eta] = res_a.flows
        row_a: dict[str, object] = {'Parameterization': parameterization, 'Instance': instance.name, 'PathSet': path_set.name, 'Model': 'Model A', 'RiskCase': classify_risk_case(eta), 'alpha': alpha_display, 'alpha_requested': alpha_requested, 'alpha_used': alpha_used, 'alpha_is_limit': alpha_is_limit, 'lambda': lam, 'eta': eta, 'tail_hinge_coeff': np.inf if alpha_is_limit and eta > 0 else eta / (1.0 - alpha_used) if eta > 0 else 0.0, 'pi_od': res_a.pi_od, 'ActiveCount': len(res_a.active_paths), 'Iterations': res_a.iterations, 'Residual': res_a.residual, 'CPUSeconds': res_a.cpu_seconds, 'Gap_A': vi_gap_model_a(instance, path_set, res_a.flows, cfg_case)}
        row_a.update(metrics_a)
        for k, name in enumerate(path_set.path_names):
            row_a[f'Path{k + 1}'] = name
            row_a[f'f{k + 1}'] = res_a.flows[k]
            row_a[f'P{k + 1}'] = res_a.probabilities[k]
            row_a[f'g{k + 1}'] = res_a.raw_costs[k]
        rows.append(row_a)
    sort_cols = ['alpha', 'lambda', 'eta', 'Model'] if 'lambda' in case_columns else ['alpha', 'eta', 'Model']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def diagnostics_from_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, grid_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in grid_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub_a = sub[sub['Model'] == 'Model A']
        sub_b = sub[sub['Model'] == 'Model B']
        if sub_a.empty or sub_b.empty:
            continue
        alpha_display = float(sub_b.iloc[0]['alpha'])
        alpha_used = float(sub_b.iloc[0]['alpha_used']) if 'alpha_used' in sub_b.columns else alpha_display
        alpha_requested = float(sub_b.iloc[0]['alpha_requested']) if 'alpha_requested' in sub_b.columns else alpha_display
        alpha_is_limit = bool(sub_b.iloc[0]['alpha_is_limit']) if 'alpha_is_limit' in sub_b.columns else False
        eta = float(sub_b.iloc[0]['eta'])
        f_a = np.array([float(sub_a.iloc[0][f'f{k + 1}']) for k in range(path_set.n_paths)])
        f_b = np.array([float(sub_b.iloc[0][f'f{k + 1}']) for k in range(path_set.n_paths)])
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=1.0 - ALPHA_LIMIT_EPS if alpha_is_limit else alpha_used, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        row: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Parameterization': str(sub_b.iloc[0]['Parameterization']), 'alpha': alpha_display, 'alpha_requested': alpha_requested, 'alpha_used': alpha_used, 'alpha_is_limit': alpha_is_limit, 'eta': eta, 'RiskCase': classify_risk_case(eta), 'D_f_AB': float(np.sum(np.abs(f_a - f_b)) / cfg.q), 'Gap_A_at_fB': vi_gap_model_a(instance, path_set, f_b, cfg_case), 'Active_A': int(sub_a.iloc[0]['ActiveCount']), 'Active_B': int(sub_b.iloc[0]['ActiveCount']), 'Active_A_minus_Active_B': int(sub_a.iloc[0]['ActiveCount'] - sub_b.iloc[0]['ActiveCount'])}
        if 'lambda' in grid_df.columns:
            row['lambda'] = float(sub_b.iloc[0]['lambda']) if not pd.isna(sub_b.iloc[0]['lambda']) else np.nan
        row.update(tail_divergence_at_fB(instance, path_set, f_b, cfg_case))
        rows.append(row)
    sort_cols = ['alpha', 'lambda', 'eta'] if 'lambda' in grid_df.columns else ['alpha', 'eta']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def modelb_base_table_from_grid(grid_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    sub = grid_df[grid_df['Model'] == 'Model B'].copy()
    keep = ['RiskCase', 'alpha']
    for col in ['alpha_requested', 'alpha_used', 'alpha_is_limit']:
        if col in sub.columns:
            keep.append(col)
    keep += (['lambda'] if include_lambda else []) + ['eta', 'tail_hinge_coeff', 'P1', 'P2', 'P3', 'pi_od', 'ActiveCount', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    return sub[keep].sort_values(['alpha'] + (['lambda'] if include_lambda else ['eta'])).reset_index(drop=True)

def compact_modelb_ab_table(modelb_df: pd.DataFrame, diag_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    key_cols = ['alpha', 'lambda', 'eta'] if include_lambda else ['alpha', 'eta']
    diag_cols = key_cols + ['Instance', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    diag_cols = [c for c in diag_cols if c in diag_df.columns]
    return modelb_df.merge(diag_df[diag_cols], on=key_cols, how='left').sort_values(key_cols).reset_index(drop=True)

def write_ab_plain_text_table(df: pd.DataFrame, output_path: Path) -> None:
    cols = ['RiskCase', 'alpha', 'alpha_used', 'eta', 'P1', 'P2', 'P3', 'pi_od', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for c in ['alpha', 'alpha_used', 'eta', 'P1', 'P2', 'P3', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: '--' if pd.isna(v) else f'{float(v):.6g}')
    if 'pi_od' in out.columns:
        out['pi_od'] = out['pi_od'].map(lambda v: '--' if pd.isna(v) else f'{float(v):.4f}')
    if 'Active_A_minus_Active_B' in out.columns:
        out['Active_A_minus_Active_B'] = out['Active_A_minus_Active_B'].astype(int)
    header = f'Full direct-eta Model B table with Model A/B diagnostics\nRequested alpha grid: {REQUESTED_ALPHA_GRID}\nEta grid: {REQUESTED_ETA_GRID}\nNote: alpha=1.00 rows are computed as the alpha-up-to-one max-tail limiting case, because CVaR requires alpha<1.\n\n'
    output_path.write_text(header + out.to_string(index=False), encoding='utf-8')

def run_all_v16(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir)
    cfg = SolverConfig()
    base_paths = base_path_set()
    route_local = build_route_local_instance()
    direct_cases = build_direct_eta_wide_cases()
    direct_grid = run_param_grid(route_local, base_paths, cfg, direct_cases, 'direct-eta')
    direct_grid.to_csv(output_dir / 'braess_direct_eta_risk_sensitivity_all_models.csv', index=False)
    direct_grid.to_csv(output_dir / 'braess_direct_eta_wide_alpha_all_models.csv', index=False)
    direct_diag = diagnostics_from_grid(route_local, base_paths, cfg, direct_grid, ['alpha', 'eta'])
    direct_diag.to_csv(output_dir / 'braess_modelA_modelB_eta_grid_diagnostics.csv', index=False)
    direct_diag.to_csv(output_dir / 'braess_modelA_modelB_wide_alpha_eta_grid_diagnostics.csv', index=False)
    direct_modelb = compact_modelb_ab_table(modelb_base_table_from_grid(direct_grid, include_lambda=False), direct_diag, include_lambda=False)
    direct_modelb.to_csv(output_dir / 'braess_modelB_direct_eta_with_AB_diagnostics.csv', index=False)
    direct_modelb.to_csv(output_dir / 'braess_modelB_direct_eta_risk_table.csv', index=False)
    direct_modelb.to_csv(output_dir / 'braess_modelB_direct_eta_wide_alpha_with_AB_diagnostics.csv', index=False)
    write_ab_plain_text_table(direct_modelb, output_dir / 'braess_modelB_direct_eta_wide_alpha_print_table.txt')
    lambda_cases = []
    for alpha in [0.9, 0.95]:
        for lam in [round(float(x), 1) for x in np.arange(0.0, 0.91, 0.1)]:
            if lam <= alpha + 1e-12:
                lambda_cases.append({'alpha': alpha, 'lambda': lam, 'eta': eta_from_lambda(alpha, lam)})
    lambda_grid = run_param_grid(route_local, base_paths, cfg, lambda_cases, 'lambda-mapped')
    lambda_grid.to_csv(output_dir / 'braess_lambda_mapped_risk_sensitivity_all_models.csv', index=False)
    lambda_diag = diagnostics_from_grid(route_local, base_paths, cfg, lambda_grid, ['alpha', 'lambda', 'eta'])
    lambda_diag.to_csv(output_dir / 'braess_modelA_modelB_lambda_grid_diagnostics.csv', index=False)
    lambda_modelb = compact_modelb_ab_table(modelb_base_table_from_grid(lambda_grid, include_lambda=True), lambda_diag, include_lambda=True)
    lambda_modelb.to_csv(output_dir / 'braess_modelB_lambda_mapped_with_AB_diagnostics.csv', index=False)
    solve_all_dro_decisions(output_dir)
    print('[v16] regime-conditioned ambiguity experiment', flush=True)
    run_braess_regime_experiment(output_dir)
    with (output_dir / 'README.md').open('a', encoding='utf-8') as fh:
        pass

def run_quick_notebook_smoke_test() -> pd.DataFrame:
    cfg = SolverConfig()
    path_set = base_path_set()
    path_set.validate()
    instance = build_route_local_instance()
    validate_instance(instance)
    results = solve_core_models(instance, path_set, cfg)
    return results_to_dataframe(results.values())

def list_output_files(output: str | Path, max_files: int=80) -> pd.DataFrame:
    output_dir = Path(output).expanduser().resolve()
    rows: list[dict[str, object]] = []
    if not output_dir.exists():
        return pd.DataFrame(columns=['file', 'size_kb'])
    for path in sorted(output_dir.rglob('*')):
        if path.is_file():
            rows.append({'file': str(path.relative_to(output_dir)), 'size_kb': round(path.stat().st_size / 1024.0, 1)})
    return pd.DataFrame(rows).head(max_files)

def show_output_links(output: str | Path, make_zip: bool=True, zip_path: str | Path | None=None) -> Path | None:
    output_dir = Path(output).expanduser().resolve()
    if not output_dir.exists():
        print(f'Output directory does not exist yet: {output_dir}')
        return None
    print(f'Output directory: {output_dir}')
    files_df = list_output_files(output_dir)
    if files_df.empty:
        print('No files found in the output directory yet.')
    else:
        try:
            from IPython.display import display
            display(files_df)
        except Exception:
            print(files_df.to_string(index=False))
    final_zip_path: Path | None = None
    if make_zip:
        final_zip_path = Path(zip_path).expanduser().resolve() if zip_path is not None else output_dir.with_suffix('.zip')
        zip_outputs(output_dir, final_zip_path)
        print(f'Zip package: {final_zip_path}')
    try:
        from IPython.display import display, FileLink, FileLinks
        if final_zip_path is not None and final_zip_path.exists():
            display(FileLink(str(final_zip_path)))
        display(FileLinks(str(output_dir)))
    except Exception:
        if final_zip_path is not None:
            print(f'Download zip from: {final_zip_path}')
    return final_zip_path

def run_base_notebook(output: str | Path='braess_integrated_results_base', make_zip: bool=True, show_links: bool=True) -> Path:
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(output_dir)
    print(f'Saved base outputs to: {output_dir}')
    if show_links:
        show_output_links(output_dir, make_zip=make_zip)
    elif make_zip:
        zip_outputs(output_dir, output_dir.with_suffix('.zip'))
    return output_dir

def run_full_v16_notebook(output: str | Path='braess_integrated_results_with_regime', make_zip: bool=True, show_links: bool=True) -> Path:
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all_v16(output_dir)
    print(f'Saved standalone v16 outputs to: {output_dir}')
    if show_links:
        show_output_links(output_dir, make_zip=make_zip)
    elif make_zip:
        zip_outputs(output_dir, output_dir.with_suffix('.zip'))
    return output_dir

def run_notebook(output: str | Path='braess_integrated_results_with_regime', make_zip: bool=True, show_links: bool=True) -> Path:
    return run_full_v16_notebook(output=output, make_zip=make_zip, show_links=show_links)

def make_output_zip(output_dir: str | Path, zip_path: str | Path | None=None) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f'Output directory does not exist: {output_dir}')
    if zip_path is None:
        zip_path = output_dir.with_suffix('.zip')
    zip_path = Path(zip_path).expanduser().resolve()
    zip_outputs(output_dir, zip_path)
    return zip_path

def show_output_files(output_dir: str | Path, make_zip: bool=True, max_files: int=80) -> pd.DataFrame:
    output_dir = Path(output_dir).expanduser().resolve()
    print(f'Current working directory: {Path.cwd().resolve()}')
    print(f'Output directory: {output_dir}')
    if not output_dir.exists():
        print('Output directory does not exist yet. Run run_base_notebook(...) or run_full_v16_notebook(...) first.')
        return pd.DataFrame(columns=['relative_path', 'size_bytes'])
    files = sorted((p for p in output_dir.rglob('*') if p.is_file()))
    rows = [{'relative_path': str(p.relative_to(output_dir)), 'size_bytes': p.stat().st_size} for p in files]
    df = pd.DataFrame(rows)
    print(f'Generated file count: {len(files)}')
    zip_path: Path | None = None
    if make_zip:
        zip_path = make_output_zip(output_dir)
        print(f'Zip archive: {zip_path}')
    try:
        from IPython.display import display, FileLink, FileLinks
        if not df.empty:
            display(df.head(max_files))
        display(FileLinks(str(output_dir)))
        if zip_path is not None and zip_path.exists():
            display(FileLink(str(zip_path)))
    except Exception:
        if not df.empty:
            print(df.head(max_files).to_string(index=False))
        if zip_path is not None:
            print(f'Open or download this zip from your Jupyter file browser: {zip_path}')
    return df

def run_base_notebook_and_show(output: str | Path='braess_integrated_results_base', make_zip: bool=True) -> Path:
    output_dir = run_base_notebook(output=output, make_zip=False)
    show_output_files(output_dir, make_zip=make_zip)
    return output_dir

def run_full_v16_notebook_and_show(output: str | Path='braess_integrated_results_with_regime', make_zip: bool=True) -> Path:
    output_dir = run_full_v16_notebook(output=output, make_zip=False)
    show_output_files(output_dir, make_zip=make_zip)
    return output_dir

def main() -> None:
    parser = argparse.ArgumentParser(description='Standalone Braess TSUE/DRO experiment; no local version imports required.')
    parser.add_argument('--output', type=Path, default=Path('braess_integrated_results_with_regime'), help='Output directory.')
    parser.add_argument('--zip', type=Path, default=None, help='Optional zip path for output directory and this script.')
    args = parser.parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all_v16(output_dir)
    if args.zip is not None:
        zip_path = args.zip.expanduser().resolve()
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(output_dir.rglob('*')):
                if path.is_file():
                    zf.write(path, path.relative_to(output_dir.parent))
            if '__file__' in globals():
                zf.write(Path(__file__).resolve(), Path(__file__).name)
        print(f'Saved zip package to: {zip_path}')
    print(f'Saved standalone v16 outputs to: {output_dir}')

try:
    from IPython.display import FileLink, FileLinks, Image, display
except ImportError:
    def display(value):
        print(value)
    def Image(filename=None, **kwargs):
        return filename
    def FileLink(path):
        return path
    def FileLinks(path):
        return path

def diagnostics_from_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, grid_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in grid_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub_a = sub[sub['Model'] == 'Model A']
        sub_b = sub[sub['Model'] == 'Model B']
        if sub_a.empty or sub_b.empty:
            continue
        row_a0 = sub_a.iloc[0]
        row_b0 = sub_b.iloc[0]
        alpha_display = float(row_b0['alpha'])
        alpha_used = float(row_b0['alpha_used']) if 'alpha_used' in sub_b.columns else alpha_display
        alpha_requested = float(row_b0['alpha_requested']) if 'alpha_requested' in sub_b.columns else alpha_display
        alpha_is_limit = bool(row_b0['alpha_is_limit']) if 'alpha_is_limit' in sub_b.columns else False
        eta = float(row_b0['eta'])
        f_a = np.array([float(row_a0[f'f{k + 1}']) for k in range(path_set.n_paths)])
        f_b = np.array([float(row_b0[f'f{k + 1}']) for k in range(path_set.n_paths)])
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=1.0 - ALPHA_LIMIT_EPS if alpha_is_limit else alpha_used, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        row: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Parameterization': str(row_b0['Parameterization']), 'alpha': alpha_display, 'alpha_requested': alpha_requested, 'alpha_used': alpha_used, 'alpha_is_limit': alpha_is_limit, 'eta': eta, 'RiskCase': classify_risk_case(eta), 'pi_A': float(row_a0['pi_od']), 'pi_B': float(row_b0['pi_od']), 'D_f_AB': float(np.sum(np.abs(f_a - f_b)) / cfg.q), 'Gap_A_at_fB': vi_gap_model_a(instance, path_set, f_b, cfg_case), 'Active_A': int(row_a0['ActiveCount']), 'Active_B': int(row_b0['ActiveCount']), 'Active_A_minus_Active_B': int(row_a0['ActiveCount'] - row_b0['ActiveCount'])}
        if 'lambda' in grid_df.columns:
            row['lambda'] = float(row_b0['lambda']) if not pd.isna(row_b0['lambda']) else np.nan
        row.update(tail_divergence_at_fB(instance, path_set, f_b, cfg_case))
        rows.append(row)
    sort_cols = ['alpha', 'lambda', 'eta'] if 'lambda' in grid_df.columns else ['alpha', 'eta']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def modelb_base_table_from_grid(grid_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    sub = grid_df[grid_df['Model'] == 'Model B'].copy()
    sub['pi_B'] = sub['pi_od']
    keep = ['RiskCase', 'alpha']
    for col in ['alpha_requested', 'alpha_used', 'alpha_is_limit']:
        if col in sub.columns:
            keep.append(col)
    keep += (['lambda'] if include_lambda else []) + ['eta', 'tail_hinge_coeff', 'P1', 'P2', 'P3', 'pi_B', 'ActiveCount', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    return sub[keep].sort_values(['alpha'] + (['lambda'] if include_lambda else ['eta'])).reset_index(drop=True)

def compact_modelb_ab_table(modelb_df: pd.DataFrame, diag_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    key_cols = ['alpha', 'lambda', 'eta'] if include_lambda else ['alpha', 'eta']
    diag_cols = key_cols + ['Instance', 'pi_A', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A', 'Active_B', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    diag_cols = [c for c in diag_cols if c in diag_df.columns]
    out = modelb_df.merge(diag_df[diag_cols], on=key_cols, how='left').sort_values(key_cols).reset_index(drop=True)
    out = out.drop(columns=['Instance'], errors='ignore')
    preferred = ['RiskCase', 'alpha', 'alpha_requested', 'alpha_used', 'alpha_is_limit', 'lambda', 'eta', 'tail_hinge_coeff', 'P1', 'P2', 'P3', 'pi_A', 'pi_B', 'ActiveCount', 'Active_A', 'Active_B', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    ordered = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[ordered]

def write_ab_plain_text_table(df: pd.DataFrame, output_path: Path) -> None:
    cols = ['RiskCase', 'alpha', 'alpha_used', 'eta', 'P1', 'P2', 'P3', 'pi_A', 'pi_B', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for c in ['alpha', 'alpha_used', 'eta', 'P1', 'P2', 'P3', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: '--' if pd.isna(v) else f'{float(v):.6g}')
    for c in ['pi_A', 'pi_B']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: '--' if pd.isna(v) else f'{float(v):.4f}')
    if 'Active_A_minus_Active_B' in out.columns:
        out['Active_A_minus_Active_B'] = out['Active_A_minus_Active_B'].astype(int)
    header = f'Full direct-eta Model B table with Model A/B reservation costs and diagnostics\nRequested alpha grid: {REQUESTED_ALPHA_GRID}\nEta grid: {REQUESTED_ETA_GRID}\nNote: alpha=1.00 rows are computed as the alpha-up-to-one max-tail limiting case, because CVaR requires alpha<1.\n\n'
    output_path.write_text(header + out.to_string(index=False), encoding='utf-8')

def add_pi_ab_columns_to_long_grid(grid_df: pd.DataFrame, group_cols: Sequence[str] | None=None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ['alpha', 'eta']
        if 'lambda' in grid_df.columns and grid_df['lambda'].notna().any():
            group_cols = ['alpha', 'lambda', 'eta']
    df = grid_df.copy()
    df['pi_A'] = np.nan
    df['pi_B'] = np.nan
    for _, idx in df.groupby(list(group_cols), dropna=False).groups.items():
        sub = df.loc[idx]
        a = sub.loc[sub['Model'] == 'Model A', 'pi_od']
        b = sub.loc[sub['Model'] == 'Model B', 'pi_od']
        if not a.empty:
            df.loc[idx, 'pi_A'] = float(a.iloc[0])
        if not b.empty:
            df.loc[idx, 'pi_B'] = float(b.iloc[0])
    preferred = ['Parameterization', 'Instance', 'PathSet', 'Model', 'RiskCase', 'alpha', 'alpha_requested', 'alpha_used', 'alpha_is_limit', 'lambda', 'eta', 'tail_hinge_coeff', 'pi_od', 'pi_A', 'pi_B', 'ActiveCount', 'Iterations', 'Residual', 'CPUSeconds']
    ordered = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]
    return df[ordered]
REQUESTED_ALPHA_GRID = [0.8, 0.84, 0.88, 0.92, 0.96]
REQUESTED_ETA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def add_pi_ab_columns_to_long_grid(grid_df: pd.DataFrame, group_cols: Sequence[str] | None=None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ['alpha', 'eta']
        if 'lambda' in grid_df.columns and grid_df['lambda'].notna().any():
            group_cols = ['alpha', 'lambda', 'eta']
    df = grid_df.copy()
    new_cols = ['pi_A', 'pi_B', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q']
    for col in new_cols:
        df[col] = np.nan
    for _, idx in df.groupby(list(group_cols), dropna=False).groups.items():
        sub = df.loc[idx]
        a = sub[sub['Model'] == 'Model A']
        b = sub[sub['Model'] == 'Model B']
        if not a.empty:
            df.loc[idx, 'pi_A'] = float(a['pi_od'].iloc[0])
            for k in range(1, 4):
                if f'P{k}' in a.columns:
                    df.loc[idx, f'f{k}_A_over_q'] = float(a[f'P{k}'].iloc[0])
        if not b.empty:
            df.loc[idx, 'pi_B'] = float(b['pi_od'].iloc[0])
            for k in range(1, 4):
                if f'P{k}' in b.columns:
                    df.loc[idx, f'f{k}_B_over_q'] = float(b[f'P{k}'].iloc[0])
    preferred = ['Parameterization', 'Instance', 'PathSet', 'Model', 'RiskCase', 'alpha', 'alpha_requested', 'alpha_used', 'alpha_is_limit', 'lambda', 'eta', 'tail_hinge_coeff', 'pi_od', 'pi_A', 'pi_B', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'ActiveCount', 'Iterations', 'Residual', 'CPUSeconds']
    ordered = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]
    return df[ordered]

def diagnostics_from_grid(instance: BraessInstance, path_set: PathSet, cfg: SolverConfig, grid_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in grid_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub_a = sub[sub['Model'] == 'Model A']
        sub_b = sub[sub['Model'] == 'Model B']
        if sub_a.empty or sub_b.empty:
            continue
        row_a0 = sub_a.iloc[0]
        row_b0 = sub_b.iloc[0]
        alpha_display = float(row_b0['alpha'])
        alpha_used = float(row_b0['alpha_used']) if 'alpha_used' in sub_b.columns else alpha_display
        alpha_requested = float(row_b0['alpha_requested']) if 'alpha_requested' in sub_b.columns else alpha_display
        alpha_is_limit = bool(row_b0['alpha_is_limit']) if 'alpha_is_limit' in sub_b.columns else False
        eta = float(row_b0['eta'])
        f_a = np.array([float(row_a0[f'f{k + 1}']) for k in range(path_set.n_paths)])
        f_b = np.array([float(row_b0[f'f{k + 1}']) for k in range(path_set.n_paths)])
        cfg_case = SolverConfig(q=cfg.q, theta=cfg.theta, r0=cfg.r0, alpha=1.0 - ALPHA_LIMIT_EPS if alpha_is_limit else alpha_used, eta=eta, bpr_beta=cfg.bpr_beta, bpr_gamma=cfg.bpr_gamma)
        row: dict[str, object] = {'Instance': instance.name, 'PathSet': path_set.name, 'Parameterization': str(row_b0['Parameterization']), 'alpha': alpha_display, 'alpha_requested': alpha_requested, 'alpha_used': alpha_used, 'alpha_is_limit': alpha_is_limit, 'eta': eta, 'RiskCase': classify_risk_case(eta), 'pi_A': float(row_a0['pi_od']), 'pi_B': float(row_b0['pi_od']), 'D_f_AB': float(np.sum(np.abs(f_a - f_b)) / cfg.q), 'Gap_A_at_fB': vi_gap_model_a(instance, path_set, f_b, cfg_case), 'Active_A': int(row_a0['ActiveCount']), 'Active_B': int(row_b0['ActiveCount']), 'Active_A_minus_Active_B': int(row_a0['ActiveCount'] - row_b0['ActiveCount'])}
        for k in range(path_set.n_paths):
            row[f'f{k + 1}_A_over_q'] = float(f_a[k] / cfg.q)
            row[f'f{k + 1}_B_over_q'] = float(f_b[k] / cfg.q)
        if 'lambda' in grid_df.columns:
            row['lambda'] = float(row_b0['lambda']) if not pd.isna(row_b0['lambda']) else np.nan
        row.update(tail_divergence_at_fB(instance, path_set, f_b, cfg_case))
        rows.append(row)
    sort_cols = ['alpha', 'lambda', 'eta'] if 'lambda' in grid_df.columns else ['alpha', 'eta']
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

def modelb_base_table_from_grid(grid_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    sub = grid_df[grid_df['Model'] == 'Model B'].copy()
    sub['pi_B'] = sub['pi_od']
    for k in range(1, 4):
        if f'P{k}' in sub.columns:
            sub[f'f{k}_B_over_q'] = sub[f'P{k}']
    keep = ['RiskCase', 'alpha']
    for col in ['alpha_requested', 'alpha_used', 'alpha_is_limit']:
        if col in sub.columns:
            keep.append(col)
    keep += (['lambda'] if include_lambda else []) + ['eta', 'tail_hinge_coeff', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'pi_B', 'ActiveCount', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    keep = [c for c in keep if c in sub.columns]
    return sub[keep].sort_values(['alpha'] + (['lambda'] if include_lambda else ['eta'])).reset_index(drop=True)

def compact_modelb_ab_table(modelb_df: pd.DataFrame, diag_df: pd.DataFrame, *, include_lambda: bool) -> pd.DataFrame:
    key_cols = ['alpha', 'lambda', 'eta'] if include_lambda else ['alpha', 'eta']
    diag_cols = key_cols + ['Instance', 'pi_A', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A', 'Active_B', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    diag_cols = [c for c in diag_cols if c in diag_df.columns]
    out = modelb_df.merge(diag_df[diag_cols], on=key_cols, how='left', suffixes=('', '_diag'))
    for k in range(1, 4):
        main = f'f{k}_B_over_q'
        diag = f'{main}_diag'
        if main not in out.columns and diag in out.columns:
            out[main] = out[diag]
        elif main in out.columns and diag in out.columns:
            out[main] = out[main].fillna(out[diag])
    out = out.drop(columns=[c for c in out.columns if c.endswith('_diag')], errors='ignore')
    out = out.drop(columns=['Instance', 'P1', 'P2', 'P3'], errors='ignore')
    out = out.sort_values(key_cols).reset_index(drop=True)
    preferred = ['RiskCase', 'alpha', 'alpha_requested', 'alpha_used', 'alpha_is_limit', 'lambda', 'eta', 'tail_hinge_coeff', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'pi_A', 'pi_B', 'ActiveCount', 'Active_A', 'Active_B', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    ordered = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[ordered]

def write_ab_plain_text_table(df: pd.DataFrame, output_path: Path) -> None:
    cols = ['RiskCase', 'alpha', 'alpha_used', 'eta', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'pi_A', 'pi_B', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB']
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for c in ['alpha', 'alpha_used', 'eta', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: '--' if pd.isna(v) else f'{float(v):.6g}')
    for c in ['pi_A', 'pi_B']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: '--' if pd.isna(v) else f'{float(v):.4f}')
    if 'Active_A_minus_Active_B' in out.columns:
        out['Active_A_minus_Active_B'] = out['Active_A_minus_Active_B'].astype(int)
    header = f'Full direct-eta Model A/B flow table with reservation costs and diagnostics\nRequested alpha grid: {REQUESTED_ALPHA_GRID}\nEta grid: {REQUESTED_ETA_GRID}\n\n'
    output_path.write_text(header + out.to_string(index=False), encoding='utf-8')

def write_direct_eta_display_csv(df: pd.DataFrame, output_path: Path) -> None:
    cols = ['RiskCase', 'alpha', 'eta', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'pi_A', 'pi_B', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B']
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    def sci_or_fixed(x: object, fixed: int, sci_threshold: float, zero_tol: float=0.0) -> str:
        if pd.isna(x):
            return '--'
        x = float(x)
        if abs(x) <= zero_tol:
            return f'{0.0:.{fixed}f}'
        if abs(x) < sci_threshold:
            exp = int(np.floor(np.log10(abs(x))))
            mant = x / 10 ** exp
            mant = round(mant, 1)
            if abs(mant) >= 10:
                mant /= 10
                exp += 1
            return f'{mant:.1f}e{exp}'
        return f'{x:.{fixed}f}'
    for c in ['alpha']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: f'{float(v):.2f}')
    for c in ['eta']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: f'{float(v):.1f}')
    for c in ['f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'D_tail_AB', 'D_p_AB']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: sci_or_fixed(v, 3 if c.startswith('f') or c == 'D_tail_AB' else 3, 0.0, 1e-12 if c in {'D_tail_AB', 'D_p_AB'} else 0.0))
    for c in ['pi_A', 'pi_B']:
        if c in out.columns:
            out[c] = out[c].map(lambda v: f'{float(v):.2f}' if not pd.isna(v) else '--')
    if 'D_f_AB' in out.columns:
        out['D_f_AB'] = out['D_f_AB'].map(lambda v: sci_or_fixed(v, 3, 0.001))
    if 'Gap_A_at_fB' in out.columns:
        out['Gap_A_at_fB'] = out['Gap_A_at_fB'].map(lambda v: sci_or_fixed(v, 4, 0.0001))
    if 'RiskCase' in out.columns:
        out['RiskCase'] = out['RiskCase'].astype(str).str.replace('Mean-CVaR', 'Mean--CVaR', regex=False)
    if 'Active_A_minus_Active_B' in out.columns:
        out['Active_A_minus_Active_B'] = out['Active_A_minus_Active_B'].astype(int)
    out.to_csv(output_path, index=False)

OUTPUT_DIR = Path('braess_01_direct_eta_risk_sensitivity_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _add_ab_flow_share_columns_from_long_grid(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for k in range(1, 4):
        for model_tag in ['A', 'B']:
            out[f'f{k}_{model_tag}_over_q'] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        sub = out.loc[idx]
        for model_name, model_tag in [('Model A', 'A'), ('Model B', 'B')]:
            m = sub[sub['Model'] == model_name]
            if m.empty:
                continue
            row = m.iloc[0]
            for k in range(1, 4):
                out.loc[idx, f'f{k}_{model_tag}_over_q'] = float(row[f'f{k}']) / float(Q_DEMAND)
    return out

def _ensure_ab_flow_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k in range(1, 4):
        for model_tag in ['A', 'B']:
            p_col = f'P{k}_{model_tag}'
            f_col = f'f{k}_{model_tag}_over_q'
            legacy_col = f'P{k}' if model_tag == 'B' else None
            if p_col not in out.columns:
                if f_col in out.columns:
                    out[p_col] = out[f_col]
                elif legacy_col and legacy_col in out.columns:
                    out[p_col] = out[legacy_col]
            if f_col not in out.columns and p_col in out.columns:
                out[f_col] = out[p_col]
    preferred = ['RiskCase', 'alpha', 'alpha_requested', 'alpha_used', 'alpha_is_limit', 'lambda', 'eta', 'tail_hinge_coeff', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'P1_A', 'P2_A', 'P3_A', 'P1_B', 'P2_B', 'P3_B', 'pi_A', 'pi_B', 'ActiveCount', 'Active_A', 'Active_B', 'D_tail_AB', 'Mean_D_tail_AB', 'D_p_AB', 'Mean_D_p_AB', 'TailTV_max_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B', 'WorstTailPath', 'TailSet_B_at_fB', 'TailSet_A_P1_at_fB', 'TailSet_A_P2_at_fB', 'TailSet_A_P3_at_fB', 'MeanPotential', 'CVaRPotential', 'CEPotential', 'SystemTailSelector']
    return out[[c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]]
cfg = SolverConfig()
base_paths = base_path_set()
route_local = build_route_local_instance()
validate_instance(route_local)
base_paths.validate()
USE_WIDE_ALPHA = True
if USE_WIDE_ALPHA:
    direct_cases = build_direct_eta_wide_cases()
else:
    direct_cases = []
    for alpha in [0.9, 0.95]:
        for eta in [0.0] + [round(float(x), 1) for x in np.arange(0.1, 0.91, 0.1)] + [1.0]:
            direct_cases.append({'alpha': alpha, 'eta': eta})
direct_grid = run_param_grid(route_local, base_paths, cfg, direct_cases, 'direct-eta')
direct_grid = add_pi_ab_columns_to_long_grid(direct_grid, ['alpha', 'eta'])
direct_grid = _add_ab_flow_share_columns_from_long_grid(direct_grid, ['alpha', 'eta'])
direct_grid.to_csv(OUTPUT_DIR / 'braess_direct_eta_all_models.csv', index=False)
direct_diag = diagnostics_from_grid(route_local, base_paths, cfg, direct_grid, ['alpha', 'eta'])
direct_diag.to_csv(OUTPUT_DIR / 'braess_modelA_modelB_eta_grid_diagnostics.csv', index=False)
direct_modelb = compact_modelb_ab_table(modelb_base_table_from_grid(direct_grid, include_lambda=False), direct_diag, include_lambda=False)
direct_modelb = _ensure_ab_flow_aliases(direct_modelb)
direct_modelb.to_csv(OUTPUT_DIR / 'braess_modelB_direct_eta_with_AB_diagnostics.csv', index=False)
direct_modelb.to_csv(OUTPUT_DIR / 'braess_modelB_direct_eta_with_AB_flows_diagnostics.csv', index=False)
direct_modelb.to_csv(OUTPUT_DIR / 'braess_modelB_direct_eta_with_AB_flows_diagnostics.csv', index=False)
write_direct_eta_display_csv(direct_modelb, OUTPUT_DIR / 'braess_modelB_direct_eta_with_AB_flows_table_display.csv')
write_ab_plain_text_table(direct_modelb, OUTPUT_DIR / 'braess_modelB_direct_eta_print_table.txt')
print(f'Rows in direct-eta Model A/B grid: {len(direct_grid)}')
print(f'Rows in Model B table: {len(direct_modelb)}')
preview_cols = [c for c in ['RiskCase', 'alpha', 'alpha_used', 'alpha_is_limit', 'eta', 'f1_A_over_q', 'f2_A_over_q', 'f3_A_over_q', 'f1_B_over_q', 'f2_B_over_q', 'f3_B_over_q', 'pi_A', 'pi_B', 'D_tail_AB', 'D_p_AB', 'D_f_AB', 'Gap_A_at_fB', 'Active_A_minus_Active_B'] if c in direct_modelb.columns]
display(direct_modelb[preview_cols].head(20))
display(direct_modelb[preview_cols].tail(10))
show_output_files(OUTPUT_DIR, make_zip=True)
