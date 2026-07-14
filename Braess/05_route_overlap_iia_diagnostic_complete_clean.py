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

def _solve_model_b_direct_objective_fallback(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0) -> tuple[np.ndarray, opt.OptimizeResult]:
    cfg_case = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    n = path_set.n_paths
    deterministic_penalty = overlap_penalty(instance, path_set, cfg_case, beta_ps=beta_ps, overlap_model=overlap_model)

    def objective_f(f_raw: np.ndarray) -> float:
        f = np.asarray(f_raw, dtype=float)
        if np.any(f < -1e-07):
            return 1e+30
        f = np.maximum(f, 0.0)
        total = f.sum()
        if total <= 0.0:
            return 1e+30
        f = cfg_case.q * f / total
        vals = potentials_by_scenario(instance, path_set, f, cfg_case)
        ce_val, _, _ = mean_cvar_value_and_tilt(vals, instance.probabilities, alpha, eta)
        return float(ce_val + entropy(f, cfg_case) / cfg_case.theta + np.dot(deterministic_penalty, f))
    constraints = ({'type': 'eq', 'fun': lambda f: float(np.sum(f) - cfg_case.q)},)
    bounds = opt.Bounds(np.zeros(n), np.full(n, cfg_case.q))
    starts: list[np.ndarray] = [np.full(n, cfg_case.q / n, dtype=float)]
    for k in range(n):
        e = np.zeros(n, dtype=float)
        e[k] = cfg_case.q
        starts.append(e)
    for i in range(n):
        for j in range(i + 1, n):
            e = np.zeros(n, dtype=float)
            e[i] = 0.5 * cfg_case.q
            e[j] = 0.5 * cfg_case.q
            starts.append(e)
    best: opt.OptimizeResult | None = None
    for x0 in starts:
        res = opt.minimize(objective_f, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-10, 'maxiter': 2500, 'disp': False})
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('Patched fallback Model B direct objective solver did not run.')
    f = np.maximum(np.asarray(best.x, dtype=float), 0.0)
    if f.sum() <= 0.0:
        f[:] = cfg_case.q / n
    else:
        f *= cfg_case.q / f.sum()
    best.x = f
    return (f, best)

def solve_model_b_iia_sweep_result(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str, beta_ps: float) -> EquilibriumResult:
    cfg_case = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    start = time.perf_counter()
    if alpha >= 0.955:
        f, res = _solve_model_b_direct_objective_fallback(instance, path_set, cfg_base, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
        raw = model_b_raw_costs(instance, path_set, f, cfg_case)[0]
        perceived, penalty = add_overlap(raw, instance, path_set, cfg_case, beta_ps=beta_ps, overlap_model=overlap_model)
        _, probs, pi = truncated_flows_from_costs(perceived, cfg_case)
        if abs(eta) <= 1e-12:
            model_label = 'Model B expectation TSUE'
        elif abs(eta - 1.0) <= 1e-12:
            model_label = 'Model B CVaR-only TSUE'
        else:
            model_label = 'Model B mean-CVaR TSUE'
        return EquilibriumResult(instance=instance.name, path_set=path_set.name, model=model_label, overlap_model=overlap_model, beta_ps=beta_ps, flows=f, perceived_costs=perceived, raw_costs=raw, overlap_penalty=penalty, probabilities=probs, pi_od=pi, link_flows=link_flows_from_path_flows(path_set, f), iterations=int(getattr(res, 'nit', -1)), residual=float('nan'), active_paths=tuple((path_set.path_names[i] for i, val in enumerate(f) if val > 1e-06)), cpu_seconds=time.perf_counter() - start)
    return solve_model_b_for_params_labeled(instance, path_set, cfg_base, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)

from matplotlib.lines import Line2D
OUTPUT_DIR = Path('braess_05_route_overlap_iia_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
cfg = SolverConfig()
route_local = build_route_local_instance()
base_paths = base_path_set()
clone_paths = cloned_p1_path_set()
for pset in [base_paths, clone_paths]:
    pset.validate()
validate_instance(route_local)
IIA_ALPHA_GRID = [0.84, 0.88, 0.92, 0.96]
IIA_ETA_GRID = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 11)]
IIA_BETA_PS_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
IIA_PLOT_DPI = 600
FORCE_RECOMPUTE_IIA_SWEEP = False
USE_COMMON_TAIL_SHORTCUT_FOR_ALPHA_096 = True

def solve_model_a_iia_sweep_result(instance: BraessInstance, path_set: PathSet, cfg_base: SolverConfig, alpha: float, eta: float, overlap_model: str='none', beta_ps: float=0.0, initial_flows: np.ndarray | None=None) -> EquilibriumResult:
    cfg_case = SolverConfig(q=cfg_base.q, theta=cfg_base.theta, r0=cfg_base.r0, alpha=alpha, eta=eta, bpr_beta=cfg_base.bpr_beta, bpr_gamma=cfg_base.bpr_gamma)
    model_label = 'Model A mean--CVaR TSUE'
    if abs(eta) <= 1e-12:
        model_label = 'Model A expectation TSUE'
    elif abs(eta - 1.0) <= 1e-12:
        model_label = 'Model A CVaR-only TSUE'
    return solve_fixed_point(instance, path_set, model_label, lambda f: model_a_raw_costs(instance, path_set, f, cfg_case)[0], cfg_case, truncated=True, overlap_model=overlap_model, beta_ps=beta_ps, relaxation=0.18, tol=1e-08, max_iter=12000, initial_flows=initial_flows)

def _clone_distortion_row(model_name: str, res_base: EquilibriumResult, res_clone: EquilibriumResult, alpha: float, eta: float, beta_ps: float, overlap_model: str) -> dict[str, object]:
    g_base = aggregate_group_shares(base_paths, res_base.flows)
    g_clone = aggregate_group_shares(clone_paths, res_clone.flows)
    tv = 0.5 * sum((abs(g_clone[g] - g_base[g]) for g in g_base))
    clone_pair_total = max(float(res_clone.flows[0] + res_clone.flows[1]), 1e-12)
    return {'Model': model_name, 'alpha': alpha, 'eta': eta, 'beta_PS': beta_ps, 'OverlapModel': overlap_model, 'CloneDistortion_TV': float(tv), 'Base_P1_corridor': float(g_base['P1-corridor']), 'Base_P2': float(g_base['P2']), 'Base_P3': float(g_base['P3']), 'Clone_P1_corridor': float(g_clone['P1-corridor']), 'Clone_P2': float(g_clone['P2']), 'Clone_P3': float(g_clone['P3']), 'CloneSplit_P1_share_within_pair': float(res_clone.flows[0] / clone_pair_total), 'Base_pi': float(res_base.pi_od) if res_base.pi_od is not None else np.nan, 'Clone_pi': float(res_clone.pi_od) if res_clone.pi_od is not None else np.nan, 'Base_ActiveCount': int(len(res_base.active_paths)), 'Clone_ActiveCount': int(len(res_clone.active_paths)), 'Base_iterations': int(res_base.iterations), 'Clone_iterations': int(res_clone.iterations), 'Base_cpu_seconds': float(res_base.cpu_seconds), 'Clone_cpu_seconds': float(res_clone.cpu_seconds)}

def run_iia_modelAB_eta_beta_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    last_model_a_base: dict[tuple[float, float, float], np.ndarray] = {}
    last_model_a_clone: dict[tuple[float, float, float], np.ndarray] = {}
    for alpha in IIA_ALPHA_GRID:
        for beta_ps in IIA_BETA_PS_GRID:
            overlap_model = 'none' if abs(beta_ps) < 1e-12 else 'path-size'
            for eta in IIA_ETA_GRID:
                res_base_b = solve_model_b_iia_sweep_result(route_local, base_paths, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
                res_clone_b = solve_model_b_iia_sweep_result(route_local, clone_paths, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps)
                if USE_COMMON_TAIL_SHORTCUT_FOR_ALPHA_096 and alpha >= 0.955:
                    res_base_a = res_base_b
                    res_clone_a = res_clone_b
                else:
                    prev_etas = [e for e in IIA_ETA_GRID if e < eta]
                    init_base = None
                    init_clone = None
                    if prev_etas:
                        init_base = last_model_a_base.get((alpha, beta_ps, max(prev_etas)))
                        init_clone = last_model_a_clone.get((alpha, beta_ps, max(prev_etas)))
                    if init_base is None:
                        init_base = res_base_b.flows
                    if init_clone is None:
                        init_clone = res_clone_b.flows
                    res_base_a = solve_model_a_iia_sweep_result(route_local, base_paths, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps, initial_flows=init_base)
                    res_clone_a = solve_model_a_iia_sweep_result(route_local, clone_paths, cfg, alpha, eta, overlap_model=overlap_model, beta_ps=beta_ps, initial_flows=init_clone)
                    last_model_a_base[alpha, beta_ps, eta] = res_base_a.flows.copy()
                    last_model_a_clone[alpha, beta_ps, eta] = res_clone_a.flows.copy()
                rows.append(_clone_distortion_row('Model B', res_base_b, res_clone_b, alpha, eta, beta_ps, overlap_model))
                rows.append(_clone_distortion_row('Model A', res_base_a, res_clone_a, alpha, eta, beta_ps, overlap_model))
            print(f'finished alpha={alpha:.2f}, beta_PS={beta_ps:.1f}')
    long_df = pd.DataFrame(rows)
    summary_df = long_df.groupby(['Model', 'alpha', 'beta_PS'], as_index=False).agg(MaxCloneDistortion_TV=('CloneDistortion_TV', 'max'), MeanCloneDistortion_TV=('CloneDistortion_TV', 'mean'))
    return (long_df, summary_df)

def plot_iia_four_single_alpha_figures(long_df: pd.DataFrame) -> list[Path]:
    cmap = plt.get_cmap('tab10')
    color_map = {beta: cmap(i % 10) for i, beta in enumerate(IIA_BETA_PS_GRID)}
    linestyle_map = {'Model B': '-', 'Model A': '--'}
    marker_map = {'Model B': 'o', 'Model A': None}
    y_label = 'Clone distortion' + chr(10) + '(total variation of aggregate route shares)'
    output_paths: list[Path] = []

    def legend_handles() -> list[Line2D]:
        beta_handles = [Line2D([0], [0], color=color_map[b], linewidth=2.0, label=f'β_PS = {b:.1f}') for b in IIA_BETA_PS_GRID]
        model_handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=2.0, marker='o', markersize=4, label='Model B'), Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label='Model A')]
        return beta_handles + model_handles
    for alpha in IIA_ALPHA_GRID:
        fig, ax = plt.subplots(figsize=(8.9, 5.6))
        sub_alpha = long_df[np.isclose(long_df['alpha'].astype(float), alpha)].copy()
        for beta_ps in IIA_BETA_PS_GRID:
            for model_name in ['Model B', 'Model A']:
                sub = sub_alpha[np.isclose(sub_alpha['beta_PS'].astype(float), beta_ps) & (sub_alpha['Model'] == model_name)].sort_values('eta')
                ax.plot(sub['eta'], sub['CloneDistortion_TV'], linestyle=linestyle_map[model_name], marker=marker_map[model_name], linewidth=1.8 if model_name == 'Model B' else 1.5, markersize=4.0, color=color_map[beta_ps])
        ax.set_xlabel('CVaR weight eta')
        ax.set_ylabel(y_label)
        ax.set_title(f'Model A (dashed) and Model B (solid), alpha={alpha:.2f}')
        ax.set_xticks(IIA_ETA_GRID)
        ax.grid(True, alpha=0.28)
        ax.set_ylim(bottom=0.0)
        ax.legend(handles=legend_handles(), loc='upper right', ncol=1, fontsize=8, frameon=True, framealpha=0.88, edgecolor='0.85')
        fig.tight_layout()
        alpha_tag = f'{int(round(alpha * 100)):03d}'
        png_path = FIG_DIR / f'braess_iia_clone_distortion_modelAB_alpha{alpha_tag}_eta_beta0_2_legend_ur_600dpi.png'
        pdf_path = FIG_DIR / f'braess_iia_clone_distortion_modelAB_alpha{alpha_tag}_eta_beta0_2_legend_ur.pdf'
        legacy_png = FIG_DIR / f'braess_iia_clone_distortion_modelAB_alpha{alpha_tag}_eta_beta_600dpi.png'
        fig.savefig(png_path, dpi=IIA_PLOT_DPI, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(legacy_png, dpi=IIA_PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        output_paths.extend([png_path, pdf_path, legacy_png])
    return output_paths
long_csv = OUTPUT_DIR / 'braess_iia_clone_distortion_modelAB_alpha_eta_beta_sweep.csv'
summary_csv = OUTPUT_DIR / 'braess_iia_clone_distortion_modelAB_alpha_beta_summary.csv'
needs_recompute = FORCE_RECOMPUTE_IIA_SWEEP or not long_csv.exists()
if not needs_recompute:
    iia_df = pd.read_csv(long_csv)
    required = pd.MultiIndex.from_product([['Model A', 'Model B'], IIA_ALPHA_GRID, IIA_ETA_GRID, IIA_BETA_PS_GRID], names=['Model', 'alpha', 'eta', 'beta_PS'])
    present = pd.MultiIndex.from_frame(iia_df[['Model', 'alpha', 'eta', 'beta_PS']].drop_duplicates())
    missing = required.difference(present)
    if len(missing) > 0:
        print(f'Cached CSV is missing {len(missing)} required rows; recomputing sweep.')
        needs_recompute = True
if needs_recompute:
    iia_df, iia_summary = run_iia_modelAB_eta_beta_sweep()
else:
    iia_df = pd.read_csv(long_csv)
    iia_df = iia_df[iia_df['alpha'].round(2).isin(IIA_ALPHA_GRID) & iia_df['eta'].round(2).isin(IIA_ETA_GRID) & iia_df['beta_PS'].round(2).isin(IIA_BETA_PS_GRID) & iia_df['Model'].isin(['Model A', 'Model B'])].copy()
    iia_summary = iia_df.groupby(['Model', 'alpha', 'beta_PS'], as_index=False).agg(MaxCloneDistortion_TV=('CloneDistortion_TV', 'max'), MeanCloneDistortion_TV=('CloneDistortion_TV', 'mean'))
iia_df.to_csv(long_csv, index=False)
iia_df.to_csv(OUTPUT_DIR / 'braess_iia_clone_distortion_modelAB_alpha_eta_beta_sweep_beta0_2.csv', index=False)
iia_summary.to_csv(summary_csv, index=False)
iia_summary.to_csv(OUTPUT_DIR / 'braess_iia_clone_distortion_modelAB_alpha_beta_summary_beta0_2.csv', index=False)
plot_paths = plot_iia_four_single_alpha_figures(iia_df)
zip_path = Path('iia_modelAB_complete_05_outputs_package.zip')
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in [long_csv, summary_csv] + plot_paths:
        if path.exists():
            zf.write(path, arcname=path.name)
print('Final IIA Model A/B outputs written to:')
for path in [long_csv, summary_csv, zip_path] + plot_paths:
    print(' -', path)
