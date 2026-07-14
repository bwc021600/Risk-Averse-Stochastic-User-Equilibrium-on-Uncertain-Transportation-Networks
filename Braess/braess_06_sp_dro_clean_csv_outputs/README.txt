Braess finite-support Wasserstein DRO / reliable B->C upgrade outputs

Run this notebook with Run All to regenerate every file in this directory.

Case grid:
  alpha = [0.92, 0.96]
  eta   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

Models:
  - Empirical Model B-SP: rho = 0.00
  - Wasserstein DRO Model B: rho = [0.05, 0.1, 0.2]

Direct DRO table radius: rho = 0.1
Alpha=0.96 radius-sensitivity table: rho = [0.05, 0.1, 0.2]
Representative deduplicated worst-case probability table: alpha = 0.96, eta = 0.6

CSV policy:
00_case_results_master_full.csv contains all solved SP/DRO cases and full solver/test diagnostics.
01_* through 07_* are source CSVs for the paper tables, with full precision values and enough parameter columns to locate the corresponding table entries.
08_* through 10_* are metadata files for the upgrade, laws, and scenario metric.
No obsolete alpha=0.90/0.95 grids, empty CSVs, or rounded-only table-check CSVs are written.

How to find a table number:
1. Use the matching SOURCE_FULL.csv for that table.
2. Filter by alpha, eta, rho, Model, and Network as needed.

Generated at: 2026-07-14 15:55:37
Python: 3.13.1 (v3.13.1:06714517797, Dec  3 2024, 14:00:22) [Clang 15.0.0 (clang-1500.3.9.4)]
Platform: macOS-15.6.1-arm64-arm-64bit-Mach-O
Elapsed seconds: 2.01
