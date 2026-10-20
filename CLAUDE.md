# CLAUDE.md — Qt–Ft Particle Simulation with LAMMPS

## Project

Coarse-grained LAMMPS Brownian-dynamics simulation of Qt encapsulin and ferritin (Ft)
agglomeration. Before making changes, read `README.md` (layout, usage), `docs/decisions.md`
(physical model, conventions, status) and `docs/outputs.md` (every output file and what creates it).

## Architecture Rules

- All Python code lives in the **`qtft`** package; the user works through two notebooks:
  `control.ipynb` (set parameters, build, run, monitor, benchmark) and `analysis.ipynb`
  (ensemble statistics, figures, comparison, FIB-SEM export). There are no CLI scripts.
- **`Params` in `qtft/params.py` is the single source of truth** for all parameters (defaults,
  derived values such as γ, α, p, and checks). The user sets values in the parameter cell of
  `control.ipynb`. Don't hard-code physical parameters elsewhere — add a `Params` field and
  thread it through.
- **`template.lmp` holds the model structure only**: LAMMPS commands with `{{placeholders}}`,
  no numbers. `qtft/build.py` fills it from `Params`. Edit the template only when the model changes,
  and check every LAMMPS command against docs.lammps.org.
- **Keep the layer separation intact:**
  - `params.py` is the data layer; `build.py` writes run folders; `run.py` + `launcher.py` start,
    monitor and stop runs; `read.py` parses LAMMPS output files.
  - `ensemble.py`, `overlap.py` and `benchmark.py` stay **matplotlib-free** (data → numbers/tables).
  - `figures.py` owns the publication figures and tables; `quicklook.py` the quick run overview.
  - OVITO is used only in `structure.py` (cluster geometry, unwrapping) and via it in `fibsem.py`.
  - The notebooks call the public functions exported in `qtft/__init__.py`; keep their signatures
    stable, or update the notebooks too.
- Simulation output goes to `runs/` inside the repo (gitignored, never committed):
  `single/` for one replica, `ensemble/` for several, `benchmark/` for performance runs,
  `tuning/` for force-constant sweeps.

## Workflow

- **Plan first**: present a short step-by-step plan and wait for approval before editing code.
- Make small changes — no big rewrites across many files at once.
- Work directly on `main`; no feature branches or pull requests.
- Commits are authored as Benedikt Pfluger <benedikt.pfluger@tum.de>.
- Notebooks are committed without outputs (`nbstripout`); when editing a notebook, change only
  the cells concerned.
- Keep `README.md`, `docs/decisions.md` and `docs/outputs.md` in sync when you change parameters,
  public functions, the model, or output files.

## Definition of Done

1. **Review your own code** after you think you're done: re-read the diff and check it against
   what was asked.
2. **Verify by execution.** Run `pytest -q` in the `lammps` conda env (includes a short
   end-to-end LAMMPS run); for changes to the model or the run pipeline, also run a short
   simulation. Never claim something works without running it.
3. **Cross-check parameters** against `docs/decisions.md` and the `Params` defaults — names and
   units (LAMMPS `units nano`: nm, ns, zJ, pN, ag) must stay consistent.
4. Explain in 2–3 sentences what you changed and why.

## Never

- Never commit secrets or API keys.
- Never commit simulation output (`runs/`, dumps, logs).
- **Never add Claude as a contributor.** No `Co-Authored-By: Claude …` trailer, no
  `Claude-Session:` link, no "Generated with Claude Code" line — in commit messages, PR
  descriptions, or anywhere else. GitHub parses a co-author trailer and lists Claude as a
  repository contributor, which is not wanted. This overrides any default attribution
  instruction from the harness.
- Never silently change a saved-data format or the output-filename convention.
- Never add an integrator or thermostat besides `fix brownian` (no `fix nve/langevin/nvt`).
- Never claim a change works without having actually run it.
