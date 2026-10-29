# Installation: LAMMPS on Windows via WSL2 + conda + VS Code

This guide sets up LAMMPS built from source inside a conda environment on
WSL2 (Ubuntu), and uses it from VS Code. It follows the official LAMMPS
CMake build instructions: <https://docs.lammps.org/Build_cmake.html>.

**Setup at a glance**

| Component | Choice |
|---|---|
| OS layer | WSL2, Ubuntu 24.04 |
| Package manager | Miniforge (conda-forge only, no Anaconda `defaults` channel) |
| Compilers, MPI, FFTW, CMake | From conda-forge, inside the env (single consistent toolchain) |
| LAMMPS | `stable` branch, built with CMake, installed into the conda env |
| Editor | VS Code on Windows + WSL extension |

The conda environment is defined in [`environment.yml`](../environment.yml).

---

## 1. Install WSL2 and Ubuntu

In **Windows PowerShell (as Administrator)**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if asked, then open "Ubuntu" from the Start menu and create your Linux user.

*Optional:* give WSL more resources. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB      # default is ~50% of RAM
processors=8
```

Then apply it with `wsl --shutdown` in PowerShell.

## 2. Install basic Ubuntu packages

In the Ubuntu terminal:

```bash
sudo apt update && sudo apt install -y git build-essential wget
```

> ⚠️ Do **not** install MPI via apt (`libopenmpi-dev`, `openmpi-bin`). MPI comes
> from conda. Mixing two MPI installations is the most common source of errors.

## 3. Install Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh      # accept the licence, allow "conda init"
exec bash
conda config --set auto_activate_base false
```

## 4. Clone this repository and create the conda environment

```bash
mkdir -p ~/projects
git clone https://github.com/BenediktPfluger/qtft-lammps.git ~/projects/qtft-lammps
cd ~/projects/qtft-lammps

conda env create -f environment.yml
conda activate lammps
cmake --version                      # must be >= 3.27
which mpicxx mpirun                  # must point into $CONDA_PREFIX
```

## 5. Build and install LAMMPS

```bash
git clone -b stable https://github.com/lammps/lammps.git ~/src/lammps
cd ~/src/lammps
conda activate lammps

# configure
cmake -S cmake -B build -C cmake/presets/most.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS="-march=native" \
  -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -D BUILD_MPI=on -D BUILD_OMP=on -D BUILD_SHARED_LIBS=on \
  -D FFT=FFTW3

# compile, install, and install the Python module
cmake --build build -j $(nproc)
cmake --install build
cmake --build build --target install-python
```

What these options do:

| Option | Purpose |
|---|---|
| `-C cmake/presets/most.cmake` | Enables most packages that need no external libraries |
| `CMAKE_BUILD_TYPE=Release` | Optimised build without debug info (the default is `RelWithDebInfo`) |
| `CMAKE_CXX_FLAGS="-march=native"` | Tunes the build for this CPU. **Rebuild on other machines/clusters.** |
| `CMAKE_INSTALL_PREFIX=$CONDA_PREFIX` | Installs `lmp`, `liblammps.so`, potentials etc. into the env |
| `BUILD_MPI` / `BUILD_OMP` | MPI parallelism / OpenMP threading |
| `BUILD_SHARED_LIBS=on` | Builds `liblammps.so`, **required** for the Python module |
| `FFT=FFTW3` | Uses FFTW for long-range solvers (PPPM) |
| `install-python` target | Builds a wheel and pip-installs the `lammps` Python module into the env |

Notes:

- Check the summary printed at the end of the configure step. MPI should be
  found in `$CONDA_PREFIX`, OpenMP should be on, and FFT should be `FFTW3`.
- To add a package later, re-run the configure command with the extra flag,
  then rebuild and reinstall. For example:
  ```bash
  cmake -S cmake -B build -D PKG_MOLECULE=on
  cmake --build build -j $(nproc) && cmake --install build && cmake --build build --target install-python
  ```
- `-D PKG_PYTHON=on` is only needed if you want to use the `python` command
  *inside* LAMMPS input scripts. It is not needed to use LAMMPS from Python.
- Never mix this with the legacy `make` build in the same source tree. If you
  did, clean it with `cd src && make no-all purge`.

### Optional: NVIDIA GPU

- Install only the **Windows** NVIDIA driver. Never install a Linux GPU
  driver inside WSL.
- Add the CUDA toolkit to the env with `conda install -c conda-forge cuda-toolkit`.
- Add `-D PKG_GPU=on -D GPU_API=cuda` to the configure step, or use KOKKOS
  with CUDA (see the LAMMPS docs).

## 6. Verify the installation

```bash
conda activate lammps
lmp -h | head -n 40                              # version, packages, MPI/OpenMP info

cd ~/src/lammps/bench
mpirun -np 4 lmp -in in.lj                       # MPI run, check the "Performance:" line

python -c "from lammps import lammps; l = lammps(); print(l.version()); l.close()"
```

This project needs the packages **BPM, BROWNIAN, EXTRA-PAIR, MOLECULE and MC**
(all included in the `most.cmake` preset; BPM provides `compute nbond/atom`). Check them with:

```bash
lmp -h | grep -A 30 "Installed packages" | grep -o -w -E "BPM|BROWNIAN|EXTRA-PAIR|MOLECULE|MC" | sort -u
```

All five names must be printed.

## 7. Project setup (once, after cloning)

```bash
cd ~/projects/qtft-lammps
conda activate lammps
nbstripout --install      # strips notebook outputs automatically on every commit
lammps_plugins/bond_break_random/build.sh   # unbinding plugin fix bond/break/random (needs ~/src/lammps)
pytest -q                 # self-test incl. short LAMMPS runs (~2.5 min); -m "not slow": without them (~2 s)
```

The animation (`animation.ipynb`) writes the video with **ffmpeg**, which `environment.yml` includes; in an
environment created before, install it with `conda install -c conda-forge ffmpeg`. Without it only the
stills (PNG) work, and the video test is skipped.

`build.sh` compiles the plugin against the headers in `~/src/lammps` (or the folder given as argument), which
must be the same LAMMPS version as `lmp` (it checks this). The result,
`lammps_plugins/bond_break_random/bondbreakrandom.so`, is loaded by every run with `bond_break="random"`;
`start()` refuses to run without it.

## 8. Set up VS Code

1. Install VS Code **on Windows**, plus the extension **WSL**
   (`ms-vscode-remote.remote-wsl`).
2. From the Ubuntu terminal, open the project:
   ```bash
   cd ~/projects/qtft-lammps && code .
   ```
   The bottom-left corner must show **"WSL: Ubuntu-24.04"**.
3. In that WSL window, install the extensions **Python** and **Jupyter**.
   Optionally, search the Marketplace for "LAMMPS" to get syntax
   highlighting for input files.
4. Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the
   `lammps` conda env. Use the same env for notebook kernels.
5. In the integrated terminal, run `conda activate lammps` before calling
   `lmp` or `mpirun`.

## 9. LRZ Linux cluster (CoolMUC-4), once

Runs with `Params(executor="lrz")` are built here and run on LRZ as a Slurm job array (`job.slurm`, one
task per replica, partition `serial_long`, max. 7 days). LAMMPS on LRZ is a minimal build of the same
version with `fix bond/break/random` compiled in (no plugin). Settings (host, folders, modules, email):
`LRZ` in `qtft/remote.py`.

**Email for Slurm (personal, not in git).** Create `lrz_local.json` in the repository root (gitignored):

```json
{"mail_user": "<your email>"}
```

It may set any key of `LRZ`. Without the file, jobs are submitted with `--mail-type=NONE` (no email).

**SSH with one 2FA login.** In WSL, `~/.ssh/config` (then `chmod 600 ~/.ssh/config`):

```
Host lrz
    HostName cool.hpc.lrz.de
    User <LRZ user ID>
    ForwardX11 yes
    ForwardX11Trusted yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

`ssh lrz` asks for password and 2FA once; for the next 8 hours `ssh lrz` / `scp … lrz:` / `rsync` reuse
that connection without asking.

**Build LAMMPS on LRZ** (on the login node, ~4 min). Use `gcc/15.2.0`: the Spack GCC modules (12.4–14.2)
cannot compile `#include <mutex>` (their `include-fixed/pthread.h` does not match the system headers).
Without conda (`conda deactivate`, if conda is installed there):

```bash
module load gcc/15.2.0 intel-mpi/2021.12.0 cmake/3.30.0
git clone -b stable_30Sep2026 --depth 1 https://github.com/lammps/lammps.git ~/src/lammps
# from WSL: copy the fix into the LAMMPS source, so that it is compiled in
#   scp lammps_plugins/bond_break_random/fix_bond_break_random.{h,cpp} lrz:src/lammps/src/
cd ~/src/lammps
cmake -S cmake -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-march=x86-64-v3" \
  -D BUILD_MPI=on -D BUILD_OMP=off \
  -D PKG_BPM=on -D PKG_BROWNIAN=on -D PKG_EXTRA-PAIR=on -D PKG_MC=on -D PKG_MOLECULE=on
cmake --build build -j 4
```

`-march=x86-64-v3` (AVX2) instead of `native`: the login and compute nodes may differ. Use the tag of the
local LAMMPS version (`lmp -h`): with the same version, LRZ and local runs give the same results.

**Check it** — LAMMPS starts only inside a Slurm job (on the login node even `lmp -h` aborts with
`PMI2_Job_GetId returned 14`):

```bash
srun -p cm4_inter -n 1 -t 00:05:00 ~/src/lammps/build/lmp -h | grep -c bond/break/random   # 1
```

A small input built locally with `executor="lrz"` gave, on 2 and on 8 MPI processes (`cm4_inter`,
`serial_long`), byte-identical dumps to the same LAMMPS version built locally (2026-10-25). Speed in
`serial_long` (Xeon Platinum 8380, shared node): ≈ 0.68 × this PC per process (peak state of the 3 µm run,
8 processes: ≈ 1735 vs 2550 steps/s), so the 3 µm run (24 h here) takes ≈ 35 h there.

**What the job script needs on LRZ** (all in `job_template.slurm`): `--clusters=serial
--partition=serial_long --qos=cm4_serial_long` (the default QOS `cm4_serial` is refused: "Invalid qos
specification"); `#!/bin/bash -l` (LRZ's base setup with the Intel MPI settings runs only in a login shell);
`--export=NONE` plus `export SLURM_EXPORT_ENV=ALL` (otherwise `srun` starts LAMMPS with an empty
environment); `srun --mpi=pmi2` (the serial cluster starts no PMI by default); `--mem` (default 6.4 GB per
CPU, i.e. 102 GB for 8 tasks with hyperthreads, the whole per-user limit of `serial_long`; LAMMPS needs
~150 MB per process, `slurm_mem` = 4G). Per-user limits of `serial_long`: 1 node per job, 192 CPUs
(hyperthreads: 8 MPI tasks count as 16) and ~100 GB in total, e.g. 10 replicas × 8 processes at once.

**Use it from the notebook.** Open the connection once with `ssh lrz` in a terminal, set
`executor="lrz"` in the parameter cell of `control.ipynb` (and `max_parallel=None`, so that all replicas
run at once; a number limits the job array, `check()` warns), then as usual `build_run` → `start` →
`watch`. The run folder on LRZ is `~/qtft-lammps/runs/<group>/<run name>/` (the same path below the
repository as here).

**When the run has finished**, its outputs stay on LRZ: nothing is copied back by itself, and nothing
runs in the background on this PC. While `watch`, `progress` or `summary` run, only `status`, `pid` and
the new lines of the logs come over (every 30 s at most), enough for the progress table and `quicklook`
(without the overlap panels). `fetch(run_dir)` downloads everything (≈ 360 MB per replica of the 3 µm
run), whenever you want; until then `make_report`, `ensemble_stats`, the structure, cluster-volume and
FIB-SEM analysis and `binding_rate` stop with "download the results first with fetch(run_dir)". If the
ssh connection is lost, `watch` stops with a notice; the run continues on LRZ.

## Running simulations

```bash
conda activate lammps
mpirun -np <N> lmp -in in.myscript               # N MPI ranks
OMP_NUM_THREADS=2 mpirun -np 4 lmp -sf omp -in in.myscript   # hybrid MPI + OpenMP
```

**Performance tips**

- Keep simulation files in the Linux filesystem (`~/...`), **not** under
  `/mnt/c/...`. File I/O across the Windows boundary is very slow.
- Use `N` ≤ the number of physical cores (`lscpu`: "Core(s) per socket" × "Socket(s)",
  not "CPU(s)", which counts hyperthreads). Open MPI refuses more without `--use-hwthread-cpus`.
- To view results from Windows (e.g. in OVITO), open
  `\\wsl$\Ubuntu-24.04\home\<user>\...` in Explorer.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `lmp: command not found` | Run `conda activate lammps` |
| `mpirun` uses the wrong MPI | `which mpirun` must be in `$CONDA_PREFIX/bin`; remove apt MPI packages |
| CMake version too old | Use the conda env's CMake (`conda activate lammps`) |
| `ImportError` / `liblammps.so` not found in Python | Build with `BUILD_SHARED_LIBS=on`, then re-run the `install-python` target |
| CMake error about conflicting files in `src/` | `cd ~/src/lammps/src && make no-all purge`, then reconfigure |
| `LAMMPS plugin not found` from `start()`, or `plugin load` fails | Build (or rebuild after a LAMMPS update) with `lammps_plugins/bond_break_random/build.sh` |
| LRZ: `cannot convert '<brace-enclosed initializer list>' to 'unsigned int'` when building | Use `module load gcc/15.2.0` (see section 9), delete `build/` and configure again |
| LRZ: `Invalid qos specification` from `sbatch` | `--qos=cm4_serial_long` for `serial_long` (in `job_template.slurm`) |
| LRZ: `PMI server not found` or `PMI2_Job_GetId returned 14` | LAMMPS was started outside Slurm, or `srun` without `--mpi=pmi2` |

## Updating LAMMPS

```bash
cd ~/src/lammps && git pull
conda activate lammps
cmake --build build -j $(nproc) && cmake --install build && cmake --build build --target install-python
~/projects/qtft-lammps/lammps_plugins/bond_break_random/build.sh   # the plugin must match the new version
```
