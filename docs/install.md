# Installation

This guide sets up everything needed to run the simulation:

- **Part A, this PC:** WSL2 with Ubuntu, a conda environment, LAMMPS built from source, and VS Code.
- **Part B, LRZ:** runs on the LRZ Linux cluster (CoolMUC-4). This part is optional and only needed for
  `Params(executor="lrz")`.

| Component | Choice |
|---|---|
| Operating system layer | WSL2 with Ubuntu 24.04 on Windows 10 (2004 or later) or Windows 11 |
| Package manager | Miniforge (conda-forge only, no Anaconda `defaults` channel) |
| Compilers, MPI, FFTW, CMake | from conda-forge, inside the environment (one consistent toolchain) |
| LAMMPS | `stable` branch, built with CMake, installed into the conda environment |
| Editor | VS Code on Windows with the WSL extension |
| Repository folder | `~/qtft-lammps` (the LRZ setup uses the same folder name) |

The conda environment is defined in [`environment.yml`](../environment.yml). The LAMMPS build follows the
official CMake instructions: <https://docs.lammps.org/Build_cmake.html>.

---

# Part A: this PC

## 1. Install WSL2 and Ubuntu

1. Open **Windows PowerShell as administrator** (right-click, "Run as administrator") and run:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   If the command only prints the WSL help text, WSL is already installed: list the available
   distributions with `wsl --list --online` and run the command again.

2. Restart Windows when asked.
3. Open **Ubuntu 24.04** from the Start menu. On the first start, choose a Linux user name and password.
4. Check in PowerShell that the distribution runs as WSL 2:

   ```powershell
   wsl --list --verbose          # VERSION must be 2
   ```

<details>
<summary>Optional: give WSL more memory and processors</summary>

By default WSL uses about half of the RAM. Create `C:\Users\<Windows user>\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=8
```

Then apply it with `wsl --shutdown` in PowerShell and open Ubuntu again.

</details>

## 2. Install the Ubuntu packages

In the Ubuntu terminal:

```bash
sudo apt update && sudo apt install -y git build-essential wget
```

> [!WARNING]
> Do not install MPI with apt (`libopenmpi-dev`, `openmpi-bin`). MPI comes from the conda environment, and
> two MPI installations side by side are the most common source of errors.

## 3. Install Miniforge (conda)

```bash
cd ~
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh     # accept the licence; answer "yes" to initialise conda
exec bash                                    # reload the shell
conda config --set auto_activate false       # do not activate the base environment in every shell
```

## 4. Clone the repository and create the environment

```bash
git clone https://github.com/BenediktPfluger/qtft-lammps.git ~/qtft-lammps
cd ~/qtft-lammps
conda env create -f environment.yml
conda activate lammps
```

Check the toolchain:

```bash
cmake --version          # 3.27 or newer
which mpicxx mpirun      # both inside ~/miniforge3/envs/lammps/bin
```

Then switch on **nbstripout** (installed with the environment) for this repository:

```bash
nbstripout --install
git check-attr filter control.ipynb      # control.ipynb: filter: nbstripout
```

Notebooks are committed without outputs: from now on `git add` stores the notebooks without their outputs,
while the notebooks on disk keep them. The setting lives in `.git/` and has to be repeated after every new
clone.

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

# compile (several minutes), install lmp into the environment, install the Python module
cmake --build build -j $(nproc)
cmake --install build
cmake --build build --target install-python
```

At the end of the configure step, check the summary: MPI found in the conda environment, OpenMP on,
FFT `FFTW3`.

| Option | Purpose |
|---|---|
| `-C cmake/presets/most.cmake` | enables most packages that need no external libraries, including the five this project needs |
| `CMAKE_BUILD_TYPE=Release` | optimised build without debug information |
| `CMAKE_CXX_FLAGS="-march=native"` | tuned for this CPU; build again on another machine |
| `CMAKE_INSTALL_PREFIX=$CONDA_PREFIX` | installs `lmp` and `liblammps.so` into the environment |
| `BUILD_MPI`, `BUILD_OMP` | MPI parallelism, OpenMP threading |
| `BUILD_SHARED_LIBS=on` | builds `liblammps.so`, which the Python module needs |
| `FFT=FFTW3` | FFTW for long-range solvers |
| target `install-python` | installs the `lammps` Python module into the environment |

<details>
<summary>Add a LAMMPS package later</summary>

Configure again with the extra option, then build and install again, for example:

```bash
cd ~/src/lammps
cmake -S cmake -B build -D PKG_MOLECULE=on
cmake --build build -j $(nproc) && cmake --install build && cmake --build build --target install-python
```

`-D PKG_PYTHON=on` is only needed for the `python` command inside LAMMPS input scripts, not to use LAMMPS
from Python. Do not mix the CMake build with the legacy `make` build in the same source folder; if that
happened, clean it with `cd ~/src/lammps/src && make no-all purge`.

</details>

<details>
<summary>Optional: NVIDIA GPU</summary>

- Install only the **Windows** NVIDIA driver, never a Linux GPU driver inside WSL.
- Add the CUDA toolkit to the environment: `conda install -c conda-forge cuda-toolkit`.
- Add `-D PKG_GPU=on -D GPU_API=cuda` to the configure step, or use KOKKOS with CUDA (see the LAMMPS docs).

The simulation in this repository does not use the GPU.

</details>

## 6. Check LAMMPS and set up the project

1. LAMMPS runs, with MPI and from Python:

   ```bash
   conda activate lammps
   lmp -h | head -n 20                     # version (e.g. "30 Sep 2026"), packages, MPI and OpenMP
   cd ~/src/lammps/bench
   mpirun -np 4 lmp -in in.lj              # a short MPI run; see the "Performance:" line
   python -c "from lammps import lammps; l = lammps(); print(l.version()); l.close()"
   ```

2. The five packages this project needs are installed:

   ```bash
   lmp -h | grep -A 30 "Installed packages" | grep -o -w -E "BPM|BROWNIAN|EXTRA-PAIR|MOLECULE|MC" | sort -u
   ```

   All five names must be printed.

3. Build the unbinding plugin (`fix bond/break/random`):

   ```bash
   cd ~/qtft-lammps
   lammps_plugins/bond_break_random/build.sh
   ```

   It compiles against the LAMMPS source in `~/src/lammps` (or the folder given as argument), which must be
   the same version as `lmp` (the script checks this). Every run with `bond_break="random"` (the default)
   loads the result, `lammps_plugins/bond_break_random/bondbreakrandom.so`; `start()` refuses to run
   without it.

4. Run the tests:

   ```bash
   pytest -q                    # all tests, including short LAMMPS runs (about 4 min)
   pytest -q -m "not slow"      # without the LAMMPS runs (a few seconds)
   ```

## 7. Set up VS Code

1. Install [VS Code](https://code.visualstudio.com/) **on Windows** and, in VS Code, the extension
   **WSL** (`ms-vscode-remote.remote-wsl`).
2. Open the project from the Ubuntu terminal:

   ```bash
   cd ~/qtft-lammps && code .
   ```

   The bottom-left corner of the window must show **WSL: Ubuntu-24.04**.
3. In this WSL window, install the extensions **Python** and **Jupyter**.
4. `Ctrl+Shift+P` → **Python: Select Interpreter** → the `lammps` environment. Choose the same environment
   as the kernel of the notebooks.
5. In the integrated terminal, run `conda activate lammps` before `lmp`, `mpirun` or `pytest`.

To view this guide formatted in VS Code, open it and press `Ctrl+Shift+V`.

---

# Part B: LRZ Linux cluster (CoolMUC-4)

Runs with `Params(executor="lrz")` are built on this PC and run on LRZ as one Slurm job array
(`job.slurm` from `job_template.slurm`, one task per replica, partition `serial_long`, at most 7 days). LAMMPS
on LRZ is a minimal build of the same version, with `fix bond/break/random` compiled in, so no plugin is
needed there. All LRZ settings (host, folders, modules, Slurm partition) are in `LRZ` in `qtft/remote.py`.

## 8. Prerequisites

- An LRZ account with access to the Linux cluster, for example through your chair or project.
- Two-factor authentication (2FA), which LRZ requires for every login. Register a token in the
  [LRZ SIM-MFA portal](https://simmfa.sim.lrz.de) with your LRZ credentials (see the LRZ page
  [Two-Factor Authentication on the Linux Cluster](https://doku.lrz.de/two-factor-authentication-on-the-linux-cluster-11484383.html)).

## 9. SSH with one login

The notebook talks to LRZ over ssh without asking for a password. For this, ssh keeps one connection open
for 8 hours after you have logged in once.

1. In WSL, create or extend `~/.ssh/config`, with your LRZ user ID in place of `<LRZ user ID>`:

   ```
   Host lrz
       HostName cool.hpc.lrz.de
       User <LRZ user ID>
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 8h
   ```

   ```bash
   chmod 600 ~/.ssh/config
   ```

2. Log in once in a terminal:

   ```bash
   ssh lrz
   ```

   ssh asks for the password, then for the 2FA code (for a push token, press Enter and confirm on the
   phone). For the next 8 hours, `ssh lrz`, `rsync` and the notebook use this connection without asking.

## 10. Email from Slurm (optional)

To get an email when a job starts, ends or fails, create `lrz_local.json` in the repository folder:

```json
{"mail_user": "<your email>"}
```

The file is gitignored, so the address never reaches GitHub. It can set any key of `LRZ` in
`qtft/remote.py`. Without the file, jobs send no email.

## 11. Build LAMMPS on LRZ

Build the same LAMMPS version as on this PC. Look up its tag on this PC:

```bash
git -C ~/src/lammps describe --tags      # e.g. stable_30Sep2026
```

1. **On LRZ** (after `ssh lrz`), clone this tag. Do not use conda there; if it is installed, run
   `conda deactivate` first.

   ```bash
   git clone -b stable_30Sep2026 --depth 1 https://github.com/lammps/lammps.git ~/src/lammps
   ```

2. **In WSL**, copy the unbinding fix into the LAMMPS source on LRZ, so that it is compiled in:

   ```bash
   cd ~/qtft-lammps
   scp lammps_plugins/bond_break_random/fix_bond_break_random.{h,cpp} lrz:src/lammps/src/
   ```

3. **On LRZ**, configure and compile (about 4 minutes on the login node):

   ```bash
   module load gcc/15.2.0 intel-mpi/2021.12.0 cmake/3.30.0
   cd ~/src/lammps
   cmake -S cmake -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-march=x86-64-v3" \
     -D BUILD_MPI=on -D BUILD_OMP=off \
     -D PKG_BPM=on -D PKG_BROWNIAN=on -D PKG_EXTRA-PAIR=on -D PKG_MC=on -D PKG_MOLECULE=on
   cmake --build build -j 4
   ```

> [!IMPORTANT]
> Use `gcc/15.2.0`. The other GCC modules (12.4 to 14.2) cannot compile `#include <mutex>`, because their
> `include-fixed/pthread.h` does not match the system headers.

`-march=x86-64-v3` (AVX2) is used instead of `native` because the login and compute nodes may differ. The
notebook expects the program at `~/src/lammps/build/lmp` on LRZ (`LRZ["lmp"]`).

## 12. Check the LRZ build

LAMMPS on LRZ starts only inside a Slurm job; on the login node even `lmp -h` aborts with
`PMI2_Job_GetId returned 14`. Run it as a short interactive job:

```bash
srun -p cm4_inter -n 1 -t 00:05:00 ~/src/lammps/build/lmp -h | grep -c bond/break/random     # prints 1
```

A small input built on this PC with `executor="lrz"` gave byte-identical dumps on LRZ (2 and 8 MPI
processes, partitions `cm4_inter` and `serial_long`) and on this PC with the same LAMMPS version
(2026-10-25). In `serial_long` (Xeon Platinum 8380, shared node) a process is about 0.68 × as fast as on this
PC: the 3 µm run takes about 35 h there instead of 24 h here.

## 13. Run on LRZ from the notebook

1. Open the connection: `ssh lrz` in a terminal (section 9).
2. In the parameter cell of `control.ipynb`, set `executor="lrz"` and `max_parallel=None`, so that all
   replicas run at once (a number limits the job array; `check()` warns about it).
3. As usual: `build_run` → `start` → `watch`.

`start()` copies the run folder to `~/qtft-lammps/runs/<group>/<run name>/` on LRZ (the same path below the
repository as here) and submits the job. While `watch`, `progress` or `summary` run, only the status files and
the new log lines come over (at most every 30 s): enough for the progress table and `quicklook`, without the
overlap panels. If the ssh connection is lost, `watch` stops with a notice and the run continues on LRZ.

When the run has finished, its results stay on LRZ: nothing is copied back by itself, and nothing runs in the
background on this PC. Download them with `fetch(run_dir)` (about 360 MB per replica of the 3 µm run).
Until then, `make_report`, `ensemble_stats`, the structure, cluster-volume and FIB-SEM analyses and
`binding_rate` stop with "download the results first with fetch(run_dir)".

<details>
<summary>Background: what the job script needs on LRZ</summary>

All of this is already in `job_template.slurm`:

| Setting | Why |
|---|---|
| `--clusters=serial --partition=serial_long --qos=cm4_serial_long` | the default QOS `cm4_serial` is refused ("Invalid qos specification") |
| `#!/bin/bash -l` | LRZ's base setup with the Intel MPI settings runs only in a login shell |
| `--export=NONE` and `export SLURM_EXPORT_ENV=ALL` | otherwise `srun` starts LAMMPS with an empty environment |
| `srun --mpi=pmi2` | the serial cluster starts no PMI by default |
| `--mem` (`slurm_mem`, 4G) | the default is 6.4 GB per CPU, i.e. 102 GB for 8 tasks with hyperthreads, the whole per-user limit; LAMMPS needs about 150 MB per process |

Per-user limits of `serial_long`: 1 node per job, 192 CPUs (hyperthreads: 8 MPI tasks count as 16) and about
100 GB in total, for example 10 replicas × 8 processes at once.

</details>

---

# Reference

## Running LAMMPS by hand

The notebooks start LAMMPS themselves. To run an input by hand:

```bash
conda activate lammps
mpirun -np <N> lmp -in in.myscript                              # N MPI processes
OMP_NUM_THREADS=2 mpirun -np 4 lmp -sf omp -in in.myscript      # MPI + OpenMP
```

- Keep simulation files in the Linux file system (`~/...`), not under `/mnt/c/...`: file access across the
  Windows boundary is very slow.
- Use at most as many MPI processes as physical cores (`lscpu`: "Core(s) per socket" × "Socket(s)", not
  "CPU(s)", which counts hyperthreads). Open MPI refuses more without `--use-hwthread-cpus`.
- To open results from Windows (for example in OVITO), enter `\\wsl$\Ubuntu-24.04\home\<Linux user>\` in
  the Explorer address bar.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `lmp: command not found` | `conda activate lammps` |
| `mpirun` uses the wrong MPI | `which mpirun` must be inside the conda environment; remove apt MPI packages |
| `mca_btl_openib … librdmacm.so.1: cannot open shared object file (ignored)` | harmless (no InfiniBand in WSL); to hide it: `export OMPI_MCA_btl=^openib` |
| CMake version too old | use the CMake of the environment (`conda activate lammps`) |
| `ImportError` or `liblammps.so` not found in Python | build with `BUILD_SHARED_LIBS=on`, then run the `install-python` target again |
| CMake error about conflicting files in `src/` | `cd ~/src/lammps/src && make no-all purge`, then configure again |
| `LAMMPS plugin not found` from `start()`, or `plugin load` fails | build the plugin (again after a LAMMPS update): `lammps_plugins/bond_break_random/build.sh` |
| notebook outputs appear in a commit | `nbstripout --install` in the repository (section 4) |
| `no connection to LRZ` from the notebook | the shared connection has expired (after 8 h) or was never opened: `ssh lrz` in a terminal |
| LRZ: `cannot convert '<brace-enclosed initializer list>' to 'unsigned int'` when building | `module load gcc/15.2.0` (section 11), delete `build/` and configure again |
| LRZ: `Invalid qos specification` from `sbatch` | `--qos=cm4_serial_long` for `serial_long` (in `job_template.slurm`) |
| LRZ: `PMI server not found` or `PMI2_Job_GetId returned 14` | LAMMPS was started outside Slurm, or `srun` without `--mpi=pmi2` |
| LRZ: `LAMMPS not found on LRZ` from `start()` | build LAMMPS on LRZ (section 11) |

## Updating LAMMPS

Update both installations together, so that runs on this PC and on LRZ use the same version.

1. On this PC:

   ```bash
   cd ~/src/lammps && git pull
   conda activate lammps
   cmake --build build -j $(nproc) && cmake --install build && cmake --build build --target install-python
   ~/qtft-lammps/lammps_plugins/bond_break_random/build.sh      # the plugin must match the new version
   git -C ~/src/lammps describe --tags                          # the new tag
   ```

2. On LRZ: delete `~/src/lammps` and repeat section 11 with the new tag.
