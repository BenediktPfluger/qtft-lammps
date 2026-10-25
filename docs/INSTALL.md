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

This project needs the packages **BROWNIAN, EXTRA-PAIR, MOLECULE and MC**
(all included in the `most.cmake` preset). Check them with:

```bash
lmp -h | grep -A 30 "Installed packages" | grep -o -w -E "BROWNIAN|EXTRA-PAIR|MOLECULE|MC" | sort -u
```

All four names must be printed.

## 7. Project setup (once, after cloning)

```bash
cd ~/projects/qtft-lammps
conda activate lammps
nbstripout --install      # strips notebook outputs automatically on every commit
lammps_plugins/bond_break_random/build.sh   # unbinding plugin fix bond/break/random (needs ~/src/lammps)
pytest -q                 # self-test incl. short LAMMPS runs (~2.5 min); -m "not slow": without them (~2 s)
```

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

## Updating LAMMPS

```bash
cd ~/src/lammps && git pull
conda activate lammps
cmake --build build -j $(nproc) && cmake --install build && cmake --build build --target install-python
~/projects/qtft-lammps/lammps_plugins/bond_break_random/build.sh   # the plugin must match the new version
```
