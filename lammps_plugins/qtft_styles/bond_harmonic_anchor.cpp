/* ----------------------------------------------------------------------
   bond style harmonic/anchor, see bond_harmonic_anchor.h
------------------------------------------------------------------------- */

#include "bond_harmonic_anchor.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static constexpr char ID_HISTORY[] = "HISTORY_HARMONIC_ANCHOR";
static constexpr char ID_DUMMY[] = "HARMONIC_ANCHOR_DUMMY";
static constexpr int NHISTORY = 6;    // anchor of the Qt end (body frame), anchor of the Ft end (body frame)

/* ---------------------------------------------------------------------- */

BondHarmonicAnchor::BondHarmonicAnchor(LAMMPS *_lmp) :
    Bond(_lmp), k(nullptr), r0(nullptr), k_qt(nullptr), k_ft(nullptr), id_fix_dummy(nullptr),
    history(nullptr), avec(nullptr)
{
  // the history fix has a fixed ID (restart files find their per-atom data by it), so only one instance
  if (modify->get_fix_by_id(ID_DUMMY) || modify->get_fix_by_id(ID_HISTORY))
    error->all(FLERR, "Bond style harmonic/anchor can be used only once (also not twice in bond_style hybrid)");

  // placeholder at this position in the fix list; init_style() replaces it by the history fix, as BondBPM does
  id_fix_dummy = utils::strdup(ID_DUMMY);
  modify->add_fix(fmt::format("{} all DUMMY", id_fix_dummy));
}

/* ---------------------------------------------------------------------- */

BondHarmonicAnchor::~BondHarmonicAnchor()
{
  if (id_fix_dummy && modify->get_fix_by_id(id_fix_dummy)) modify->delete_fix(id_fix_dummy);
  if (history && modify->get_fix_by_id(ID_HISTORY)) modify->delete_fix(ID_HISTORY);
  delete[] id_fix_dummy;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(k_qt);
    memory->destroy(k_ft);
  }
}

/* ---------------------------------------------------------------------- */

void BondHarmonicAnchor::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
  int *type = atom->type;
  int *ellipsoid = atom->ellipsoid;
  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;

  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double **store = history->bondstore;

  for (int n = 0; n < nbondlist; n++) {
    int i1 = bondlist[n][0];    // with newton_bond on: the owned atom that stores the bond
    int i2 = bondlist[n][1];
    int btype = bondlist[n][2];
    if (btype <= 0) continue;    // bond broken in this step

    // q = Qt end (lower type, or lower ID for equal types), t = Ft end
    int q = i1, t = i2;
    if (type[i2] < type[i1] || (type[i2] == type[i1] && tag[i2] < tag[i1])) {
      q = i2;
      t = i1;
    }
    if (ellipsoid[q] < 0 || ellipsoid[t] < 0)
      error->one(FLERR, "Bond harmonic/anchor: atom {} or {} is not an ellipsoid", tag[i1], tag[i2]);

    double rot_q[3][3], rot_t[3][3];
    MathExtra::quat_to_mat(bonus[ellipsoid[q]].quat, rot_q);
    MathExtra::quat_to_mat(bonus[ellipsoid[t]].quat, rot_t);

    double del[3] = {x[t][0] - x[q][0], x[t][1] - x[q][1], x[t][2] - x[q][2]};    // Qt -> Ft
    double r = sqrt(MathExtra::lensq3(del));
    if (r == 0.0) error->one(FLERR, "Bond harmonic/anchor: atoms {} and {} coincide", tag[i1], tag[i2]);
    double b[3] = {del[0] / r, del[1] / r, del[2] / r};
    double mb[3] = {-b[0], -b[1], -b[2]};

    // new bond (history slot still zero): anchors = current bond direction in each body frame; written to
    // the per-atom history of the storing atom, so that they move with it and survive re-neighbouring
    double *s = store[n];
    if (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] < 0.25) {
      MathExtra::transpose_matvec(rot_q, b, &s[0]);
      MathExtra::transpose_matvec(rot_t, mb, &s[3]);
      tagint partner = tag[i2];
      int m = 0;
      while (m < num_bond[i1] && !(bond_atom[i1][m] == partner && atom->bond_type[i1][m] == btype)) m++;
      if (m == num_bond[i1])
        error->one(FLERR, "Bond harmonic/anchor: bond {}-{} not found with atom {}", tag[i1], partner, tag[i1]);
      for (int d = 0; d < NHISTORY; d++) history->update_atom_value(i1, m, d, s[d]);
    }

    double a_q[3], a_t[3];    // anchors in the lab frame
    MathExtra::matvec(rot_q, &s[0], a_q);
    MathExtra::matvec(rot_t, &s[3], a_t);
    double cos_q = MathExtra::dot3(a_q, b);
    double cos_t = -MathExtra::dot3(a_t, b);

    double dr = r - r0[btype];
    double f_t[3];    // force on the Ft end
    for (int d = 0; d < 3; d++)
      f_t[d] = -2.0 * k[btype] * dr * b[d] + k_qt[btype] * (a_q[d] - cos_q * b[d]) / r -
          k_ft[btype] * (a_t[d] + cos_t * b[d]) / r;
    double tau_q[3], tau_t[3];
    MathExtra::cross3(a_q, b, tau_q);
    MathExtra::cross3(a_t, mb, tau_t);

    if (newton_bond || q < nlocal) {
      for (int d = 0; d < 3; d++) {
        f[q][d] -= f_t[d];
        torque[q][d] += k_qt[btype] * tau_q[d];
      }
    }
    if (newton_bond || t < nlocal) {
      for (int d = 0; d < 3; d++) {
        f[t][d] += f_t[d];
        torque[t][d] += k_ft[btype] * tau_t[d];
      }
    }

    if (evflag) {
      double ebond = 0.0;
      if (eflag) ebond = k[btype] * dr * dr + k_qt[btype] * (1.0 - cos_q) + k_ft[btype] * (1.0 - cos_t);
      // ev_tally_xyz takes the force on i1 and del = x[i1] - x[i2]
      double sign = (i1 == q) ? -1.0 : 1.0;
      ev_tally_xyz(i1, i2, nlocal, newton_bond, ebond, sign * f_t[0], sign * f_t[1], sign * f_t[2],
                   x[i1][0] - x[i2][0], x[i1][1] - x[i2][1], x[i1][2] - x[i2][2]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void BondHarmonicAnchor::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;
  memory->create(k, np1, "bond:k");
  memory->create(r0, np1, "bond:r0");
  memory->create(k_qt, np1, "bond:k_qt");
  memory->create(k_ft, np1, "bond:k_ft");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ---------------------------------------------------------------------- */

void BondHarmonicAnchor::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR, "Incorrect args for bond coefficients: bond_coeff N K r0 K_qt K_ft");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);
  double k_one = utils::numeric(FLERR, arg[1], false, lmp);
  double r0_one = utils::numeric(FLERR, arg[2], false, lmp);
  double kq_one = utils::numeric(FLERR, arg[3], false, lmp);
  double kf_one = utils::numeric(FLERR, arg[4], false, lmp);
  if (k_one < 0.0 || kq_one < 0.0 || kf_one < 0.0)
    error->all(FLERR, "Bond harmonic/anchor: K, K_qt and K_ft must be >= 0");
  if (r0_one <= 0.0) error->all(FLERR, "Bond harmonic/anchor: r0 must be > 0");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    r0[i] = r0_one;
    k_qt[i] = kq_one;
    k_ft[i] = kf_one;
    setflag[i] = 1;
    count++;
  }
  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ---------------------------------------------------------------------- */

void BondHarmonicAnchor::init_style()
{
  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR, "Bond style harmonic/anchor requires atom style ellipsoid");
  if (!atom->torque_flag) error->all(FLERR, "Bond style harmonic/anchor requires per-atom torque");
  // every bond stored and computed once; the anchors are written to the atom that stores the bond
  if (!force->newton_bond) error->all(FLERR, "Bond style harmonic/anchor requires newton_bond on");

  if (!history) {
    auto *newfix = modify->replace_fix(id_fix_dummy,
                                       fmt::format("{} all BOND_HISTORY 0 {}", ID_HISTORY, NHISTORY), 1);
    history = dynamic_cast<FixBondHistory *>(newfix);
    if (!history) error->all(FLERR, "Bond style harmonic/anchor: could not create the bond history fix");
    delete[] id_fix_dummy;
    id_fix_dummy = nullptr;
  }
  history->setflag = setflag;
}

/* ---------------------------------------------------------------------- */

double BondHarmonicAnchor::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes the coefficients to a restart file (the anchors are in the history fix)
------------------------------------------------------------------------- */

void BondHarmonicAnchor::write_restart(FILE *fp)
{
  const int n = atom->nbondtypes;
  fwrite(&k[1], sizeof(double), n, fp);
  fwrite(&r0[1], sizeof(double), n, fp);
  fwrite(&k_qt[1], sizeof(double), n, fp);
  fwrite(&k_ft[1], sizeof(double), n, fp);
}

/* ---------------------------------------------------------------------- */

void BondHarmonicAnchor::read_restart(FILE *fp)
{
  allocate();
  const int n = atom->nbondtypes;
  if (comm->me == 0) {
    utils::sfread(FLERR, &k[1], sizeof(double), n, fp, nullptr, error);
    utils::sfread(FLERR, &r0[1], sizeof(double), n, fp, nullptr, error);
    utils::sfread(FLERR, &k_qt[1], sizeof(double), n, fp, nullptr, error);
    utils::sfread(FLERR, &k_ft[1], sizeof(double), n, fp, nullptr, error);
  }
  MPI_Bcast(&k[1], n, MPI_DOUBLE, 0, world);
  MPI_Bcast(&r0[1], n, MPI_DOUBLE, 0, world);
  MPI_Bcast(&k_qt[1], n, MPI_DOUBLE, 0, world);
  MPI_Bcast(&k_ft[1], n, MPI_DOUBLE, 0, world);
  for (int i = 1; i <= n; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes the Bond Coeffs section of a data file
------------------------------------------------------------------------- */

void BondHarmonicAnchor::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp, "%d %.15g %.15g %.15g %.15g\n", i, k[i], r0[i], k_qt[i], k_ft[i]);
}

/* ----------------------------------------------------------------------
   radial part only: the bending terms need the orientations, which single() does not get
------------------------------------------------------------------------- */

double BondHarmonicAnchor::single(int type, double rsq, int /*i*/, int /*j*/, double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  fforce = (r > 0.0) ? -2.0 * k[type] * dr / r : 0.0;
  return k[type] * dr * dr;
}

/* ---------------------------------------------------------------------- */

void *BondHarmonicAnchor::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "k") == 0) return (void *) k;
  if (strcmp(str, "r0") == 0) return (void *) r0;
  if (strcmp(str, "k_qt") == 0) return (void *) k_qt;
  if (strcmp(str, "k_ft") == 0) return (void *) k_ft;
  return nullptr;
}
