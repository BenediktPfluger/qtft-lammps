/* ----------------------------------------------------------------------
   fix bond/break/random, see fix_bond_break_random.h
------------------------------------------------------------------------- */

#include "fix_bond_break_random.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"
#include "variable.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondBreakRandom::FixBondBreakRandom(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), pstr(nullptr), copy(nullptr), random(nullptr)
{
  if (narg != 7) error->all(FLERR, "Illegal fix bond/break/random command: "
                            "fix ID group bond/break/random Nevery btype prob seed");

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Fix bond/break/random: Nevery must be > 0");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = utils::expand_type_int(FLERR, arg[4], Atom::BOND, lmp);
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR, "Fix bond/break/random: invalid bond type {}", arg[4]);

  prob = 0.0;
  if (utils::strmatch(arg[5], "^v_")) pstr = utils::strdup(arg[5] + 2);
  else {
    prob = utils::numeric(FLERR, arg[5], false, lmp);
    if (prob < 0.0 || prob > 1.0) error->all(FLERR, "Fix bond/break/random: prob must be in [0, 1]");
  }

  seed = utils::inumeric(FLERR, arg[6], false, lmp);
  if (seed <= 0) error->all(FLERR, "Fix bond/break/random: seed must be > 0");

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR, "Cannot use fix bond/break/random with non-molecular systems");

  // processor-unique seed, as in fix bond/break
  random = new RanMars(lmp, seed + me);

  // forward comm: 1-2 neighbours of an atom (count + IDs)
  comm_forward = 1 + atom->maxspecial;

  // special list of one atom while it is rebuilt: 1-2 neighbours, then their 1-2 neighbours (1-3),
  // then those of the 1-3 neighbours (1-4); with dedup in between, ms^2 + ms is enough (fix bond/break)
  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial * maxspecial + maxspecial];

  counts.resize(nprocs);
  displs.resize(nprocs);
  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondBreakRandom::~FixBondBreakRandom()
{
  delete random;
  delete[] pstr;
  delete[] copy;
}

/* ---------------------------------------------------------------------- */

int FixBondBreakRandom::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRandom::init()
{
  if (pstr) {
    pvar = input->variable->find(pstr);
    if (pvar < 0) error->all(FLERR, "Fix bond/break/random: variable {} does not exist", pstr);
    if (!input->variable->equalstyle(pvar))
      error->all(FLERR, "Fix bond/break/random: variable {} is not equal-style", pstr);
  }

  // every bond must be stored exactly once, so that it is tested exactly once
  if (!force->newton_bond) error->all(FLERR, "Fix bond/break/random requires newton_bond on");

  // only bonds; no angles, dihedrals or impropers to remove, no per-bond history
  if (atom->nangles || atom->ndihedrals || atom->nimpropers)
    error->all(FLERR, "Fix bond/break/random does not support angles, dihedrals or impropers");
  if (modify->get_fix_by_style("BOND_HISTORY").size())
    error->all(FLERR, "Fix bond/break/random does not support bond history");
  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix bond/break/random does not support run_style respa");

  lastcheck = -1;
}

/* ----------------------------------------------------------------------
   probability per attempt: the constant, or the variable evaluated now
   (an equal-style variable has the same value on every proc)
------------------------------------------------------------------------- */

double FixBondBreakRandom::probability()
{
  if (!pstr) return prob;
  double p = input->variable->compute_equal(pvar);
  if (p < 0.0 || p > 1.0)
    error->all(FLERR, "Fix bond/break/random: variable {} = {} is outside [0, 1]", pstr, p);
  return p;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRandom::post_integrate()
{
  if (update->ntimestep % nevery) return;

  breakcount = 0;
  double p = probability();
  if (p == 0.0) return;

  // the special lists of the 1-2 and 1-3 neighbours are needed to rebuild those of owned atoms

  if (lastcheck < neighbor->lastcall) check_ghosts();

  // test every bond stored with an owned atom once (newton_bond on: each bond is stored once);
  // a broken bond is removed from the bond list at once, so an atom can lose several bonds

  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;

  mine.clear();
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int m = 0;
    while (m < num_bond[i]) {
      if (bond_type[i][m] == btype) {
        tagint jtag = bond_atom[i][m];
        int j = atom->map(jtag);
        if (j < 0)
          error->one(FLERR, "Fix bond/break/random: bond partner {} of atom {} missing", jtag, tag[i]);
        if ((mask[j] & groupbit) && random->uniform() < p) {
          for (int k = m; k < num_bond[i] - 1; k++) {
            bond_atom[i][k] = bond_atom[i][k + 1];
            bond_type[i][k] = bond_type[i][k + 1];
          }
          num_bond[i]--;
          mine.push_back(tag[i]);
          mine.push_back(jtag);
          continue;
        }
      }
      m++;
    }
  }

  // all broken pairs on every proc

  int nmine = mine.size();
  MPI_Allgather(&nmine, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  int nall = 0;
  for (int k = 0; k < nprocs; k++) {
    displs[k] = nall;
    nall += counts[k];
  }
  breakcount = nall / 2;
  if (!breakcount) return;

  broken.resize(nall);
  MPI_Allgatherv(mine.data(), nmine, MPI_LMP_TAGINT, broken.data(), counts.data(), displs.data(),
                 MPI_LMP_TAGINT, world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  // the neighbour lists must reflect the new topology at once (repulsion of the pair back on)

  next_reneighbor = update->ntimestep;

  update_special();
}

/* ----------------------------------------------------------------------
   update the special lists after the bonds in `broken` were removed
   1. both atoms of a broken bond (if owned) drop each other from their 1-2 neighbours
   2. ghosts get the new 1-2 lists
   3. every owned atom influenced by a broken bond rebuilds its 1-3 and 1-4 neighbours
------------------------------------------------------------------------- */

void FixBondBreakRandom::update_special()
{
  int nlocal = atom->nlocal;
  int npairs = broken.size() / 2;

  for (int k = 0; k < npairs; k++) {
    tagint id1 = broken[2 * k], id2 = broken[2 * k + 1];
    int i = atom->map(id1);
    if (i >= 0 && i < nlocal) remove_special_12(i, id2);
    i = atom->map(id2);
    if (i >= 0 && i < nlocal) remove_special_12(i, id1);
  }

  comm->forward_comm(this);

  // influenced: one of the two atoms, or both atoms in the special list (1-3, 1-4 via this bond)

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  for (int i = 0; i < nlocal; i++) {
    tagint *slist = special[i];
    int n3 = nspecial[i][2];
    int influenced = 0;
    for (int k = 0; k < npairs && !influenced; k++) {
      tagint id1 = broken[2 * k], id2 = broken[2 * k + 1];
      if (tag[i] == id1 || tag[i] == id2) influenced = 1;
      else {
        int found = 0;
        for (int m = 0; m < n3; m++)
          if (slist[m] == id1 || slist[m] == id2) found++;
        if (found == 2) influenced = 1;
      }
    }
    if (influenced) rebuild_special_one(i);
  }
}

/* ----------------------------------------------------------------------
   remove partner from the 1-2 neighbours of owned atom i (1-3, 1-4 shift down)
------------------------------------------------------------------------- */

void FixBondBreakRandom::remove_special_12(int i, tagint partner)
{
  int **nspecial = atom->nspecial;
  tagint *slist = atom->special[i];
  int n1 = nspecial[i][0];
  int m = 0;
  while (m < n1 && slist[m] != partner) m++;
  if (m == n1)
    error->one(FLERR, "Fix bond/break/random: atom {} is not a 1-2 neighbour of atom {}", partner,
               atom->tag[i]);
  int n3 = nspecial[i][2];
  for (; m < n3 - 1; m++) slist[m] = slist[m + 1];
  nspecial[i][0]--;
  nspecial[i][1]--;
  nspecial[i][2]--;
}

/* ----------------------------------------------------------------------
   all 1-2 and 1-3 neighbours of owned atoms must be known (owned or ghost),
   so that their 1-2 lists can be used to rebuild the 1-3 and 1-4 lists (as in fix bond/break)
------------------------------------------------------------------------- */

void FixBondBreakRandom::check_ghosts()
{
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    for (int j = 0; j < nspecial[i][1]; j++)
      if (atom->map(special[i][j]) < 0) flag = 1;

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
  if (flagall) error->all(FLERR, "Fix bond/break/random needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   rebuild the 1-3 and 1-4 neighbours of owned atom m from its 1-2 neighbours
   (as fix bond/break and fix bond/create do)
------------------------------------------------------------------------- */

void FixBondBreakRandom::rebuild_special_one(int m)
{
  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // 1-2 neighbours of atom m

  tagint *slist = special[m];
  int cn1 = 0;
  for (int i = 0; i < nspecial[m][0]; i++) copy[cn1++] = slist[i];

  // 1-3: 1-2 neighbours of the 1-2 neighbours, without atom m and duplicates

  int cn2 = cn1;
  for (int i = 0; i < cn1; i++) {
    int n = atom->map(copy[i]);
    if (n < 0) error->one(FLERR, "Fix bond/break/random needs ghost atoms from further away");
    slist = special[n];
    for (int j = 0; j < nspecial[n][0]; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }
  cn2 = dedup(cn1, cn2, copy);
  if (cn2 > atom->maxspecial) error->one(FLERR, "Special list size exceeded in fix bond/break/random");

  // 1-4: 1-2 neighbours of the 1-3 neighbours, without atom m and duplicates

  int cn3 = cn2;
  for (int i = cn1; i < cn2; i++) {
    int n = atom->map(copy[i]);
    if (n < 0) error->one(FLERR, "Fix bond/break/random needs ghost atoms from further away");
    slist = special[n];
    for (int j = 0; j < nspecial[n][0]; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }
  cn3 = dedup(cn2, cn3, copy);
  if (cn3 > atom->maxspecial) error->one(FLERR, "Special list size exceeded in fix bond/break/random");

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m], copy, cn3 * sizeof(tagint));
}

/* ----------------------------------------------------------------------
   remove IDs in copy[nstart:nstop] that occur earlier in copy; return the new nstop
------------------------------------------------------------------------- */

int FixBondBreakRandom::dedup(int nstart, int nstop, tagint *copy)
{
  int m = nstart;
  while (m < nstop) {
    int i;
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop - 1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }
  return nstop;
}

/* ---------------------------------------------------------------------- */

int FixBondBreakRandom::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                          int * /*pbc*/)
{
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    int ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (int k = 0; k < ns; k++) buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondBreakRandom::unpack_forward_comm(int n, int first, double *buf)
{
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    int ns = (int) ubuf(buf[m++]).i;
    nspecial[i][0] = ns;
    for (int j = 0; j < ns; j++) special[i][j] = (tagint) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

double FixBondBreakRandom::compute_vector(int n)
{
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}
