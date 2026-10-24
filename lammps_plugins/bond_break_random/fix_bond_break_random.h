/* -*- c++ -*- ----------------------------------------------------------
   fix bond/break/random: every bond of a type breaks independently with probability p per attempt

   Syntax: fix ID group bond/break/random Nevery btype prob seed
     prob = probability per attempt (0..1) or v_name of an equal-style variable, evaluated every attempt

   Unlike fix bond/break, there is no length criterion and no "longest bond" rule: every bond is tested
   once per attempt, and an atom can lose several bonds in one attempt. Output vector: [1] bonds broken
   in the last attempt, [2] bonds broken since the fix was defined. Based on fix bond/break (LAMMPS, MC
   package, stable 30 Sep 2026); the special lists (1-2, 1-3, 1-4) are updated the same way.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(bond/break/random,FixBondBreakRandom);
// clang-format on
#else

#ifndef LMP_FIX_BOND_BREAK_RANDOM_H
#define LMP_FIX_BOND_BREAK_RANDOM_H

#include "fix.h"

#include <vector>

namespace LAMMPS_NS {

class FixBondBreakRandom : public Fix {
 public:
  FixBondBreakRandom(class LAMMPS *, int, char **);
  ~FixBondBreakRandom() override;
  int setmask() override;
  void init() override;
  void post_integrate() override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;

 private:
  int me, nprocs;
  int btype, seed;
  double prob;          // constant probability (if no variable)
  char *pstr;           // name of the probability variable, or nullptr
  int pvar;             // index of that variable
  bigint lastcheck;

  bigint breakcount, breakcounttotal;

  std::vector<tagint> mine, broken;    // broken pairs (ID1, ID2, ...): on this proc, on all procs
  std::vector<int> counts, displs;
  tagint *copy;                        // special list of one atom during its rebuild

  class RanMars *random;

  double probability();
  void check_ghosts();
  void remove_special_12(int, tagint);
  void update_special();
  void rebuild_special_one(int);
  int dedup(int, int, tagint *);
};

}    // namespace LAMMPS_NS

#endif
#endif
