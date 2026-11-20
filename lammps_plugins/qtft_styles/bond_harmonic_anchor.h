/* -*- c++ -*- ----------------------------------------------------------
   bond style harmonic/anchor: harmonic bond whose ends are anchored in the body frames of oriented particles

     E = K (r - r0)^2 + K_qt (1 - cos theta_qt) + K_ft (1 - cos theta_ft)

   Syntax: bond_style harmonic/anchor
           bond_coeff N K r0 K_qt K_ft     (energy/distance^2, distance, energy, energy)

   theta_qt is the angle between the bond direction (from the Qt end to the Ft end) and the anchor of the Qt
   end, a unit vector fixed in that particle's body frame; theta_ft likewise for the Ft end (direction Ft -> Qt).
   The Qt end is the atom with the lower atom type (type 1 = Qt in qtft-lammps), the Ft end the other one;
   for equal types the lower atom ID takes the role of the Qt end.

   The anchors are set when a bond is first evaluated: the bond direction at that moment, expressed in each
   body frame (contact direction at binding). They are stored per bond in an internal fix BOND_HISTORY
   (ID HISTORY_HARMONIC_ANCHOR, 6 values: anchor of the Qt end, anchor of the Ft end), kept in binary restart
   files but not in data files (read_data starts with new anchors from the current geometry). Bonds that are
   removed by a fix must have their history shifted (fix bond/break/random does this).

   Forces and torques: F_ft = -2K(r - r0) b + K_qt (a_qt - cos theta_qt b)/r - K_ft (a_ft + cos theta_ft b)/r,
   F_qt = -F_ft, tau_qt = K_qt a_qt x b, tau_ft = K_ft a_ft x (-b), with b the unit bond vector Qt -> Ft and
   a_qt, a_ft the anchors in the lab frame; total force and angular momentum are conserved. No twist term.

   Needs atom style ellipsoid (e.g. hybrid bond ellipsoid; quaternion and torque per atom, every bonded atom
   an ellipsoid) and newton_bond on. single() returns the full energy (engpot of compute bond/local) with the
   radial force, and as extra values (compute bond/local b1 b2) theta_qt and theta_ft in rad.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(harmonic/anchor,BondHarmonicAnchor);
// clang-format on
#else

#ifndef LMP_BOND_HARMONIC_ANCHOR_H
#define LMP_BOND_HARMONIC_ANCHOR_H

#include "bond.h"

namespace LAMMPS_NS {

class BondHarmonicAnchor : public Bond {
 public:
  BondHarmonicAnchor(class LAMMPS *);
  ~BondHarmonicAnchor() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  void *extract(const char *, int &) override;

 protected:
  double *k, *r0, *k_qt, *k_ft;
  char *id_fix_dummy;              // placeholder fix, replaced by the history fix in init_style()
  class FixBondHistory *history;
  class AtomVecEllipsoid *avec;

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
