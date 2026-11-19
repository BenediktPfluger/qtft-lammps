// Plugin qtftstyles: registers the LAMMPS styles of qtft-lammps

#include "lammpsplugin.h"

#include "version.h"

#include "bond_harmonic_anchor.h"
#include "fix_bond_break_random.h"

using namespace LAMMPS_NS;

static Fix *fix_bond_break_random_creator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixBondBreakRandom(lmp, argc, argv);
}

static Bond *bond_harmonic_anchor_creator(LAMMPS *lmp)
{
  return new BondHarmonicAnchor(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  auto register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "bond/break/random";
  plugin.info = "fix bond/break/random: every bond breaks independently with probability p (qtft-lammps)";
  plugin.author = "Benedikt Pfluger";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &fix_bond_break_random_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "bond";
  plugin.name = "harmonic/anchor";
  plugin.info = "bond harmonic/anchor: harmonic bond with ends anchored in the body frames (qtft-lammps)";
  plugin.author = "Benedikt Pfluger";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &bond_harmonic_anchor_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
