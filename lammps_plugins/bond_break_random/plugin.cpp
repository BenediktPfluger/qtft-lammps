#include "lammpsplugin.h"

#include "version.h"

#include "fix_bond_break_random.h"

using namespace LAMMPS_NS;

static Fix *bondbreakrandom_creator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixBondBreakRandom(lmp, argc, argv);
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
  plugin.creator.v2 = (lammpsplugin_factory2 *) &bondbreakrandom_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
