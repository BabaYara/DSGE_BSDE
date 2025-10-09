import Lake
open Lake DSL

package LucasNotes {}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.20.0"

require checkdecls from git
  "https://github.com/PatrickMassot/checkdecls.git"

@[default_target]
lean_lib LucasNotes {}
