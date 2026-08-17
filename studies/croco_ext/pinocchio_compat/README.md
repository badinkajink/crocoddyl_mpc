Shim headers for building against this conda prefix.

crocoddyl 3.2.1's `crocoddyl/{core,multibody}/pch.hpp` -- pulled in unconditionally
by `crocoddyl/multibody/fwd.hpp` -- includes six Pinocchio headers that Pinocchio
4.1.0 no longer installs at those paths:

    pinocchio/multibody/data.hpp     pinocchio/spatial/force.hpp
    pinocchio/multibody/model.hpp    pinocchio/spatial/motion.hpp
    pinocchio/spatial/se3.hpp
    pinocchio/multibody/fcl.hpp      pinocchio/multibody/geometry.hpp   (hpp-fcl only)

Their contents moved behind the umbrella headers `pinocchio/multibody.hpp` and
`pinocchio/spatial.hpp`.  crocoddyl's own library was built before the move, so
nothing in the installed prefix notices -- but any translation unit that includes
a crocoddyl *multibody* header (which `croco_ext/keepout.cpp` must, to reach
`StateMultibody` and `DataCollectorMultibody`) fails on the first missing file.

These four files forward to the umbrella headers.  The directory goes LAST on the
include path so it shadows nothing that the prefix actually ships.
