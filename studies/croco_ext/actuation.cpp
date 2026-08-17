// The plant's passive joint torques as a native crocoddyl actuation model.
//
// WHY.  croco_plan._make_actuation is a Python subclass of
// ActuationModelAbstract carrying the MJCF's joint damping and friction loss:
//
//     tau = [0; u - b v_j - f tanh(v_j / eps)]
//
// It is correct (croco_bridge checks it against MuJoCo's own passive torques)
// and it is the last interpreted object left in the hot path.  Measured on the
// braced node (croco_speed.py pieces): calc 2.11 us and calcDiff 5.01 us against
// 0.17 / 0.17 for crocoddyl's stock ActuationModelFloatingBase -- the difference
// is a Python round-trip plus, in calcDiff, a fresh 33x66 `np.zeros` and an
// `np.diag` allocated per node per sweep.
//
// That is ~7 us of a 142 us node, so this is a 4-5% change to the step and not a
// headline.  It is here for two other reasons:
//
//  1. calc is on the LINE-SEARCH path.  FDDP rolls the whole horizon out once per
//     trial step and the measured median is ~4 trials per iteration, so the 2.11
//     us shows up four times per node per iteration rather than once.
//  2. It is the blocker for ever using `ShootingProblem.nthreads`.  This
//     conda-forge crocoddyl is built without OpenMP so the knob is inert today,
//     but a build with it would parallelise calc/calcDiff across nodes -- and a
//     Python actuation model in every node means every worker thread contending
//     for the GIL, which is not a speed-up but a deadlock risk.  With this, no
//     node holds an interpreter object.
//
// The maths is transcribed from croco_plan._make_actuation and
// croco_ext/test_passive.py checks the two agree on tau, dtau_dx and dtau_du.
//
// build:  croco_ext/build.sh passive

#include <pinocchio/fwd.hpp>

#include <boost/python.hpp>

#include <Eigen/Dense>
#include <memory>
#include <sstream>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/state-base.hpp"

#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;

namespace {

typedef crocoddyl::ActuationModelAbstractTpl<double> Base;
typedef crocoddyl::ActuationDataAbstractTpl<double> BaseData;
typedef crocoddyl::StateAbstractTpl<double> State;

class ActuationModelJointPassive : public Base {
 public:
  ActuationModelJointPassive(std::shared_ptr<State> state,
                             const Eigen::VectorXd& damping,
                             const Eigen::VectorXd& friction, double eps)
      : Base(state, state->get_nv() - 6),
        b_(damping),
        f_(friction),
        eps_(eps),
        nv_(state->get_nv()),
        nq_(state->get_nq()) {
    if (b_.size() != static_cast<Eigen::Index>(nu_) ||
        f_.size() != static_cast<Eigen::Index>(nu_)) {
      throw_pretty("Invalid argument: damping and friction must have nv - 6 = " +
                   std::to_string(nu_) + " entries");
    }
  }

  void calc(const std::shared_ptr<BaseData>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override {
    // v_j are the ACTUATED joint velocities: x is [q (nq); v (nv)] and the first
    // six of v are the floating base.
    const auto vj = x.segment(nq_ + 6, nu_);
    data->tau.head(6).setZero();
    data->tau.tail(nu_) = u.array() - b_.array() * vj.array() -
                          f_.array() * (vj.array() / eps_).tanh();
  }

  void calcDiff(const std::shared_ptr<BaseData>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>&) override {
    // Only the joint-velocity block is state dependent, and dtau_du is constant
    // -- both the zeros around this block and dtau_du itself were written once
    // in createData and are never disturbed, which is where most of the saving
    // over the Python model is (it rebuilt the whole 33x66 every call).
    const auto vj = x.segment(nq_ + 6, nu_);
    const Eigen::ArrayXd th = (vj.array() / eps_).tanh();
    data->dtau_dx.block(6, nv_ + 6, nu_, nu_).diagonal() =
        -(b_.array() + f_.array() * (1.0 - th.square()) / eps_);
  }

  void commands(const std::shared_ptr<BaseData>& data,
                const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>& tau) override {
    data->u = tau.tail(nu_);
  }

  void torqueTransform(const std::shared_ptr<BaseData>& data,
                       const Eigen::Ref<const Eigen::VectorXd>&,
                       const Eigen::Ref<const Eigen::VectorXd>&) override {
    data->Mtau = Mtau_;
  }

  std::shared_ptr<BaseData> createData() override {
    std::shared_ptr<BaseData> data =
        std::allocate_shared<BaseData>(Eigen::aligned_allocator<BaseData>(),
                                       this);
    // The constant structure, written once: the selection matrix on the
    // actuated rows, its transpose as the torque transform, and the fact that
    // the floating-base rows are unactuated.
    data->dtau_du.diagonal(-6).setOnes();
    data->Mtau.diagonal(6).setOnes();
    for (std::size_t k = 0; k < 6; ++k) data->tau_set[k] = false;
    Mtau_ = data->Mtau;
    return data;
  }

  std::shared_ptr<crocoddyl::ActuationModelBase> cloneAsDouble() const override {
    return std::make_shared<ActuationModelJointPassive>(state_, b_, f_, eps_);
  }
  std::shared_ptr<crocoddyl::ActuationModelBase> cloneAsFloat() const override {
    return cloneAsDouble();
  }

  void print(std::ostream& os) const override {
    os << "ActuationModelJointPassive {nu=" << nu_ << ", eps=" << eps_ << "}";
  }

 private:
  Eigen::VectorXd b_;
  Eigen::VectorXd f_;
  double eps_;
  std::size_t nv_;
  std::size_t nq_;
  mutable Eigen::MatrixXd Mtau_;
};

// calc / calcDiff / commands / createData are re-exported below rather than
// inherited: crocoddyl registers them on its Python *wrapper* subclass, not on
// the abstract base, so a plain C++ derivative inherits the base in the bp
// registry but none of its methods (the same trap keepout.cpp documents).  The
// solver drives these through the C++ virtual and never notices; the test does.
// The thunks take VectorXd by const reference because eigenpy's converter for a
// const Eigen::Ref hands boost::python a temporary whose storage is already gone
// at call time -- a stack-smash abort, not a type error.
void py_calc(ActuationModelJointPassive& a, const std::shared_ptr<BaseData>& d,
             const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
  a.calc(d, x, u);
}

void py_calc_diff(ActuationModelJointPassive& a,
                  const std::shared_ptr<BaseData>& d, const Eigen::VectorXd& x,
                  const Eigen::VectorXd& u) {
  a.calcDiff(d, x, u);
}

void py_commands(ActuationModelJointPassive& a,
                 const std::shared_ptr<BaseData>& d, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& tau) {
  a.commands(d, x, tau);
}

std::string py_repr(const ActuationModelJointPassive& a) {
  std::ostringstream os;
  a.print(os);
  return os.str();
}

}  // namespace

BOOST_PYTHON_MODULE(croco_passive) {
  bp::import("crocoddyl");
  // eigenpy's converters have to be live for the VectorXd constructor
  // arguments; unlike croco_keepout this module cannot avoid Eigen at the
  // boundary, because the damping and friction are 27-vectors read off the MJCF.
  eigenpy::enableEigenPy();

  bp::class_<ActuationModelJointPassive, bp::bases<Base>,
             std::shared_ptr<ActuationModelJointPassive>, boost::noncopyable>(
      "ActuationModelJointPassive",
      "Floating-base actuation carrying the plant's passive joint torques:\n"
      "    tau = [0; u - b v_j - f tanh(v_j / eps)]\n"
      "Transcribed from croco_plan._make_actuation; test_passive.py checks it.",
      bp::init<std::shared_ptr<State>, Eigen::VectorXd, Eigen::VectorXd, double>(
          bp::args("self", "state", "damping", "friction", "eps")))
      .def("calc", &py_calc, bp::args("self", "data", "x", "u"))
      .def("calcDiff", &py_calc_diff, bp::args("self", "data", "x", "u"))
      .def("commands", &py_commands, bp::args("self", "data", "x", "tau"))
      .def("createData", &ActuationModelJointPassive::createData,
           bp::args("self"))
      .def("__repr__", &py_repr);
}
