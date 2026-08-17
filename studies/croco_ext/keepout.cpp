// Box keep-out activation for crocoddyl, in C++.
//
// WHY.  The keep-out is the OCP's only geometric constraint on the table
// (croco_geom.py explains why a half-space cannot be it), and it is written as
// one activation per sampled point per node.  The S13 plan carries 86 points
// over 290 nodes; as a Python subclass of `crocoddyl.ActivationModelAbstract`
// that is 24 940 interpreter round-trips per calc/calcDiff sweep, and the
// measured cost was the dominant term in both the offline solve (50 s) and the
// receding-horizon MPC (119 ms/step against a 20 ms control period).
//
// The maths is unchanged from croco_geom.ActivationModelBoxKeepOut -- this is a
// transcription, and `croco_ext/test_keepout.py` checks the two agree to 1e-12
// on random residuals, including on the branch boundaries.
//
// ONE THING WORTH KNOWING about the Hessian.  crocoddyl stores `Arr` as an
// Eigen DiagonalMatrix, and the Python binding's setter is
//     data.Arr.diagonal() = Arr.diagonal()
// so the Python model's `data.Arr = np.outer(g, g)` was only ever contributing
// diag(g_i^2) -- the off-diagonal half of the Gauss-Newton term was silently
// discarded by the binding.  This C++ version writes diag(g_i^2) explicitly, so
// it reproduces the Python behaviour exactly rather than "fixing" it into a
// different optimisation problem.
//
// build:  croco_ext/build.sh      (writes croco_keepout*.so next to this file)

// ---------------------------------------------------------------------------
// S15 ADDENDUM: the activation was the wrong axis.
//
// Making each point's activation native took the S13 solve from 50 s to 14 s and
// the MPC step from 119 ms to 76 ms, and then stopped, because what a keep-out
// point costs is only half its own arithmetic.  Measured (croco_speed.py
// scaling): a cost term that computes NOTHING -- a 1-row control residual --
// costs 1.61 us per node, because CostModelSum::calcDiff unconditionally does
//     Lx += w*Lx_i;  Lxx += w*Lxx_i;  Lxu += w*Lxu_i;  Luu += w*Luu_i
// which for n x = 66, n u = 27 is 6960 doubles of dense accumulation per term,
// over a private CostDataAbstract that holds its own copy of all four.  A
// keep-out point costs 2.20 us, so 73% of it is bookkeeping over a residual that
// is zero at almost every point at almost every node.
//
// 86 points x 50 nodes of that is a 5.4 MB-per-node working set and ~184 us per
// node of accumulation, and it is why an in-situ calcDiff sweep costs 550 us per
// node against the 323 us the same node measures in isolation: the isolated
// benchmark is cache-warm and the sweep is not.
//
// So `CostModelBoxKeepOut` below fuses all 86 points into ONE cost term:
//   * one accumulation into the shared Lxx instead of 86;
//   * one CostDataAbstract instead of 86;
//   * and, because the term now owns its own loop, the points whose activation
//     is zero cost one SDF evaluation (~20 flops) instead of a frame Jacobian
//     and a 33x33 Gauss-Newton block.  Typically 0-3 of the 86 are active.
//
// The maths is identical to the per-point stack, including the diag(g^2)
// Hessian discussed below -- `test_keepout.py` checks the fused cost against a
// CostModelSum of per-point costs on the real states, cost, Lx and Lxx alike.

#include <pinocchio/fwd.hpp>

#include <boost/python.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace bp = boost::python;

namespace {

typedef crocoddyl::ActivationModelAbstractTpl<double> Base;
typedef crocoddyl::ActivationDataAbstractTpl<double> BaseData;

// Signed distance from an already box-centred point to an axis-aligned box.
inline double sdfBox(const Eigen::Vector3d& p, const Eigen::Vector3d& half) {
  const Eigen::Vector3d a = p.cwiseAbs() - half;
  const Eigen::Vector3d amax = a.cwiseMax(0.0);
  return amax.norm() + std::min(a.maxCoeff(), 0.0);
}

// d(sdf)/dp.  Outside the box it is the normalised outward part; inside, the
// axis of least penetration.  Both branches match croco_geom.sdf_box_grad.
inline Eigen::Vector3d sdfBoxGrad(const Eigen::Vector3d& p,
                                  const Eigen::Vector3d& half) {
  const Eigen::Vector3d a = p.cwiseAbs() - half;
  Eigen::Vector3d sg;
  for (int i = 0; i < 3; ++i) sg[i] = (p[i] < 0.0) ? -1.0 : 1.0;
  if (a.maxCoeff() > 0.0) {
    const Eigen::Vector3d amax = a.cwiseMax(0.0);
    const double n = amax.norm();
    return sg.cwiseProduct(amax) / std::max(n, 1e-12);
  }
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
  int i = 0;
  a.maxCoeff(&i);
  g[i] = sg[i];
  return g;
}

class ActivationModelBoxKeepOut : public Base {
 public:
  ActivationModelBoxKeepOut(double hx, double hy, double hz, double r_min)
      : Base(3), half_(hx, hy, hz), r_min_(r_min) {}

  void calc(const std::shared_ptr<BaseData>& data,
            const Eigen::Ref<const Eigen::VectorXd>& r) override {
    const double v = r_min_ - sdfBox(r.head<3>(), half_);
    data->a_value = (v > 0.0) ? 0.5 * v * v : 0.0;
  }

  void calcDiff(const std::shared_ptr<BaseData>& data,
                const Eigen::Ref<const Eigen::VectorXd>& r) override {
    const Eigen::Vector3d p = r.head<3>();
    const double v = r_min_ - sdfBox(p, half_);
    if (v <= 0.0) {
      data->Ar.setZero();
      data->Arr.diagonal().setZero();
      return;
    }
    const Eigen::Vector3d g = sdfBoxGrad(p, half_);
    data->Ar = -v * g;
    data->Arr.diagonal() = g.array().square();
  }

  // Required by CROCODDYL_BASE_CAST.  Only the double instantiation is ever
  // used here; the float scalar exists for the library's codegen paths, which
  // this study does not use, so it clones as double rather than pretending to
  // support a scalar the SDF branches were never checked in.  (The macro also
  // declares cloneAsADDouble, but only in a CppAD-enabled build; this
  // conda-forge crocoddyl is not one, so declaring it here fails to override.)
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsDouble() const override {
    return std::make_shared<ActivationModelBoxKeepOut>(half_[0], half_[1],
                                                       half_[2], r_min_);
  }
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsFloat() const override {
    return cloneAsDouble();
  }

  void print(std::ostream& os) const override {
    os << "ActivationModelBoxKeepOut {half=[" << half_.transpose()
       << "], r_min=" << r_min_ << "}";
  }

  const Eigen::Vector3d& get_half() const { return half_; }
  double get_r_min() const { return r_min_; }

 private:
  Eigen::Vector3d half_;
  double r_min_;
};

// --------------------------------------------------------------------------- //
// The fused cost: every keep-out point in ONE CostModelAbstract.
//
// Written against `CostModelAbstractTpl` rather than as a residual + activation
// pair on purpose.  A residual is the right abstraction when crocoddyl's
// Gauss-Newton machinery should do the work; here the whole point is to NOT do
// that work -- the term wants to write Lx and Lxx itself, touching only the
// active points and only their nv x nv block.  Going through a residual of
// dimension 86 would put a 86 x 66 Rx and a 66x66 GEMM back in the hot path,
// which is the thing being removed.
//
// Lu, Lxu and Luu stay at the zeros the data was constructed with and are never
// written, because the keep-out does not depend on u.  CostModelSum still
// accumulates them (it cannot know), but that is one term's worth, not 86.
class CostModelBoxKeepOut : public crocoddyl::CostModelAbstractTpl<double> {
 public:
  typedef crocoddyl::CostModelAbstractTpl<double> CostBase;
  typedef crocoddyl::CostDataAbstractTpl<double> CostDataBase;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef crocoddyl::DataCollectorMultibodyTpl<double> DataCollectorMultibody;
  typedef crocoddyl::DataCollectorAbstractTpl<double> DataCollectorAbstract;

  struct Data : public CostDataBase {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // The base's constructor is templated as `Model<Scalar>*` and this cost is
    // not a template, so it is handed the pointer as the BASE -- which is a
    // template of exactly that form, and is all the base constructor touches
    // (get_activation, get_residual, get_state, get_nu).
    Data(CostModelBoxKeepOut* const model, DataCollectorAbstract* const data)
        : CostDataBase(static_cast<CostBase*>(model), data),
          J6(6, model->get_state()->get_nv()) {
      J6.setZero();
      pinocchio = nullptr;
      DataCollectorMultibody* d = dynamic_cast<DataCollectorMultibody*>(data);
      if (d == nullptr) {
        throw_pretty(
            "Invalid argument: the shared data should be derived from "
            "DataCollectorMultibody");
      }
      pinocchio = d->pinocchio;
      const std::size_t np = model->get_npoints();
      active.reserve(np);
      values.resize(np);
    }
    pinocchio::DataTpl<double>* pinocchio;
    Eigen::MatrixXd J6;            // scratch for one frame Jacobian
    std::vector<std::size_t> active;
    Eigen::VectorXd values;        // r_min - sdf, per point (only active used)
  };

  // `nr = 1` is a formality: the base class insists on a residual model and an
  // activation, and this cost uses neither.  Nothing reads the 1-vector.
  CostModelBoxKeepOut(std::shared_ptr<StateMultibody> state, std::size_t nu,
                      const Eigen::Vector3d& half, const Eigen::Vector3d& center,
                      const std::vector<pinocchio::FrameIndex>& frames,
                      const std::vector<double>& thresholds)
      : CostBase(state, std::size_t(1), nu),
        half_(half),
        center_(center),
        frames_(frames),
        thresh_(thresholds),
        pin_model_(state->get_pinocchio().get()) {
    if (frames_.size() != thresh_.size()) {
      throw_pretty("Invalid argument: frames and thresholds differ in length");
    }
  }

  void calc(const std::shared_ptr<CostDataBase>& data,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&) override {
    Data* d = static_cast<Data*>(data.get());
    d->active.clear();
    double cost = 0.0;
    for (std::size_t i = 0; i < frames_.size(); ++i) {
      // The placement has to be refreshed here.  crocoddyl's contact forward
      // dynamics calls pinocchio::computeAllTerms, which does NOT update frame
      // placements -- ResidualModelFrameTranslation::calc updates the one frame
      // it owns, which is why the per-point stack never had to think about it.
      // Left out, every oMf holds whatever the last residual to touch it wrote,
      // the active set comes back empty at every state, and the keep-out
      // silently stops constraining anything.
      pinocchio::updateFramePlacement(*pin_model_, *d->pinocchio, frames_[i]);
      const Eigen::Vector3d p =
          d->pinocchio->oMf[frames_[i]].translation() - center_;
      const double v = thresh_[i] - sdfBox(p, half_);
      d->values[i] = v;
      if (v > 0.0) {
        cost += 0.5 * v * v;
        d->active.push_back(i);
      }
    }
    d->cost = cost;
  }

  // NB `calc` must have run at this x: the active set and the residual values
  // come from it.  That is crocoddyl's own contract for every residual in the
  // library (calcDiff reads what calc cached), and the solver honours it.
  void calcDiff(const std::shared_ptr<CostDataBase>& data,
                const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&) override {
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nv = state_->get_nv();
    d->Lx.setZero();
    d->Lxx.setZero();
    if (d->active.empty()) {
      return;
    }
    for (std::size_t k = 0; k < d->active.size(); ++k) {
      const std::size_t i = d->active[k];
      const Eigen::Vector3d p =
          d->pinocchio->oMf[frames_[i]].translation() - center_;
      const Eigen::Vector3d g = sdfBoxGrad(p, half_);
      const double v = d->values[i];
      // LOCAL_WORLD_ALIGNED: the residual this replaces is a
      // ResidualModelFrameTranslation, whose Rx is oRf * fJf.topRows(3) -- the
      // world-aligned translation Jacobian, which is exactly the top three rows
      // of the LWA frame Jacobian.
      d->J6.setZero();
      pinocchio::getFrameJacobian(*pin_model_, *d->pinocchio, frames_[i],
                                  pinocchio::LOCAL_WORLD_ALIGNED, d->J6);
      const auto J = d->J6.topRows(3);
      // Lx += Rx^T * Ar  with Ar = -v * g, over the q block only.
      d->Lx.head(nv).noalias() -= v * (J.transpose() * g);
      // Lxx += Rq^T * diag(g^2) * Rq.  diag and not the outer product g g^T:
      // crocoddyl stores Arr as an Eigen DiagonalMatrix, so the per-point stack
      // this replaces only ever contributed the diagonal (see the note above),
      // and reproducing it is what makes the two paths the same optimisation
      // problem rather than merely similar ones.
      const Eigen::Vector3d g2 = g.array().square();
      d->Lxx.topLeftCorner(nv, nv).noalias() +=
          J.transpose() * g2.asDiagonal() * J;
    }
  }

  std::shared_ptr<CostDataBase> createData(
      DataCollectorAbstract* const data) override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
  }

  std::size_t get_npoints() const { return frames_.size(); }

  // Diagnostic: how many points are active at the state `calc` last saw.  This
  // is the number that decides whether the fused form is fast, so it is
  // readable rather than inferred.
  static std::size_t n_active(const std::shared_ptr<CostDataBase>& data) {
    return static_cast<Data*>(data.get())->active.size();
  }

  std::shared_ptr<crocoddyl::CostModelBase> cloneAsDouble() const override {
    return std::make_shared<CostModelBoxKeepOut>(
        std::static_pointer_cast<StateMultibody>(state_), nu_, half_, center_,
        frames_, thresh_);
  }
  std::shared_ptr<crocoddyl::CostModelBase> cloneAsFloat() const override {
    return cloneAsDouble();
  }

  void print(std::ostream& os) const override {
    os << "CostModelBoxKeepOut {" << frames_.size() << " points, half=["
       << half_.transpose() << "]}";
  }

 private:
  Eigen::Vector3d half_;
  Eigen::Vector3d center_;
  std::vector<pinocchio::FrameIndex> frames_;
  std::vector<double> thresh_;
  const pinocchio::ModelTpl<double>* pin_model_;
};

std::shared_ptr<CostModelBoxKeepOut> make_keepout_cost(
    std::shared_ptr<crocoddyl::StateMultibodyTpl<double> > state,
    std::size_t nu, double hx, double hy, double hz, double cx, double cy,
    double cz, const bp::list& frames, const bp::list& thresholds) {
  std::vector<pinocchio::FrameIndex> f;
  std::vector<double> t;
  for (bp::ssize_t i = 0; i < bp::len(frames); ++i) {
    f.push_back(bp::extract<pinocchio::FrameIndex>(frames[i]));
  }
  for (bp::ssize_t i = 0; i < bp::len(thresholds); ++i) {
    t.push_back(bp::extract<double>(thresholds[i]));
  }
  return std::make_shared<CostModelBoxKeepOut>(
      state, nu, Eigen::Vector3d(hx, hy, hz), Eigen::Vector3d(cx, cy, cz), f, t);
}

// Standalone SDF exports, so the Python side can check the two implementations
// agree without going through an activation data.
double py_sdf(double x, double y, double z, double hx, double hy, double hz) {
  return sdfBox(Eigen::Vector3d(x, y, z), Eigen::Vector3d(hx, hy, hz));
}

bp::tuple py_sdf_grad(double x, double y, double z, double hx, double hy,
                      double hz) {
  const Eigen::Vector3d g =
      sdfBoxGrad(Eigen::Vector3d(x, y, z), Eigen::Vector3d(hx, hy, hz));
  return bp::make_tuple(g[0], g[1], g[2]);
}

// calc/calcDiff take `Eigen::Ref<const VectorXd>` because that is the signature
// the crocoddyl base declares.  Binding those directly does NOT work: eigenpy's
// converter for a const Ref hands boost::python a temporary whose storage is
// gone by the time the call is made, and the observable symptom is a stack-smash
// abort with no traceback, not a type error.  These thunks take a VectorXd by
// const reference -- a plain, well-supported eigenpy conversion -- and forward.
void py_calc(ActivationModelBoxKeepOut& a, const std::shared_ptr<BaseData>& d,
             const Eigen::VectorXd& r) {
  a.calc(d, r);
}

void py_calc_diff(ActivationModelBoxKeepOut& a,
                  const std::shared_ptr<BaseData>& d, const Eigen::VectorXd& r) {
  a.calcDiff(d, r);
}

std::string py_repr(const ActivationModelBoxKeepOut& a) {
  std::ostringstream os;
  a.print(os);
  return os.str();
}

}  // namespace

BOOST_PYTHON_MODULE(croco_keepout) {
  // The base class is registered by crocoddyl's own pywrap module, so importing
  // it here is what makes `bp::bases<Base>` resolvable.  Without it boost
  // python raises at REGISTRATION time with an unhelpful message about an
  // unregistered base, which is a confusing way to learn about an import order.
  bp::import("crocoddyl");

  // No Eigen in the constructor signature on purpose: the half extents come in
  // as three doubles, so this module needs no eigenpy numpy converters at all.
  // The hot path (calc/calcDiff) is called from C++ and never crosses numpy.
  bp::class_<ActivationModelBoxKeepOut, bp::bases<Base>,
             std::shared_ptr<ActivationModelBoxKeepOut>, boost::noncopyable>(
      "ActivationModelBoxKeepOut",
      "0.5 * max(0, r_min - sdf_box(r))^2 on a 3-vector residual.\n\n"
      "Pair with ResidualModelFrameTranslation(state, fid, box_centre, nu) so\n"
      "the residual handed in is the point relative to the box centre.",
      bp::init<double, double, double, double>(
          bp::args("self", "hx", "hy", "hz", "r_min")))
      // calc / calcDiff / createData are re-exported HERE rather than inherited.
      // crocoddyl registers them on its Python *wrapper* subclass, not on the
      // abstract base, so a plain C++ derivative inherits the base class in the
      // bp registry but none of its methods -- `createData()` on this object
      // raises an ArgumentError naming ActivationModelAbstractTpl_wrap.  It only
      // matters for direct Python calls (the solver drives calc/calcDiff in C++
      // through the virtual), but "only matters for the test" is exactly the
      // thing a test needs.
      .def("calc", &py_calc, bp::args("self", "data", "r"))
      .def("calcDiff", &py_calc_diff, bp::args("self", "data", "r"))
      .def("createData", &ActivationModelBoxKeepOut::createData,
           bp::args("self"))
      .add_property("r_min", &ActivationModelBoxKeepOut::get_r_min)
      .def("__repr__", &py_repr);

  // The fused cost.  Exposed through a factory rather than a bp::init so the
  // point list can arrive as plain Python lists -- this module deliberately
  // registers no eigenpy converters of its own (see the note on the ctor above).
  bp::class_<CostModelBoxKeepOut, bp::bases<crocoddyl::CostModelAbstractTpl<double> >,
             std::shared_ptr<CostModelBoxKeepOut>, boost::noncopyable>(
      "CostModelBoxKeepOut",
      "Every table keep-out point as ONE cost term.\n\n"
      "Equivalent to a CostModelSum of one CostModelResidual per point with\n"
      "ActivationModelBoxKeepOut on a ResidualModelFrameTranslation, but it\n"
      "skips the points whose activation is zero and accumulates into the\n"
      "shared Lxx once instead of once per point.",
      bp::no_init)
      .def("__init__", bp::make_constructor(
          &make_keepout_cost, bp::default_call_policies(),
          bp::args("state", "nu", "hx", "hy", "hz", "cx", "cy", "cz",
                   "frames", "thresholds")))
      .def("n_active", &CostModelBoxKeepOut::n_active, bp::args("data"))
      .staticmethod("n_active")
      .add_property("npoints", &CostModelBoxKeepOut::get_npoints);

  bp::def("sdf_box", &py_sdf, bp::args("x", "y", "z", "hx", "hy", "hz"));
  bp::def("sdf_box_grad", &py_sdf_grad,
          bp::args("x", "y", "z", "hx", "hy", "hz"));
}
