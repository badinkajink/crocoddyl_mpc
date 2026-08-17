// Matrix-free Delassus, measured on the lean OCP's own dimensions.
//
// Pinocchio 4.1.0 ships Sathya et al.'s matrix-free operator as
// `DelassusOperatorRigidBodySystemsTpl` (delassus-operator-rigid-body.hxx) and
// does NOT expose it through the Python bindings -- only the Dense, Sparse and
// CholeskyExpression variants are wrapped.  So a Python-side measurement can
// only quote the paper.  This translation unit exists to stop quoting: it is a
// boost::python module that takes the SAME pinocchio Model/Data the crocoddyl
// OCP is built on, instantiates the C++-only operator against it, and times it
// beside the explicit path crocoddyl actually runs.
//
// Three entry points, each answering one question:
//
//   stage_bench   Where does crocoddyl's contact dynamics spend its time?  This
//                 replays pinocchio's `forwardDynamics` and the 4-argument
//                 `getKKTContactDynamicMatrixInverse` STAGE BY STAGE, with the
//                 same J and the same damping crocoddyl passes, so the
//                 Delassus-attributable slice can be separated from the M^-1
//                 and GEMM slices it is bundled with.  That slice is the
//                 numerator of every speed-up claim downstream.
//
//   mf_bench      What does the matrix-free operator cost here?  compute(),
//                 one G*x, one G^-1*x, and G materialised column by column.
//                 Returns the materialised G so Python can check it against
//                 pinocchio's explicit `computeDelassusMatrix`.
//
//   explicit_bench  The same numbers for the explicit operator on the SAME
//                 constraint set, in-process (no binding overhead), so the
//                 comparison is like-for-like.
//
// Everything is best-of-`reps` on a steady-state warm cache, reported in
// microseconds.
//
// build: croco_ext/build.sh   (see that script for why there is no cmake)

// pinocchio/fwd.hpp FIRST, before boost/python: it is what raises the
// boost::mpl list/vector limits that the 25-type constraint variant needs, and
// including boost first freezes the preprocessed headers at the default 20.
#include <pinocchio/fwd.hpp>

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>

#include <pinocchio/multibody.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/delassus.hpp>
#include <pinocchio/algorithm/delassus-operator.hpp>

#include <chrono>
#include <vector>
#include <algorithm>

namespace bp = boost::python;
namespace pin = pinocchio;

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXd;

typedef pin::ConstraintModelTpl<Scalar, 0, pin::ConstraintCollectionDefaultTpl> CModel;
typedef pin::ConstraintDataTpl<Scalar, 0, pin::ConstraintCollectionDefaultTpl> CData;
typedef std::vector<CModel> CModelVec;
typedef std::vector<CData> CDataVec;

typedef pin::DelassusOperatorRigidBodySystemsTpl<
    Scalar, 0, pin::JointCollectionDefaultTpl, CModel, std::reference_wrapper>
    MFOperator;

// ---------------------------------------------------------------- timing --
// Best-of-N rather than mean.  These are 1-30 us operations on a machine that
// is also running a browser; the mean measures the machine's mood, the minimum
// measures the code.  (The MPC-level numbers elsewhere in this study are means,
// because there the question is "what does a control period cost in practice".)
template <typename F>
double best_us(F &&f, int reps) {
  using clk = std::chrono::steady_clock;
  double best = 1e30;
  f();  // warm
  for (int i = 0; i < reps; ++i) {
    const auto t0 = clk::now();
    f();
    const auto t1 = clk::now();
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    best = std::min(best, us);
  }
  return best;
}

// ------------------------------------------------------------ stage_bench --
// Replays, stage by stage, exactly what crocoddyl's
// DifferentialActionModelContactFwdDynamics runs per node:
//
//   calc      pinocchio::forwardDynamics(model, data, tau, J, gamma, damping)
//   calcDiff  pinocchio::getKKTContactDynamicMatrixInverse(model, data, J, Kinv)
//
// The 4-argument getKKT... reuses data.llt_JMinvJt from calc -- it does NOT
// refactorize -- which is why the split matters: the only Delassus work left in
// calcDiff is one nc-column triangular solve.
bp::dict stage_bench(const pin::Model &model, pin::Data &data,
                     const VectorXd &q, const VectorXd &tau,
                     const MatrixXd &J, const VectorXd &gamma,
                     double inv_damping, int reps) {
  const Eigen::Index nv = model.nv;
  const Eigen::Index nc = J.rows();
  bp::dict out;

  // Bring data to the state crocoddyl leaves it in before forwardDynamics.
  pin::crba(model, data, q, pin::Convention::WORLD);
  pin::computeJointJacobians(model, data, q);

  // ---- calc: forwardDynamics, whole and by stage --------------------------
  out["fwd_total"] = best_us(
      [&] { pin::forwardDynamics(model, data, tau, J, gamma, inv_damping); },
      reps);

  out["fwd_cholesky_decompose"] =
      best_us([&] { pin::cholesky::decompose(model, data); }, reps);

  // sDUiJt = U^-1 J^T / sqrt(D), then JMinvJt = sDUiJt^T sDUiJt.  THIS is the
  // Delassus assembly -- the object the matrix-free operator would replace.
  out["fwd_delassus_build"] = best_us(
      [&] {
        data.sDUiJt = J.transpose();
        pin::cholesky::Uiv(model, data, data.sDUiJt);
        for (Eigen::Index k = 0; k < nv; ++k)
          data.sDUiJt.row(k) /= std::sqrt(data.D[k]);
        data.JMinvJt.noalias() = data.sDUiJt.transpose() * data.sDUiJt;
        data.JMinvJt.diagonal().array() += inv_damping;
      },
      reps);

  out["fwd_delassus_llt"] = best_us(
      [&] { data.llt_JMinvJt.compute(data.JMinvJt); }, reps);

  VectorXd lam(nc);
  out["fwd_delassus_solve1"] = best_us(
      [&] {
        lam.noalias() = -J * data.torque_residual - gamma;
        data.llt_JMinvJt.solveInPlace(lam);
      },
      reps);

  VectorXd rhs(nv);
  out["fwd_minv_solve1"] = best_us(
      [&] {
        rhs = tau - data.nle;
        pin::cholesky::solve(model, data, rhs);
      },
      reps);

  // ---- calcDiff: getKKTContactDynamicMatrixInverse, whole and by stage ----
  MatrixXd Kinv(nv + nc, nv + nc);
  pin::forwardDynamics(model, data, tau, J, gamma, inv_damping);  // set up llt
  out["kkt_total"] = best_us(
      [&] { pin::getKKTContactDynamicMatrixInverse(model, data, J, Kinv); },
      reps);

  MatrixXd bottomRight(nc, nc), topLeft(nv, nv), bottomLeft(nc, nv),
      topRight(nv, nc);

  // The ONLY Delassus-dependent stage inside calcDiff: nc right-hand sides
  // against the factor computed back in calc.
  out["kkt_delassus_solve_nc"] = best_us(
      [&] {
        bottomRight = -MatrixXd::Identity(nc, nc);
        data.llt_JMinvJt.solveInPlace(bottomRight);
      },
      reps);

  // M^-1 as a dense nv x nv block: nv sparse-Cholesky solves.  Not Delassus.
  out["kkt_minv_dense"] = best_us(
      [&] {
        topLeft.setIdentity();
        pin::cholesky::solve(model, data, topLeft);
      },
      reps);

  // The three dense products that assemble the Schur complement.  Not Delassus.
  out["kkt_gemms"] = best_us(
      [&] {
        bottomLeft.noalias() = J * topLeft;
        topRight.noalias() = bottomLeft.transpose() * (-bottomRight);
        topLeft.noalias() -= topRight * bottomLeft;
        bottomLeft = topRight.transpose();
      },
      reps);

  out["nv"] = (int)nv;
  out["nc"] = (int)nc;
  return out;
}

// --------------------------------------------------------------- mf_bench --
// The C++-only operator, on the caller's own model and constraint set.
//
// compute() is split into its two halves because they are separately useful:
// apply_on_the_right prepares the forward operator (the paper's Algorithm 1
// backward/forward sweep over Yaba), solve_in_place prepares the constrained-ABA
// augmented factorization (Algorithm 2).  A user who only needs G*x pays for
// half of what a user who needs G^-1*x pays.
bp::dict mf_bench(const pin::Model &model, pin::Data &data, const VectorXd &q,
                  const CModelVec &cmodels, double damping, int reps) {
  bp::dict out;

  // The operator's own datas.  Built here rather than taken from Python so the
  // vector is guaranteed to outlive the operator that holds a reference to it.
  CDataVec cdatas;
  cdatas.reserve(cmodels.size());
  for (const auto &cm : cmodels) cdatas.push_back(cm.createData());

  Eigen::Index nc = 0;
  for (const auto &cm : cmodels) nc += cm.residualSize();

  // compute() documents its precondition: data.oMi / data.lMi / data.J must be
  // current.  computeJointJacobians in the WORLD convention is what supplies
  // them, and it is work the explicit path does too (crocoddyl gets it from
  // computeAllTerms), so it is excluded from both sides of the comparison.
  auto prep = [&] {
    pin::computeJointJacobians(model, data, q);
    for (std::size_t i = 0; i < cmodels.size(); ++i)
      cmodels[i].calc(model, data, cdatas[i]);
  };
  prep();
  const double t_prep = best_us(prep, reps);
  out["prep"] = t_prep;

  MFOperator op(std::cref(model), std::ref(data), std::cref(cmodels),
                std::cref(cdatas), damping);
  op.updateDamping(VectorXd::Constant(nc, damping));

  // compute() mutates data, so each timed call has to be preceded by prep() to
  // put data back where it started; t_prep is subtracted back out.
  out["compute_apply"] =
      best_us([&] { prep(); op.compute(true, false); }, reps) - t_prep;
  out["compute_solve"] =
      best_us([&] { prep(); op.compute(false, true); }, reps) - t_prep;
  out["compute_both"] =
      best_us([&] { prep(); op.compute(true, true); }, reps) - t_prep;

  prep();
  op.compute(true, true);

  VectorXd x = VectorXd::Random(nc), y(nc);
  out["apply1"] = best_us([&] { op.applyOnTheRight(x, y); }, reps);

  VectorXd z(nc);
  out["solve1"] = best_us(
      [&] {
        z = x;
        op.solveInPlace(z);
      },
      reps);

  // G, one column at a time -- what it would cost to hand crocoddyl the dense
  // matrix it currently gets from JMinvJt.
  MatrixXd G(nc, nc);
  out["materialize"] = best_us(
      [&] {
        VectorXd e = VectorXd::Zero(nc), col(nc);
        for (Eigen::Index i = 0; i < nc; ++i) {
          e.setZero();
          e[i] = 1.0;
          op.applyOnTheRight(e, col);
          G.col(i) = col;
        }
      },
      reps);

  // There is no block shortcut.  applyOnTheRight is declared for a general
  // MatrixIn/MatrixOut, but instantiating it with an nc x nc argument fails to
  // compile in 4.1.0 -- mapConstraintForcesToJointSpace inside it calls a
  // vector-only Eigen method on the column argument.  Column-by-column above is
  // the only way to get G out of the operator.
  {
    VectorXd e = VectorXd::Zero(nc), col(nc);
    for (Eigen::Index i = 0; i < nc; ++i) {
      e.setZero();
      e[i] = 1.0;
      op.applyOnTheRight(e, col);
      G.col(i) = col;
    }
  }
  out["G"] = G;
  out["bytes"] = (int)op.sizeInBytes();
  out["nc"] = (int)nc;
  return out;
}

// --------------------------------------------------------- explicit_bench --
// The explicit operator on the SAME constraint set, computed the way crocoddyl
// computes it -- stack the constraint Jacobians, then G = J M^-1 J^T + mu I via
// pinocchio's sparse LTDL factor of M (this is verbatim the sDUiJt/JMinvJt
// block of pinocchio::forwardDynamics).  Using crocoddyl's own algorithm rather
// than pinocchio's faster `computeDelassusMatrix` is deliberate: the question
// is what the matrix-free operator would DISPLACE, and this is it.
//
// Returns G too, so Python can check the matrix-free operator against it.
bp::dict explicit_bench(const pin::Model &model, pin::Data &data,
                        const VectorXd &q, const CModelVec &cmodels,
                        double damping, int reps) {
  bp::dict out;
  const Eigen::Index nv = model.nv;
  CDataVec cdatas;
  cdatas.reserve(cmodels.size());
  for (const auto &cm : cmodels) cdatas.push_back(cm.createData());

  Eigen::Index nc = 0;
  for (const auto &cm : cmodels) nc += cm.residualSize();

  // Same precondition work as the matrix-free side, excluded from both.
  auto prep = [&] {
    pin::computeJointJacobians(model, data, q);
    for (std::size_t i = 0; i < cmodels.size(); ++i)
      cmodels[i].calc(model, data, cdatas[i]);
  };
  prep();
  out["prep"] = best_us(prep, reps);

  MatrixXd J(nc, nv);
  auto stack = [&] {
    // jacobianImpl writes only the constraint's own sparsity pattern and leaves
    // the rest of the block alone, so the destination has to start at zero --
    // otherwise the untouched columns keep whatever was in the allocation and
    // G comes back with NaNs in exactly the rows nobody wrote.  (crocoddyl gets
    // this for free: its ContactModelMultiple owns a persistent zeroed Jc.)
    J.setZero();
    Eigen::Index row = 0;
    for (std::size_t i = 0; i < cmodels.size(); ++i) {
      const Eigen::Index r = cmodels[i].residualSize();
      cmodels[i].jacobian(model, data, cdatas[i], J.middleRows(row, r));
      row += r;
    }
  };
  stack();
  out["stack_jacobian"] = best_us(stack, reps);

  pin::crba(model, data, q, pin::Convention::WORLD);
  MatrixXd G(nc, nc);
  out["build"] = best_us(
      [&] {
        pin::cholesky::decompose(model, data);
        data.sDUiJt = J.transpose();
        pin::cholesky::Uiv(model, data, data.sDUiJt);
        for (Eigen::Index k = 0; k < nv; ++k)
          data.sDUiJt.row(k) /= std::sqrt(data.D[k]);
        G.noalias() = data.sDUiJt.transpose() * data.sDUiJt;
        G.diagonal().array() += damping;
      },
      reps);

  // crba + decompose are shared with the rest of the dynamics (crocoddyl's
  // computeAllTerms already paid for crba), so report the marginal cost too.
  out["build_decompose_only"] =
      best_us([&] { pin::cholesky::decompose(model, data); }, reps);

  Eigen::LLT<MatrixXd> llt(nc);
  out["llt"] = best_us([&] { llt.compute(G); }, reps);
  llt.compute(G);

  VectorXd x = VectorXd::Random(nc), y(nc);
  out["apply1"] = best_us([&] { y.noalias() = G * x; }, reps);
  out["solve1"] = best_us(
      [&] {
        y = x;
        llt.solveInPlace(y);
      },
      reps);

  out["G"] = G;
  out["J"] = J;
  // The dense nc x nc matrix plus its Cholesky factor, which is what has to be
  // resident for the explicit route.
  out["bytes"] = (int)(2 * nc * nc * (Eigen::Index)sizeof(Scalar));
  out["nc"] = (int)nc;
  return out;
}

// -------------------------------------------------------- kkt_route_bench --
// The drop-in itself.
//
// crocoddyl's calcDiff needs one thing from the contact dynamics: the top-left
// block of the inverse KKT matrix applied to a set of right-hand sides,
//
//     a_partial_dtau = (M^-1 - M^-1 J^T G^-1 J M^-1),      G = J M^-1 J^T + mu I
//
// and it gets it by materialising a_partial_dtau densely and then GEMMing.  A
// matrix-free operator cannot materialise anything -- its whole premise is that
// you apply G^-1 to vectors -- so the drop-in has to invert that order: apply
// the KKT solve to each right-hand side column instead.
//
// Route A is crocoddyl's, verbatim.  Route B is the matrix-free replacement,
// and it is given every advantage: it reuses the same stacked J (an explicit
// object it should not be allowed) and the same sparse LTDL factor of M, so the
// ONLY thing that differs between the two routes is how G^-1 is obtained.
//
// n_rhs is the real column count.  For the S13 problem crocoddyl needs
// a_partial_dtau against 2*nv columns for Fx and nu for Fu.
bp::dict kkt_route_bench(const pin::Model &model, pin::Data &data,
                         const VectorXd &q, const CModelVec &cmodels,
                         int n_rhs, double damping, int reps) {
  bp::dict out;
  const Eigen::Index nv = model.nv;

  CDataVec cdatas;
  cdatas.reserve(cmodels.size());
  for (const auto &cm : cmodels) cdatas.push_back(cm.createData());
  Eigen::Index nc = 0;
  for (const auto &cm : cmodels) nc += cm.residualSize();

  auto prep = [&] {
    pin::computeJointJacobians(model, data, q);
    for (std::size_t i = 0; i < cmodels.size(); ++i)
      cmodels[i].calc(model, data, cdatas[i]);
  };
  prep();

  MatrixXd J = MatrixXd::Zero(nc, nv);
  {
    Eigen::Index row = 0;
    for (std::size_t i = 0; i < cmodels.size(); ++i) {
      const Eigen::Index r = cmodels[i].residualSize();
      cmodels[i].jacobian(model, data, cdatas[i], J.middleRows(row, r));
      row += r;
    }
  }
  pin::crba(model, data, q, pin::Convention::WORLD);
  pin::cholesky::decompose(model, data);

  const MatrixXd RHS = MatrixXd::Random(nv, n_rhs);
  MatrixXd outA(nv, n_rhs), outB(nv, n_rhs);

  // ---- route A: materialise a_partial_dtau, then one GEMM -----------------
  MatrixXd G(nc, nc), Minv(nv, nv), aptau(nv, nv), JMinv(nc, nv), MJtGi(nv, nc);
  Eigen::LLT<MatrixXd> llt(nc);
  auto routeA = [&] {
    pin::cholesky::decompose(model, data);
    data.sDUiJt = J.transpose();
    pin::cholesky::Uiv(model, data, data.sDUiJt);
    for (Eigen::Index k = 0; k < nv; ++k)
      data.sDUiJt.row(k) /= std::sqrt(data.D[k]);
    G.noalias() = data.sDUiJt.transpose() * data.sDUiJt;
    G.diagonal().array() += damping;
    llt.compute(G);

    Minv.setIdentity();
    pin::cholesky::solve(model, data, Minv);   // nv sparse solves
    JMinv.noalias() = J * Minv;                // nc x nv
    MJtGi = JMinv.transpose();                 // nv x nc
    llt.solveInPlace(MJtGi.transpose());       // (G^-1 J M^-1)^T, nv solves
    aptau = Minv;
    aptau.noalias() -= MJtGi * JMinv;
    outA.noalias() = aptau * RHS;
  };
  routeA();
  out["route_a"] = best_us(routeA, reps);

  // ---- route B: the matrix-free operator, one KKT solve per column --------
  MFOperator op(std::cref(model), std::ref(data), std::cref(cmodels),
                std::cref(cdatas), damping);
  op.updateDamping(VectorXd::Constant(nc, damping));

  MatrixXd work(nv, n_rhs);
  VectorXd s(nc);
  auto routeB = [&] {
    prep();
    op.compute(false, true);                 // solve_in_place half only
    pin::cholesky::decompose(model, data);
    work = RHS;
    pin::cholesky::solve(model, data, work); // t = M^-1 rhs, n_rhs solves
    for (int j = 0; j < n_rhs; ++j) {
      s.noalias() = J * work.col(j);         // s = J t
      op.solveInPlace(s);                    // s <- G^-1 s   [matrix-free]
      outB.col(j).noalias() = J.transpose() * s;
    }
    pin::cholesky::solve(model, data, outB); // M^-1 J^T G^-1 J M^-1 rhs
    outB = work - outB;
  };
  routeB();
  out["route_b"] = best_us(routeB, reps);

  // How much of route B is the operator, and how much is the shared M^-1 work?
  out["route_b_solves_only"] = best_us(
      [&] {
        for (int j = 0; j < n_rhs; ++j) {
          s.noalias() = J * work.col(j);
          op.solveInPlace(s);
        }
      },
      reps);

  out["max_abs_diff"] = (outA - outB).cwiseAbs().maxCoeff();
  out["rel_diff"] = (outA - outB).norm() / outA.norm();
  out["n_rhs"] = n_rhs;
  out["nv"] = (int)nv;
  out["nc"] = (int)nc;
  return out;
}

// ------------------------------------------------------------ DelassusMF --
// The operator itself, exposed to Python.
//
// This is the binding pinocchio 4.1.0 does not ship.  It is deliberately thin:
// no new algorithm, just a lifetime-safe holder so the header-only operator can
// be driven from a script.  The holder owns its constraint-data vector and keeps
// the Python model/data objects alive, because the operator stores bare
// references to all four and outliving any of them is a segfault, not an error.
//
//   op = croco_mfd.DelassusMF(model, data, constraint_models, damping)
//   op.compute()          # after computeJointJacobians(model, data, q)
//   y = op.apply(x)       # G x
//   z = op.solve(x)       # (G + damping I)^-1 x
//   G = op.matrix()       # dense, column by column -- for checking, not for speed
struct DelassusMF {
  DelassusMF(bp::object model_obj, bp::object data_obj,
             const CModelVec &cmodels, double damping)
      : m_model_obj(model_obj), m_data_obj(data_obj), m_cmodels(cmodels),
        m_damping(damping) {
    pin::Model &model = bp::extract<pin::Model &>(model_obj);
    pin::Data &data = bp::extract<pin::Data &>(data_obj);
    m_cdatas.reserve(m_cmodels.size());
    for (const auto &cm : m_cmodels) m_cdatas.push_back(cm.createData());
    m_size = 0;
    for (const auto &cm : m_cmodels) m_size += cm.residualSize();
    m_op.reset(new MFOperator(std::cref(model), std::ref(data),
                              std::cref(m_cmodels), std::cref(m_cdatas),
                              damping));
    m_op->updateDamping(VectorXd::Constant(m_size, damping));
  }

  // Runs each constraint's calc, then the operator's decomposition.  The caller
  // is responsible for data.oMi / data.J being current (computeJointJacobians).
  void compute(bool apply_on_the_right, bool solve_in_place) {
    const pin::Model &model = bp::extract<const pin::Model &>(m_model_obj);
    pin::Data &data = bp::extract<pin::Data &>(m_data_obj);
    for (std::size_t i = 0; i < m_cmodels.size(); ++i)
      m_cmodels[i].calc(model, data, m_cdatas[i]);
    m_op->compute(apply_on_the_right, solve_in_place);
  }

  VectorXd apply(const VectorXd &x) const {
    VectorXd y(m_size);
    m_op->applyOnTheRight(x, y);
    return y;
  }

  VectorXd solve(const VectorXd &x) const {
    VectorXd y = x;
    m_op->solveInPlace(y);
    return y;
  }

  MatrixXd matrix() const {
    MatrixXd G(m_size, m_size);
    VectorXd e = VectorXd::Zero(m_size), col(m_size);
    for (Eigen::Index i = 0; i < m_size; ++i) {
      e.setZero();
      e[i] = 1.0;
      m_op->applyOnTheRight(e, col);
      G.col(i) = col;
    }
    return G;
  }

  Eigen::Index size() const { return m_size; }
  std::size_t bytes() const { return m_op->sizeInBytes(); }
  double damping() const { return m_damping; }

private:
  bp::object m_model_obj, m_data_obj;  // keep the Python owners alive
  CModelVec m_cmodels;
  CDataVec m_cdatas;
  double m_damping;
  Eigen::Index m_size;
  std::unique_ptr<MFOperator> m_op;
};

BOOST_PYTHON_MODULE(croco_mfd) {
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<MatrixXd>();
  eigenpy::enableEigenPySpecific<VectorXd>();

  bp::def("stage_bench", &stage_bench,
          (bp::arg("model"), bp::arg("data"), bp::arg("q"), bp::arg("tau"),
           bp::arg("J"), bp::arg("gamma"), bp::arg("inv_damping"),
           bp::arg("reps") = 200),
          "Per-stage timing of the explicit contact-dynamics pipeline "
          "crocoddyl runs, in microseconds (best-of-reps).");
  bp::def("mf_bench", &mf_bench,
          (bp::arg("model"), bp::arg("data"), bp::arg("q"),
           bp::arg("constraint_models"), bp::arg("damping"),
           bp::arg("reps") = 200),
          "Timing of pinocchio's matrix-free DelassusOperatorRigidBodySystems.");
  bp::def("explicit_bench", &explicit_bench,
          (bp::arg("model"), bp::arg("data"), bp::arg("q"),
           bp::arg("constraint_models"), bp::arg("damping"),
           bp::arg("reps") = 200),
          "Timing of the explicit Delassus (crocoddyl's own JMinvJt algorithm) "
          "on the same constraint set.");
  bp::def("kkt_route_bench", &kkt_route_bench,
          (bp::arg("model"), bp::arg("data"), bp::arg("q"),
           bp::arg("constraint_models"), bp::arg("n_rhs"), bp::arg("damping"),
           bp::arg("reps") = 100),
          "Head-to-head on the block crocoddyl's calcDiff actually needs: "
          "materialise-then-GEMM (route A, crocoddyl's) vs one matrix-free KKT "
          "solve per right-hand side (route B).");

  bp::class_<DelassusMF, boost::noncopyable>(
      "DelassusMF",
      "pinocchio's matrix-free DelassusOperatorRigidBodySystems, which 4.1.0\n"
      "ships as C++ headers only.  Construct, call compute() whenever the\n"
      "configuration changes (with data.oMi/data.J already current from\n"
      "computeJointJacobians), then apply()/solve() as many times as needed.",
      bp::init<bp::object, bp::object, const CModelVec &, double>(
          (bp::arg("model"), bp::arg("data"), bp::arg("constraint_models"),
           bp::arg("damping"))))
      .def("compute", &DelassusMF::compute,
           (bp::arg("apply_on_the_right") = true,
            bp::arg("solve_in_place") = true),
           "Update the decomposition.  Ask for only the half you will use.")
      .def("apply", &DelassusMF::apply, bp::arg("x"), "G x")
      .def("solve", &DelassusMF::solve, bp::arg("x"), "(G + damping I)^-1 x")
      .def("matrix", &DelassusMF::matrix,
           "Dense G, materialised column by column.  For checking, not speed.")
      .add_property("size", &DelassusMF::size)
      .add_property("bytes", &DelassusMF::bytes)
      .add_property("damping", &DelassusMF::damping);
}
