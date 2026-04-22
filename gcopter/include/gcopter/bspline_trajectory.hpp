#ifndef GCOPTER_BSPLINE_TRAJECTORY_HPP
#define GCOPTER_BSPLINE_TRAJECTORY_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace bsplinetrajectory
{

class BandedSystem
{
private:
    int N = 0;
    int lowerBw = 0;
    int upperBw = 0;
    std::vector<double> data;

public:
    inline void create(const int &n, const int &p, const int &q)
    {
        N = n;
        lowerBw = p;
        upperBw = q;
        data.assign(N * (lowerBw + upperBw + 1), 0.0);
    }

    inline double &operator()(const int &i, const int &j)
    {
        return data[(i - j + upperBw) * N + j];
    }

    inline const double &operator()(const int &i, const int &j) const
    {
        return data[(i - j + upperBw) * N + j];
    }

    template <typename EIGENMAT>
    inline EIGENMAT multiply(const EIGENMAT &x) const
    {
        EIGENMAT res = EIGENMAT::Zero(N, x.cols());
        for (int i = 0; i < N; ++i)
        {
            const int j_start = std::max(0, i - lowerBw);
            const int j_end = std::min(N - 1, i + upperBw);
            for (int j = j_start; j <= j_end; ++j)
            {
                res.row(i) += operator()(i, j) * x.row(j);
            }
        }
        return res;
    }

    inline void factorizeLU()
    {
        int iM, jM;
        double cVl;
        for (int k = 0; k <= N - 2; ++k)
        {
            iM = std::min(k + lowerBw, N - 1);
            cVl = operator()(k, k);
            if (std::abs(cVl) < 1.0e-14)
            {
                throw std::runtime_error("BandedSystem::factorizeLU(): near-zero pivot.");
            }
            for (int i = k + 1; i <= iM; ++i)
            {
                if (operator()(i, k) != 0.0)
                {
                    operator()(i, k) /= cVl;
                }
            }
            jM = std::min(k + upperBw, N - 1);
            for (int j = k + 1; j <= jM; ++j)
            {
                cVl = operator()(k, j);
                if (cVl != 0.0)
                {
                    for (int i = k + 1; i <= iM; ++i)
                    {
                        if (operator()(i, k) != 0.0)
                        {
                            operator()(i, j) -= operator()(i, k) * cVl;
                        }
                    }
                }
            }
        }
        if (N > 0 && std::abs(operator()(N - 1, N - 1)) < 1.0e-14)
        {
            throw std::runtime_error("BandedSystem::factorizeLU(): near-zero final pivot.");
        }
    }

    template <typename EIGENMAT>
    inline void solve(EIGENMAT &b) const
    {
        int iM;
        for (int j = 0; j <= N - 1; ++j)
        {
            iM = std::min(j + lowerBw, N - 1);
            for (int i = j + 1; i <= iM; ++i)
            {
                if (operator()(i, j) != 0.0)
                {
                    b.row(i) -= operator()(i, j) * b.row(j);
                }
            }
        }
        for (int j = N - 1; j >= 0; --j)
        {
            b.row(j) /= operator()(j, j);
            iM = std::max(0, j - upperBw);
            for (int i = iM; i <= j - 1; ++i)
            {
                if (operator()(i, j) != 0.0)
                {
                    b.row(i) -= operator()(i, j) * b.row(j);
                }
            }
        }
    }

    template <typename EIGENMAT>
    inline void solveAdj(EIGENMAT &b) const
    {
        int iM;
        for (int j = 0; j <= N - 1; ++j)
        {
            b.row(j) /= operator()(j, j);
            iM = std::min(j + upperBw, N - 1);
            for (int i = j + 1; i <= iM; ++i)
            {
                if (operator()(j, i) != 0.0)
                {
                    b.row(i) -= operator()(j, i) * b.row(j);
                }
            }
        }
        for (int j = N - 1; j >= 0; --j)
        {
            iM = std::max(0, j - lowerBw);
            for (int i = iM; i <= j - 1; ++i)
            {
                if (operator()(j, i) != 0.0)
                {
                    b.row(i) -= operator()(j, i) * b.row(j);
                }
            }
        }
    }
};

template <int Dim, int MaxP = 7>
class NUBSTrajectory
{
private:
    int s;
    int p;
    int order;
    int N_c;

    Eigen::VectorXd durations_;
    Eigen::VectorXd knots;
    Eigen::MatrixXd knotJacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Dim> control_points;

    struct Jet
    {
        double val = 0.0;
        Eigen::VectorXd dT;

        Jet() = default;
        explicit Jet(const int M) : val(0.0), dT(Eigen::VectorXd::Zero(M)) {}
    };

    struct BasisJetCache
    {
        const Eigen::VectorXd &u;
        const Eigen::MatrixXd &du_dT;
        const int numCtrl;
        const int Mdur;
        const double t;
        mutable std::map<std::tuple<int, int, int>, Jet> memo;

        BasisJetCache(const Eigen::VectorXd &u_,
                      const Eigen::MatrixXd &du_dT_,
                      const int numCtrl_,
                      const int Mdur_,
                      const double t_)
            : u(u_), du_dT(du_dT_), numCtrl(numCtrl_), Mdur(Mdur_), t(t_) {}

        inline Jet zeroJet() const
        {
            return Jet(Mdur);
        }

        inline Eigen::VectorXd knotSens(const int k) const
        {
            return du_dT.row(k).transpose();
        }

        Jet eval(const int i, const int q, const int d) const
        {
            if (d > q || q < 0 || i < 0 || i + q + 1 >= u.size())
            {
                return zeroJet();
            }

            const auto key = std::make_tuple(i, q, d);
            const auto iter = memo.find(key);
            if (iter != memo.end())
            {
                return iter->second;
            }

            const double eps = 1.0e-12;
            Jet out(Mdur);
            if (q == 0)
            {
                if (d == 0)
                {
                    out.val = ((u(i) <= t && t < u(i + 1)) ||
                               (std::abs(t - u(numCtrl)) <= eps && i == numCtrl - 1))
                                  ? 1.0
                                  : 0.0;
                }
                memo[key] = out;
                return out;
            }

            if (d == 0)
            {
                const Jet N1 = eval(i, q - 1, 0);
                const Jet N2 = eval(i + 1, q - 1, 0);
                const double den1 = u(i + q) - u(i);
                const double den2 = u(i + q + 1) - u(i + 1);

                double alpha = 0.0;
                double beta = 0.0;
                Eigen::VectorXd dAlpha = Eigen::VectorXd::Zero(Mdur);
                Eigen::VectorXd dBeta = Eigen::VectorXd::Zero(Mdur);

                if (std::abs(den1) > eps)
                {
                    const double n1 = t - u(i);
                    const Eigen::VectorXd dn1 = -knotSens(i);
                    const Eigen::VectorXd dden1 = knotSens(i + q) - knotSens(i);
                    alpha = n1 / den1;
                    dAlpha = (dn1 * den1 - n1 * dden1) / (den1 * den1);
                }
                if (std::abs(den2) > eps)
                {
                    const double n2 = u(i + q + 1) - t;
                    const Eigen::VectorXd dn2 = knotSens(i + q + 1);
                    const Eigen::VectorXd dden2 = knotSens(i + q + 1) - knotSens(i + 1);
                    beta = n2 / den2;
                    dBeta = (dn2 * den2 - n2 * dden2) / (den2 * den2);
                }

                out.val = alpha * N1.val + beta * N2.val;
                out.dT = dAlpha * N1.val + alpha * N1.dT +
                         dBeta * N2.val + beta * N2.dT;
                memo[key] = out;
                return out;
            }

            const Jet D1 = eval(i, q - 1, d - 1);
            const Jet D2 = eval(i + 1, q - 1, d - 1);
            const double den1 = u(i + q) - u(i);
            const double den2 = u(i + q + 1) - u(i + 1);

            double c1 = 0.0;
            double c2 = 0.0;
            Eigen::VectorXd dc1 = Eigen::VectorXd::Zero(Mdur);
            Eigen::VectorXd dc2 = Eigen::VectorXd::Zero(Mdur);
            if (std::abs(den1) > eps)
            {
                const Eigen::VectorXd dden1 = knotSens(i + q) - knotSens(i);
                c1 = static_cast<double>(q) / den1;
                dc1 = -static_cast<double>(q) * dden1 / (den1 * den1);
            }
            if (std::abs(den2) > eps)
            {
                const Eigen::VectorXd dden2 = knotSens(i + q + 1) - knotSens(i + 1);
                c2 = static_cast<double>(q) / den2;
                dc2 = -static_cast<double>(q) * dden2 / (den2 * den2);
            }

            out.val = c1 * D1.val - c2 * D2.val;
            out.dT = dc1 * D1.val + c1 * D1.dT -
                     dc2 * D2.val - c2 * D2.dT;
            memo[key] = out;
            return out;
        }
    };

    static inline void gaussRule(const int n,
                                 std::vector<double> &nodes,
                                 std::vector<double> &weights)
    {
        switch (n)
        {
        case 1:
            nodes = {0.0};
            weights = {2.0};
            break;
        case 2:
            nodes = {-0.5773502691896257, 0.5773502691896257};
            weights = {1.0, 1.0};
            break;
        case 3:
            nodes = {-0.7745966692414834, 0.0, 0.7745966692414834};
            weights = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
            break;
        case 4:
            nodes = {-0.8611363115940526, -0.3399810435848563,
                     0.3399810435848563, 0.8611363115940526};
            weights = {0.3478548451374539, 0.6521451548625461,
                       0.6521451548625461, 0.3478548451374539};
            break;
        default:
            nodes = {-0.9061798459386640, -0.5384693101056831, 0.0,
                     0.5384693101056831, 0.9061798459386640};
            weights = {0.2369268850561891, 0.4786286704993665,
                       0.5688888888888889, 0.4786286704993665,
                       0.2369268850561891};
            break;
        }
    }

    inline void buildSystemMatrixA(const int M,
                                   const Eigen::VectorXd &u_vec,
                                   BandedSystem &A_out) const
    {
        A_out.create(N_c, p, p);
        int row = 0;
        Eigen::Matrix<double, MaxP + 1, MaxP + 1> ders;

        for (int d = 0; d < s; ++d)
        {
            dersBasisFuns(d, p, u_vec(p), u_vec, ders);
            for (int j = 0; j <= p; ++j)
            {
                A_out(row, j) = ders(d, j);
            }
            ++row;
        }

        for (int i = 1; i < M; ++i)
        {
            const int span = p + i;
            dersBasisFuns(0, span, u_vec(span), u_vec, ders);
            for (int j = 0; j <= p; ++j)
            {
                A_out(row, span - p + j) = ders(0, j);
            }
            ++row;
        }

        const double t_end = u_vec(N_c);
        for (int d = s - 1; d >= 0; --d)
        {
            dersBasisFuns(d, N_c - 1, t_end, u_vec, ders);
            for (int j = 0; j <= p; ++j)
            {
                A_out(row, N_c - 1 - p + j) = ders(d, j);
            }
            ++row;
        }
    }

    inline Eigen::MatrixXd buildKnotJacobian(const Eigen::VectorXd &T,
                                             const int nc) const
    {
        const int M = static_cast<int>(T.size());
        const int num_knots = nc + p + 1;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_knots, M);
        for (int k = 0; k < num_knots; ++k)
        {
            if (k <= p)
            {
                continue;
            }
            if (k <= nc)
            {
                const int last_seg = k - p - 1;
                for (int i = 0; i <= last_seg && i < M; ++i)
                {
                    J(k, i) = 1.0;
                }
            }
            else
            {
                J.row(k).setOnes();
            }
        }
        return J;
    }

    inline void evalLocalBasisAndTimeGrad(
        const int d,
        const int span,
        const double t,
        const Eigen::VectorXd &dt_dT,
        Eigen::Matrix<double, MaxP + 1, 1> &vals,
        Eigen::MatrixXd &dvals_dT_total) const
    {
        const int M = durations_.size();
        vals.setZero();
        dvals_dT_total = Eigen::MatrixXd::Zero(p + 1, M);
        BasisJetCache cache(knots, knotJacobian, N_c, M, t);
        for (int j = 0; j <= p; ++j)
        {
            const int basis_idx = span - p + j;
            const Jet jd = cache.eval(basis_idx, p, d);
            const Jet jd1 = cache.eval(basis_idx, p, d + 1);
            vals(j) = jd.val;
            dvals_dT_total.row(j) = (jd.dT + jd1.val * dt_dT).transpose();
        }
    }

public:
    BandedSystem A;

    explicit NUBSTrajectory(const int sys_order = 3)
        : s(sys_order), p(2 * sys_order - 1), order(2 * sys_order), N_c(0)
    {
        if (p > MaxP)
        {
            throw std::runtime_error("Increase MaxP template argument.");
        }
    }

    inline int getS() const { return s; }
    inline int getP() const { return p; }
    inline int getCtrlPtNum(const int M) const { return M + 2 * s - 1; }
    inline int getPieceNum() const { return durations_.size(); }
    inline double getTotalDuration() const { return durations_.sum(); }
    inline const Eigen::VectorXd &getDurations() const { return durations_; }
    inline const Eigen::VectorXd &getKnots() const { return knots; }
    inline const Eigen::Matrix<double, Eigen::Dynamic, Dim> &getControlPoints() const
    {
        return control_points;
    }

    inline int findSpan(const double t,
                        const int num_ctrl_pts,
                        const Eigen::VectorXd &u) const
    {
        if (t >= u(num_ctrl_pts))
        {
            return num_ctrl_pts - 1;
        }
        if (t <= u(p))
        {
            return p;
        }
        int low = p;
        int high = num_ctrl_pts;
        while (low < high)
        {
            const int mid = (low + high) / 2;
            if (t < u(mid))
            {
                high = mid;
            }
            else
            {
                low = mid + 1;
            }
        }
        return low - 1;
    }

    inline void dersBasisFuns(int n,
                              const int span,
                              const double t,
                              const Eigen::VectorXd &u,
                              Eigen::Matrix<double, MaxP + 1, MaxP + 1> &ders) const
    {
        n = std::min(n, p);
        ders.setZero();
        double ndu[MaxP + 1][MaxP + 1] = {{0.0}};
        double left[MaxP + 1] = {0.0};
        double right[MaxP + 1] = {0.0};
        ndu[0][0] = 1.0;

        for (int j = 1; j <= p; ++j)
        {
            left[j] = t - u(span + 1 - j);
            right[j] = u(span + j) - t;
            double saved = 0.0;
            for (int r = 0; r < j; ++r)
            {
                ndu[j][r] = right[r + 1] + left[j - r];
                const double temp =
                    std::abs(ndu[j][r]) < 1.0e-15 ? 0.0 : ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        for (int j = 0; j <= p; ++j)
        {
            ders(0, j) = ndu[j][p];
        }
        if (n == 0)
        {
            return;
        }

        double a[2][MaxP + 1] = {{0.0}};
        for (int r = 0; r <= p; ++r)
        {
            int s1 = 0;
            int s2 = 1;
            a[0][0] = 1.0;
            for (int k = 1; k <= n; ++k)
            {
                double d = 0.0;
                const int rk = r - k;
                const int pk = p - k;
                if (r >= k)
                {
                    const double den = ndu[pk + 1][rk];
                    a[s2][0] = std::abs(den) < 1.0e-15 ? 0.0 : a[s1][0] / den;
                    d = a[s2][0] * ndu[rk][pk];
                }
                const int j1 = rk >= -1 ? 1 : -rk;
                const int j2 = r - 1 <= pk ? k - 1 : p - r;
                for (int j = j1; j <= j2; ++j)
                {
                    const double den = ndu[pk + 1][rk + j];
                    a[s2][j] = std::abs(den) < 1.0e-15 ? 0.0 : (a[s1][j] - a[s1][j - 1]) / den;
                    d += a[s2][j] * ndu[rk + j][pk];
                }
                if (r <= pk)
                {
                    const double den = ndu[pk + 1][r];
                    a[s2][k] = std::abs(den) < 1.0e-15 ? 0.0 : -a[s1][k - 1] / den;
                    d += a[s2][k] * ndu[r][pk];
                }
                ders(k, r) = d;
                std::swap(s1, s2);
            }
        }

        double fac = p;
        for (int k = 1; k <= n; ++k)
        {
            for (int j = 0; j <= p; ++j)
            {
                ders(k, j) *= fac;
            }
            fac *= (p - k);
        }
    }

    inline Eigen::VectorXd generateKnots(const Eigen::VectorXd &T,
                                         const int nc) const
    {
        const int num_knots = nc + p + 1;
        Eigen::VectorXd u = Eigen::VectorXd::Zero(num_knots);
        double current_t = 0.0;
        for (int i = 0; i < T.size(); ++i)
        {
            current_t += T(i);
            u(p + 1 + i) = current_t;
        }
        for (int i = p + 1 + T.size(); i < num_knots; ++i)
        {
            u(i) = current_t;
        }
        return u;
    }

    inline void generate(const Eigen::MatrixXd &P_inner,
                         const Eigen::MatrixXd &headState,
                         const Eigen::MatrixXd &tailState,
                         const Eigen::VectorXd &T,
                         Eigen::MatrixXd &P_full)
    {
        const int M = T.size();
        N_c = getCtrlPtNum(M);
        durations_ = T;
        P_full.resize(N_c, Dim);
        knots = generateKnots(T, N_c);
        knotJacobian = buildKnotJacobian(T, N_c);
        buildSystemMatrixA(M, knots, A);

        Eigen::Matrix<double, Eigen::Dynamic, Dim> b =
            Eigen::Matrix<double, Eigen::Dynamic, Dim>::Zero(N_c, Dim);
        int row = 0;
        for (int d = 0; d < s; ++d)
        {
            b.row(row++) = headState.col(d).transpose();
        }
        for (int i = 1; i < M; ++i)
        {
            b.row(row++) = P_inner.row(i - 1);
        }
        for (int d = s - 1; d >= 0; --d)
        {
            b.row(row++) = tailState.col(d).transpose();
        }

        A.factorizeLU();
        A.solve(b);
        P_full = b;
        control_points = P_full;
    }

    inline Eigen::Matrix<double, Dim, 1> evaluate(double t,
                                                  const int d_ord = 0) const
    {
        if (t <= 0.0)
        {
            t = 0.0;
        }
        const double total_duration = getTotalDuration();
        if (t >= total_duration)
        {
            t = std::max(0.0, total_duration - 1.0e-12);
        }

        const int span = findSpan(t, N_c, knots);
        Eigen::Matrix<double, MaxP + 1, MaxP + 1> ders;
        dersBasisFuns(d_ord, span, t, knots, ders);
        Eigen::Matrix<double, Dim, 1> res =
            Eigen::Matrix<double, Dim, 1>::Zero();
        for (int j = 0; j <= p; ++j)
        {
            res += ders(d_ord, j) * control_points.row(span - p + j).transpose();
        }
        return res;
    }

    inline double getEnergy() const
    {
        double cost = 0.0;
        std::vector<double> nodes;
        std::vector<double> weights;
        gaussRule(std::min(s, 5), nodes, weights);
        Eigen::Matrix<double, MaxP + 1, MaxP + 1> ders;
        for (int i = p; i < knots.size() - p - 1; ++i)
        {
            const double t_start = knots(i);
            const double t_end = knots(i + 1);
            if (t_end - t_start < 1.0e-12)
            {
                continue;
            }
            const double len = t_end - t_start;
            const double mid = 0.5 * (t_end + t_start);
            for (std::size_t k = 0; k < nodes.size(); ++k)
            {
                const double t = mid + 0.5 * len * nodes[k];
                const double w = weights[k] * 0.5 * len;
                dersBasisFuns(s, i, t, knots, ders);
                Eigen::Matrix<double, Dim, 1> val =
                    Eigen::Matrix<double, Dim, 1>::Zero();
                for (int j = 0; j <= p; ++j)
                {
                    val += ders(s, j) * control_points.row(i - p + j).transpose();
                }
                cost += w * val.squaredNorm();
            }
        }
        return cost;
    }

    inline void getEnergyPartialGradByCoeffs(double &cost,
                                             Eigen::MatrixXd &gdC) const
    {
        cost = 0.0;
        gdC.setZero(N_c, Dim);
        std::vector<double> nodes;
        std::vector<double> weights;
        gaussRule(std::min(s, 5), nodes, weights);
        Eigen::Matrix<double, MaxP + 1, MaxP + 1> ders;
        for (int i = p; i < knots.size() - p - 1; ++i)
        {
            const double t_start = knots(i);
            const double t_end = knots(i + 1);
            if (t_end - t_start < 1.0e-12)
            {
                continue;
            }
            const double len = t_end - t_start;
            const double mid = 0.5 * (t_end + t_start);
            for (std::size_t k = 0; k < nodes.size(); ++k)
            {
                const double t = mid + 0.5 * len * nodes[k];
                const double w = weights[k] * 0.5 * len;
                dersBasisFuns(s, i, t, knots, ders);
                Eigen::Matrix<double, Dim, 1> val =
                    Eigen::Matrix<double, Dim, 1>::Zero();
                for (int j = 0; j <= p; ++j)
                {
                    val += ders(s, j) * control_points.row(i - p + j).transpose();
                }
                cost += w * val.squaredNorm();
                for (int j = 0; j <= p; ++j)
                {
                    gdC.row(i - p + j) += 2.0 * w * ders(s, j) * val.transpose();
                }
            }
        }
    }

    inline void getEnergyPartialGradByTimesAnalytic(Eigen::VectorXd &gdT_direct) const
    {
        const int M = durations_.size();
        gdT_direct.setZero(M);
        std::vector<double> nodes;
        std::vector<double> weights;
        gaussRule(std::min(s, 5), nodes, weights);

        for (int span = p; span < knots.size() - p - 1; ++span)
        {
            const double t_start = knots(span);
            const double t_end = knots(span + 1);
            if (t_end - t_start < 1.0e-12)
            {
                continue;
            }
            const Eigen::VectorXd du0 = knotJacobian.row(span).transpose();
            const Eigen::VectorXd du1 = knotJacobian.row(span + 1).transpose();
            for (std::size_t k = 0; k < nodes.size(); ++k)
            {
                const double xi = nodes[k];
                const double gw = weights[k];
                const double half = 0.5 * (t_end - t_start);
                const double mid = 0.5 * (t_end + t_start);
                const double t = mid + half * xi;
                const double w = gw * half;
                const Eigen::VectorXd dt_dT =
                    0.5 * (1.0 - xi) * du0 + 0.5 * (1.0 + xi) * du1;
                const Eigen::VectorXd dw_dT = 0.5 * gw * (du1 - du0);

                Eigen::Matrix<double, MaxP + 1, 1> vals;
                Eigen::MatrixXd dvals_dT;
                evalLocalBasisAndTimeGrad(s, span, t, dt_dT, vals, dvals_dT);

                Eigen::Matrix<double, Dim, 1> acc =
                    Eigen::Matrix<double, Dim, 1>::Zero();
                Eigen::MatrixXd dacc_dT = Eigen::MatrixXd::Zero(M, Dim);
                for (int j = 0; j <= p; ++j)
                {
                    const int gidx = span - p + j;
                    acc += vals(j) * control_points.row(gidx).transpose();
                    for (int m = 0; m < M; ++m)
                    {
                        dacc_dT.row(m) += dvals_dT(j, m) * control_points.row(gidx);
                    }
                }

                const double sq = acc.squaredNorm();
                gdT_direct += dw_dT * sq;
                for (int m = 0; m < M; ++m)
                {
                    gdT_direct(m) += 2.0 * w * dacc_dT.row(m).dot(acc.transpose());
                }
            }
        }
    }

    inline void getEnergyPartialGradByTimesFiniteDiff(const Eigen::VectorXd &T,
                                                      Eigen::VectorXd &gdT_direct)
    {
        const double eps = 1.0e-5;
        gdT_direct.resize(T.size());

        const Eigen::VectorXd original_knots = knots;
        for (int i = 0; i < T.size(); ++i)
        {
            Eigen::VectorXd T_p = T;
            Eigen::VectorXd T_m = T;
            T_p(i) += eps;
            T_m(i) = std::max(1.0e-8, T_m(i) - eps);

            knots = generateKnots(T_p, N_c);
            const double cost_p = getEnergy();

            knots = generateKnots(T_m, N_c);
            const double cost_m = getEnergy();

            gdT_direct(i) = (cost_p - cost_m) / (T_p(i) - T_m(i));
        }
        knots = original_knots;
    }

    inline void propagateGradAnalytic(const Eigen::MatrixXd &gdC,
                                      const Eigen::VectorXd &gdT_direct,
                                      Eigen::MatrixXd &gradByPoints,
                                      Eigen::VectorXd &gradByTimes) const
    {
        const int M = durations_.size();
        gradByPoints.resize(std::max(0, M - 1), Dim);
        gradByTimes = gdT_direct;

        Eigen::MatrixXd adjGrad = gdC;
        A.solveAdj(adjGrad);
        for (int i = 0; i < M - 1; ++i)
        {
            gradByPoints.row(i) = adjGrad.row(s + i);
        }

        auto accumulate_row_adj = [&](const int row,
                                      const int d_ord,
                                      const int span,
                                      const double t,
                                      const Eigen::VectorXd &dt_dT)
        {
            Eigen::Matrix<double, MaxP + 1, 1> vals_dummy;
            Eigen::MatrixXd dvals_dT;
            evalLocalBasisAndTimeGrad(d_ord, span, t, dt_dT, vals_dummy, dvals_dT);
            Eigen::MatrixXd rowVecByT = Eigen::MatrixXd::Zero(M, Dim);
            for (int j = 0; j <= p; ++j)
            {
                const int gidx = span - p + j;
                for (int m = 0; m < M; ++m)
                {
                    rowVecByT.row(m) += dvals_dT(j, m) * control_points.row(gidx);
                }
            }
            for (int m = 0; m < M; ++m)
            {
                gradByTimes(m) -= adjGrad.row(row).dot(rowVecByT.row(m));
            }
        };

        int row = 0;
        for (int d = 0; d < s; ++d)
        {
            accumulate_row_adj(row++, d, p, knots(p), Eigen::VectorXd::Zero(M));
        }
        for (int i = 1; i < M; ++i)
        {
            accumulate_row_adj(row++, 0, p + i, knots(p + i),
                               knotJacobian.row(p + i).transpose());
        }
        for (int d = s - 1; d >= 0; --d)
        {
            accumulate_row_adj(row++, d, N_c - 1, knots(N_c),
                               knotJacobian.row(N_c).transpose());
        }
    }

    inline void propagateGradFiniteDiff(const Eigen::MatrixXd &gdC,
                                        const Eigen::VectorXd &gdT_direct,
                                        const Eigen::VectorXd &T,
                                        Eigen::MatrixXd &gradByPoints,
                                        Eigen::VectorXd &gradByTimes) const
    {
        const int M = T.size();
        gradByPoints.resize(std::max(0, M - 1), Dim);
        gradByTimes.resize(M);

        Eigen::MatrixXd adjGrad = gdC;
        A.solveAdj(adjGrad);
        for (int i = 0; i < M - 1; ++i)
        {
            gradByPoints.row(i) = adjGrad.row(s + i);
        }

        const double eps = 1.0e-5;
        for (int i = 0; i < M; ++i)
        {
            Eigen::VectorXd T_p = T;
            Eigen::VectorXd T_m = T;
            T_p(i) += eps;
            T_m(i) = std::max(1.0e-8, T_m(i) - eps);

            const Eigen::VectorXd u_plus = generateKnots(T_p, N_c);
            BandedSystem A_plus;
            buildSystemMatrixA(M, u_plus, A_plus);
            const Eigen::MatrixXd Ap_C = A_plus.multiply(control_points);

            const Eigen::VectorXd u_minus = generateKnots(T_m, N_c);
            BandedSystem A_minus;
            buildSystemMatrixA(M, u_minus, A_minus);
            const Eigen::MatrixXd Am_C = A_minus.multiply(control_points);

            const Eigen::MatrixXd dAdT_C = (Ap_C - Am_C) / (T_p(i) - T_m(i));
            gradByTimes(i) = gdT_direct(i) - (adjGrad.cwiseProduct(dAdT_C)).sum();
        }
    }

    inline void getEnergyAndAnalyticGrad(double &cost,
                                         Eigen::MatrixXd &gradByPoints,
                                         Eigen::VectorXd &gradByTimes) const
    {
        Eigen::MatrixXd gdC;
        getEnergyPartialGradByCoeffs(cost, gdC);
        Eigen::VectorXd gdT_direct;
        getEnergyPartialGradByTimesAnalytic(gdT_direct);
        propagateGradAnalytic(gdC, gdT_direct, gradByPoints, gradByTimes);
    }

    inline void getEnergyAndFiniteDiffGrad(double &cost,
                                           Eigen::MatrixXd &gradByPoints,
                                           Eigen::VectorXd &gradByTimes)
    {
        Eigen::MatrixXd gdC;
        getEnergyPartialGradByCoeffs(cost, gdC);
        Eigen::VectorXd gdT_direct;
        getEnergyPartialGradByTimesFiniteDiff(durations_, gdT_direct);
        propagateGradFiniteDiff(gdC, gdT_direct, durations_, gradByPoints, gradByTimes);
    }
};

} // namespace bsplinetrajectory

#endif
