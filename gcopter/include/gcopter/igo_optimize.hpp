#ifndef GCOPTER_IGO_OPTIMIZE_HPP
#define GCOPTER_IGO_OPTIMIZE_HPP

#include "gcopter/gcopter.hpp"
#include "gcopter/trajectory.hpp"

namespace gcopter
{
    inline GCOPTER_PolytopeSFC::SolverResult
    GCOPTER_PolytopeSFC::solveIGO(const Eigen::VectorXd &initialX,
                                  const IGOSolveOptions &options)
    {
        return solveIGOSeededLBFGS(initialX, options);
    }

    class CorridorIGOTrajectoryPlanner
    {
    public:
        typedef GCOPTER_PolytopeSFC::SolverResult SolverResult;

        struct Result
        {
            SolverResult summary;
            Trajectory<5> trajectory;
        };

    public:
        inline Result solve(GCOPTER_PolytopeSFC &solver,
                            const Eigen::VectorXd &initial_x,
                            const GCOPTER_PolytopeSFC::IGOSolveOptions &options) const
        {
            Result result;
            result.summary = solver.solveIGO(initial_x, options);
            if (result.summary.has_solution)
            {
                solver.evaluateObjectiveOnly(result.summary.best_x,
                                             nullptr,
                                             &result.trajectory);
            }
            return result;
        }
    };
}

#endif
