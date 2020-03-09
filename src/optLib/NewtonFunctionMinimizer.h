    #pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"
#include <iostream>

class NewtonFunctionMinimizer : public GradientDescentLineSearch {
public:
    NewtonFunctionMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
        : GradientDescentLineSearch(maxIterations, solveResidual, maxLineSearchIterations) {	}

    virtual ~NewtonFunctionMinimizer() {}

protected:
    virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) {


        Eigen::SimplicialLDLT<SparseMatrixd, Eigen::Lower> solver;
        SparseMatrixd Eye(x.size(), x.size());
        Eye.setIdentity();
        function->getHessian(x, hessian);
        dx = function->getGradient(x);
        hessian = hessian + reg * Eye;
        solver.compute(hessian);
        dx = solver.solve((dx));
    }

public:
    SparseMatrixd hessian;
    std::vector<Triplet<double>> hessianEntries;
    double reg = 1.0;
};
