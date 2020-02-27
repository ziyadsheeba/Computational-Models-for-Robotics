    #pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"

class NewtonFunctionMinimizer : public GradientDescentLineSearch {
public:
    NewtonFunctionMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
        : GradientDescentLineSearch(maxIterations, solveResidual, maxLineSearchIterations) {	}

    virtual ~NewtonFunctionMinimizer() {}

protected:
    virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) {

        //////////////////// 1.3

        // your code goes here

        //////////////////// 1.3
    }

public:
    SparseMatrixd hessian;
    std::vector<Triplet<double>> hessianEntries;
    double reg = 1.0;
};
