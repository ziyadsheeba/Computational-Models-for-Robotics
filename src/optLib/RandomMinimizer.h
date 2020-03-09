#pragma once

#include "Minimizer.h"
#include <random>

class RandomMinimizer : public Minimizer
{
public:
    RandomMinimizer(const VectorXd &upperLimit = VectorXd(), const VectorXd &lowerLimit = VectorXd(), double fBest = HUGE_VAL, const VectorXd &xBest = VectorXd())
        : searchDomainMax(upperLimit), searchDomainMin(lowerLimit), fBest(fBest), xBest(xBest){
        fBest = HUGE_VAL;

        // initialize random number generator
        rng.seed(std::random_device()());
        dist = std::uniform_real_distribution<>(0.0,1.0);
    }

    virtual ~RandomMinimizer() {}

    virtual bool minimize(const ObjectiveFunction *function, VectorXd &x) {
        
        double a;
        fBest = function->evaluate(x);
        xBest = x;
        for (int i = 0; i < iterations; ++i) {

            //Element wise rescaling of the random variable from between [0,1] to [min,max]
            for (int j = 0; j < x.size(); j++) {
                a = dist(rng);
                x[j] = (searchDomainMax[j] - searchDomainMin[j]) * a + searchDomainMin[j];
            }
            if (function->evaluate(x) < fBest)
            {
                xBest = x;
                fBest = function->evaluate(x);
            }
        }
        return false;
    }

public:
    int iterations = 1;
    VectorXd searchDomainMax, searchDomainMin;
    VectorXd xBest;
    double fBest;

    std::uniform_real_distribution<double> dist;
    std::mt19937 rng;
};
