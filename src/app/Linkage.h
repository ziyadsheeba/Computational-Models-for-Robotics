#include <Eigen/Core>
#include <array>
#include <ObjectiveFunction.h>

using Eigen::Vector2d;
using Eigen::Matrix2d;

typedef std::array<Matrix2d, 2> Tensor2x2x2;

struct Linkage
{
    Vector2d p0 = {0, 0};
    double length[2] = {1, 2};

    // forward kinematics
    std::array<Vector2d, 3> fk(const Vector2d &angles) const {
        auto p1 = p0;
        auto p2 = p1;
        //////////////////// 2.1

        // your code goes here
        // compute p1 and p2!

        //////////////////// 2.1
        return {p0, p1, p2};
    }

    // Jacobian of fk(angles)[2]
    Matrix2d dfk_dangles(const Vector2d &angles) const {
        Matrix2d dp2_dangles = Matrix2d::Zero();

        //////////////////// 2.2

        // your code goes here

        //////////////////// 2.2

        return dp2_dangles;
    }

    // derivative of dfk_dangles(angles)[2]
    Tensor2x2x2 ddfk_ddangles(const Vector2d &angles) const {
        Tensor2x2x2 tensor;
        tensor[0] = tensor[1] = Matrix2d::Zero();

        //////////////////// 2.3

        // your code goes here

        //////////////////// 2.3


        return tensor;
    }
};

class InverseKinematics : public ObjectiveFunction
{
public:
    const Linkage *linkage;
    const Vector2d *target;
public:
    double evaluate(const VectorXd& x) const override {
        double e = 0;
        // hint: `linkage->forwardKinematics(x)[2]` returns the end-effector position

        //////////////////// 2.1

        // your code goes here

        //////////////////// 2.1

        return e;
    }
    void addGradientTo(const VectorXd& x, VectorXd& grad) const override {
        //////////////////// 2.2

        // your code goes here

        //////////////////// 2.2
    }

    Matrix2d hessian(const VectorXd &x) const {
        Matrix2d hess = Matrix2d::Zero();

        //////////////////// 2.3

        // your code goes here

        //////////////////// 2.3

        return hess;
    }

    void addHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const override {

        auto hess = hessian(x);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                hessianEntries.push_back(Triplet<double>(i, j, hess(i,j)));
            }
        }
    }

};
