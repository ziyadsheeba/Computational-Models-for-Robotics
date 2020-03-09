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
        p1 << (length[0]*cos(angles[0])), (length[0]*sin(angles[0]));
        p2 << (length[0] * cos(angles[0]) + length[1] * cos(angles[0] + angles[1]))
            , (length[0] * sin(angles[0]) + length[1] * sin(angles[0] + angles[1]));
        //////////////////// 2.1
        return {p0, p1, p2};
    }

    // Jacobian of fk(angles)[2]
    Matrix2d dfk_dangles(const Vector2d &angles) const {
        Matrix2d dp2_dangles = Matrix2d::Zero();

        //////////////////// 2.2

        // your code goes here
        dp2_dangles << -length[0] * sin(angles[0]) - length[1] * sin(angles[0] + angles[1]), -length[1] * sin(angles[0] + angles[1]),
            length[0] * cos(angles[0]) + length[1] * cos(angles[0] + angles[1]), length[1] * cos(angles[0] + angles[1]);
        //////////////////// 2.2
        return dp2_dangles;
    }

    // derivative of dfk_dangles(angles)[2]
    Tensor2x2x2 ddfk_ddangles(const Vector2d &angles) const {
        Tensor2x2x2 tensor;
        tensor[0] = tensor[1] = Matrix2d::Zero();

        //////////////////// 2.3

        // your code goes here

        tensor[0] << -length[0] * cos(angles[0]) - length[1] * cos(angles[1] + angles[0]), -length[1] * cos(angles[1] + angles[0]),
                     -length[0] * sin(angles[0]) - length[1] * sin(angles[1] + angles[0]), -length[1] * sin(angles[1] + angles[0]);
        
        tensor[1] << -length[1] * cos(angles[1] + angles[0]), -length[1] * cos(angles[0] + angles[1]),
                     -length[1] * sin(angles[1] + angles[0]), -length[1] * sin(angles[1] + angles[0]);
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
        Vector2d current_position = linkage->fk(x)[2];
        Vector2d error = current_position - *target;
        e = 0.5*error.transpose() * error;
        return e;
    }
    void addGradientTo(const VectorXd& x, VectorXd& grad) const override {
        //////////////////// 2.2

        // your code goes here

        //////////////////// 2.2
        grad = grad + (linkage->dfk_dangles(x)).transpose() * (linkage->fk(x)[2] -  *target);
    }

    Matrix2d hessian(const VectorXd &x) const {
        Matrix2d hess = Matrix2d::Zero();

        //////////////////// 2.3

        // your code goes here

        //////////////////// 2.3
        Matrix2d J = linkage->dfk_dangles(x);
        Tensor2x2x2 dJ = linkage->ddfk_ddangles(x);
        Vector2d  currentPosition = linkage->fk(x)[2];
        Matrix2d  tensorProd = Matrix2d::Zero();
        tensorProd.col(0) = dJ[0].transpose() * (currentPosition - *target);
        tensorProd.col(1) = dJ[1].transpose() * (currentPosition - *target);
        hess = J.transpose() * J + tensorProd;

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
