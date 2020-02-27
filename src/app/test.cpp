#include <NewtonFunctionMinimizer.h>
#include "Linkage.h"

#include "TestResult.h"

using namespace tests;

TestResult testJacobian(const Linkage &linkage, const VectorXd &x) {
    Matrix2d dp2 = linkage.dfk_dangles(x);

    double h = 1e-8;
    Matrix2d dp2_FD;
    for (int i = 0; i < x.size(); ++i) {
        VectorXd dx = VectorXd::Zero(x.size());
        dx[i] = h;
        dp2_FD.col(i) = (linkage.fk(x+dx)[2] - linkage.fk(x-dx)[2]) / (2*h);
    }

    return SAME(dp2, dp2_FD, 1e-6);
}

TestResult testdJacobian_dx(const Linkage &linkage, const VectorXd &x) {
    Tensor2x2x2 ddp2 = linkage.ddfk_ddangles(x);

    double h = 1e-8;
    Tensor2x2x2 ddp2_FD;
    for (int i = 0; i < x.size(); ++i) {
        VectorXd dx = VectorXd::Zero(x.size());
        dx[i] = h;
        ddp2_FD[i] = (linkage.dfk_dangles(x+dx) - linkage.dfk_dangles(x-dx)) / (2*h);
    }

    return SAME(ddp2[0], ddp2_FD[0], 1e-6) + SAME(ddp2[1], ddp2_FD[1], 1e-6);
}

TestResult testGradient(const ObjectiveFunction *obj, const VectorXd &x){
    VectorXd grad = obj->getGradient(x);

    double h = 1e-8;
    VectorXd gradFD = VectorXd::Zero(x.size());
    for (int i = 0; i < x.size(); ++i) {
        VectorXd dx = VectorXd::Zero(x.size());
        dx[i] = h;
        gradFD[i] = (obj->evaluate(x+dx) - obj->evaluate(x-dx)) / (2*h);
    }

    return SAME(grad, gradFD, 1e-6);
}

TestResult testHessian(const ObjectiveFunction *obj, const VectorXd &x){

    using Eigen::MatrixXd;

    SparseMatrixd hess;
    obj->getHessian(x, hess);

    double h = 1e-8;
    MatrixXd hessFD = MatrixXd::Zero(x.size(),x.size());
    for (int i = 0; i < x.size(); ++i) {
        VectorXd dx = VectorXd::Zero(x.size());
        dx[i] = h;
        hessFD.col(i) = (obj->getGradient(x+dx) - obj->getGradient(x-dx)) / (2*h);
    }

    return SAME(hess.toDense(), hessFD, 1e-6);
}

TestResult testLinkageJacobian() {
    Linkage linkage;
    Vector2d target = {0.1, 0.2};

    VectorXd x(2);
    x << 0.1, 0.2;

    return testJacobian(linkage, x);
}

TestResult testLinkagedJacobian_dx() {
    Linkage linkage;
    Vector2d target = {0.1, 0.2};

    VectorXd x(2);
    x << 0.1, 0.2;

    return testdJacobian_dx(linkage, x);
}

TestResult testIKGradient() {
    Linkage linkage;
    Vector2d target = {0.1, 0.2};

    VectorXd x(2);
    x << 0.1, 0.2;

    InverseKinematics ik;
    ik.linkage = &linkage;
    ik.target = &target;
    return testGradient(&ik, x);
}

TestResult testIKHessian() {
    Linkage linkage;
    Vector2d target = {0.1, 0.2};

    VectorXd x(2);
    x << 0.1, 0.2;

    InverseKinematics ik;
    ik.linkage = &linkage;
    ik.target = &target;
    return testHessian(&ik, x);
}

int main(int argc, char *argv[])
{

    // 2.2
    TEST(testLinkageJacobian);
    TEST(testIKGradient);

    // 2.3
    TEST(testLinkagedJacobian_dx);
    TEST(testIKHessian);


    return (allTestsOk ? 0 : 1);
}

