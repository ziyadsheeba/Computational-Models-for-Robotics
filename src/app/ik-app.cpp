#ifdef WIN32
#define NOMINMAX
#endif
#include <application.h>
#include <imgui.h>
#include <imgui_multiplot.h>
#include "ContourPlot.h"

#include <ObjectiveFunction.h>
#include <RandomMinimizer.h>
#include <GradientDescentMinimizer.h>
#include <NewtonFunctionMinimizer.h>

#include <iostream>
#include <math.h>
#include <deque>
#include <chrono>
#include <algorithm>

#include <Eigen/Core>
using Eigen::Vector2f;
using Eigen::Vector2d;
using Eigen::VectorXd;

#include "Linkage.h"

class IKApp : public Application
{
public:
    IKApp(int w, int h, const char * title, float pixelRatio = 2.f)
        : Application(title, w, h) {

        clearColor[0] = clearColor[1] = clearColor[2] = 0.15f;

        lastFrame = std::chrono::high_resolution_clock::now();

        ik.linkage = &linkage;
        ik.target = &target;

        // create solvers
        minimizers[RANDOM] = {&random, "Random Minimizer", nvgRGBA(100, 255, 255, 150)};
        minimizers[GD_FIXED] = {&gdFixed, "Gradient Descent Fixed Step Size", nvgRGBA(255, 100, 100, 150)};
        minimizers[GD_LS] = {&gdLineSearch, "Gradient Descent w/ Line Search", nvgRGBA(255, 255, 100, 150)};
        minimizers[NEWTON] = {&newton, "Newton's method",nvgRGBA(255, 100, 255, 150)};
        resetMinimizers();

//        // set up `functionValues`, used for plots
//        functionValues = new float*[minimizers.size()];
//        for (int i = 0; i < minimizers.size(); ++i) {
//            functionValues[i] = new float[plotN];
//            for (int j = 0; j < plotN; ++j) {
//                functionValues[i][j] = 0.f;
//            }
//        }

        // create iso contours
        plot.size = width/pixelRatio;
        plot.from = {-M_PI, -M_PI};
        plot.to = {M_PI, M_PI};
        generatePlot(true);
    }

    void process() override {
        // move image if left mouse button is pressed
        if(mouseState.rButtonPressed){
            int dw = (mouseState.lastMouseX - cursorPosDown[0]);
            int dh = (mouseState.lastMouseY - cursorPosDown[1]);
            cursorPosDown[0] = mouseState.lastMouseX;
            cursorPosDown[1] = mouseState.lastMouseY;

            plot.translate(&ik, width/2, height, dw, dh);
            if(img != -1)
                nvgDeleteImage(vg, img);
            img = nvgCreateImageRGBA(vg, width/2, height, 0, plot.imgData);

        }

        bool settingAlpha = false;
        if(mouseState.lButtonPressed){
            if(mouseState.lastMouseX < width/2 * pixelRatio){
                settingAlpha = true;
                angles = plot.fromScreen(mouseState.lastMouseX, mouseState.lastMouseY);
                anglesHistory.clear();
                anglesHistory.push_back(angles);
            }
            else {
                float w = 20*pixelRatio;
                Vector2d x0 = {.5f*width/pixelRatio + w, .5f*height/pixelRatio};
                target = (Vector2d(mouseState.lastMouseX, mouseState.lastMouseY) - x0)/pixelRatio/linkageScale;
                generatePlot(true);
            }
        }

        // run at 60fps, or in slow mo
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::milliseconds>(now-lastFrame).count() > ((slowMo) ? 320 : 16)){

            // call all minimizers
            if(isMinimize && !settingAlpha){
                VectorXd x = angles;
                minimizers[currentMinimizer].minimizer->minimize(&ik, x);
                angles = x;
                //                for(auto &a : angles)
                //                    if(a > M_PI) a -= 2*M_PI;
                //                    else if(a < -M_PI) a+= 2*M_PI;

                anglesHistory.push_back(angles);
                while(anglesHistory.size() > 30)
                    anglesHistory.pop_front();

                //                int i = 0;
                //                for(auto &m : minimizers) {
                //                    if(m.path.size() > 0) {
                //                        VectorXd x = m.path[m.path.size()-1];
                //                        m.minimizer.minimize(obj, x);
                //                        m.path.push_back(x);
                //                        // record function values for plots
                //                        float f = (float)obj->evaluate(x);
                //                        functionValues[i++][plotCounter] = f;
                //                    }
                //                }
                //                plotCounter = (plotCounter+1) % plotN;
                //                if(plotCounter == 0) plotStatic = false;
            }

            lastFrame = now;
        }
    }

    void drawImGui() override {

        using namespace ImGui;

        BeginMainMenuBar();
        if(BeginMenu("Contour Plot")){
            if(InputInt("# of contours", &plot.nContours)){
                generatePlot();
            }

            if(Button("Recompute contours")){
                generatePlot(true);
            }

            if(Checkbox("show contour edges", &plot.showContourEdges)){
                generatePlot();
            }
            EndMenu();
        }
        if(BeginMenu("Minimizer Settings")){
            const double smin = 0.0, smax = 1.0;
            SliderScalar("GD Fixed: step size", ImGuiDataType_Double, &gdFixed.stepSize, &smin, &smax);
            InputDouble("Newton: regularizer", &newton.reg);
            EndMenu();
        }
        EndMainMenuBar();

        Begin("IK");

        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "left mouse:  start optimization");
        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "right mouse: move function landscape");
        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "mouse wheel: zoom function landscape");
        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "space bar:   play/pause optimization");

        Separator();

        Text("Linkage:");

        float angle0 = angles[0];
        if(SliderAngle("angle 0", &angle0))
            angles[0] = angle0;

        float angle1 = angles[1];
        if(SliderAngle("angle 1", &angle1))
            angles[1] = angle1;

        const double min = 0., max = 4;
        if(SliderScalarN("target", ImGuiDataType_Double, target.data(), 2, &min, &max))
            generatePlot(true);
        InputFloat("link scale", &linkageScale);

        Text("Inverse Kinematics:");

        Checkbox("play [space]", &isMinimize);
        Checkbox("Slow motion", &slowMo);

        const char* names[] = {"Random", "GD fixed step size", "GD w/ Line Search", "Newton"};
        Combo("active minimizer", (int*)&currentMinimizer, names, 4);

        End();
    }

    void drawNanoVG() override {

        // draw contour plot
        {
            // draw contour lines
            nvgBeginPath(vg);
            NVGpaint imgPaint = nvgImagePattern(vg, 0.f, 0.f, (width/2) / pixelRatio, height / pixelRatio, 0.f, img, .8f);
            nvgRect(vg, 0.f, 0.f, (width/2)/ pixelRatio, height/ pixelRatio);
            nvgFillPaint(vg, imgPaint);
            nvgFill(vg);

            nvgReset(vg);
            int i=0;
            for(const auto &angle : anglesHistory){
                float x = ((float)i+1)/(float)anglesHistory.size();

                nvgBeginPath(vg);
                nvgCircle(vg, plot.toScreen(angle[0], 0), plot.toScreen(angle[1], 1), 5);
                nvgStrokeColor(vg, nvgRGBAf(.6f, .5f, 1.f, .8f*x));
                nvgStrokeWidth(vg, 2.f);
                nvgStroke(vg);
                nvgFillColor(vg, nvgRGBAf(.6f, .5f, 1.f, .4f*x));
                nvgFill(vg);

                if(i < anglesHistory.size()-1){
                    nvgBeginPath(vg);
                    nvgMoveTo(vg, plot.toScreen(angle[0], 0), plot.toScreen(angle[1], 1));
                    const auto &b = anglesHistory[i+1];
                    nvgLineTo(vg, plot.toScreen(b[0], 0), plot.toScreen(b[1], 1));
                    nvgStrokeColor(vg, nvgRGBAf(.6f, .5f, 1.f, .8f*x));
                    nvgStrokeWidth(vg, 10.f);
                    nvgStroke(vg);
                }

                i++;
            }

            nvgBeginPath(vg);
            nvgCircle(vg, plot.toScreen(angles[0], 0), plot.toScreen(angles[1], 1), 5);
            nvgStrokeColor(vg, nvgRGBAf(1.f, .5f, 1.f, .8f));
            nvgStrokeWidth(vg, 2.f);
            nvgStroke(vg);
            nvgFillColor(vg, nvgRGBAf(1.f, .5f, 1.f, .4f));
            nvgFill(vg);
        }

        // draw linkage
        {
            auto points = linkage.fk(angles);
            Vector2f pts[3];
            float w = 20*pixelRatio;
            Vector2f x0 = {.5f*width/pixelRatio + w, .5f*height/pixelRatio};
            int i=0;
            for (auto p : points)
                pts[i++] = x0 + Vector2f(p.x(), p.y())*pixelRatio*linkageScale;

            double angle = 0;
            for (int i=0; i<2; i++) {
                angle += angles[i];
                nvgReset(vg);
                nvgBeginPath(vg);
                nvgTranslate(vg, pts[i].x(), pts[i].y());
                nvgRotate(vg, angle);
                nvgRoundedRect(vg, -w/2, -w/2, linkage.length[i]*linkageScale*pixelRatio + w, w, w/3);
                nvgStrokeColor(vg, nvgRGBf(.7f, .7f, .7f));
                nvgStrokeWidth(vg, 2.f);
                nvgStroke(vg);
                nvgFillColor(vg, nvgRGBAf(.3f, .3f, .3f, .4f));
                nvgFill(vg);
            }

            nvgReset(vg);
            for (auto p : pts){
                nvgBeginPath(vg);
                nvgCircle(vg, p.x(), p.y(), 5.f*pixelRatio);
                nvgStrokeColor(vg, nvgRGBAf(.3f, .3f, .8f, .8f));
                nvgStrokeWidth(vg, 2.f*pixelRatio);
                nvgStroke(vg);
                nvgFillColor(vg, nvgRGBAf(.3f, .3f, .8f, .4f));
                nvgFill(vg);
            }

            nvgReset(vg);
            nvgBeginPath(vg);
            nvgCircle(vg, x0.x() + target.x()*linkageScale*pixelRatio, x0.y() + target.y()*linkageScale*pixelRatio, 5.f*pixelRatio);
            nvgStrokeColor(vg, nvgRGBf(0, .8f, 0));
            nvgStrokeWidth(vg, 2.f*pixelRatio);
            nvgStroke(vg);
            nvgFillColor(vg, nvgRGBf(0, .8f, 0));
            nvgFill(vg);

        }

    }

protected:
    void keyPressed(int key, int mods) override {
        // play / pause with space bar
        if(key == GLFW_KEY_SPACE)
            isMinimize = !isMinimize;
    }

    void mouseButtonPressed(int button, int mods) override {
        cursorPosDown[0] = mouseState.lastMouseX;
        cursorPosDown[1] = mouseState.lastMouseY;
    }

    void scrollWheel(double xoffset, double yoffset) override {
        Vector2d center = (plot.to-plot.from) / 2;
        center.y() *= -1;
        double zoom = std::pow(1.10, yoffset);
        plot.from = center + zoom * (plot.from - center);
        plot.to   = center + zoom * (plot.to   - center);

        generatePlot();
    }

    void resizeWindow(int w, int h) override {
        Application::resizeWindow(w, h);
        generatePlot();
    }


private:

    void resetMinimizers(const VectorXd &x = VectorXd()){
        // reset random minimizer
        random.searchDomainMax = plot.fromScreen(0,0);
        random.searchDomainMin = plot.fromScreen(width/pixelRatio,height/pixelRatio);
        random.fBest = (x.size() == 0) ? HUGE_VAL : ik.evaluate(x);
        random.xBest = x;
    }

    void generatePlot(bool recomputeContours = false) {
        plot.generate(&ik, (width/2), height, recomputeContours);
        if(img != -1)
            nvgDeleteImage(vg, img);
        img = nvgCreateImageRGBA(vg, (width/2), height, 0, plot.imgData);
    }

private:
    Linkage linkage;
    Vector2d angles = {0,0};
    Vector2d target = {0.1, 0.2};
    InverseKinematics ik;

    // for visualization
    std::deque<Vector2d> anglesHistory;
    float linkageScale = 200.f;

    // minimizers
    RandomMinimizer random;
    GradientDescentFixedStep gdFixed;
    GradientDescentLineSearch gdLineSearch;
    NewtonFunctionMinimizer newton;
    bool isMinimize = false;
    bool slowMo = false;
    enum MinimizerTypes {
        RANDOM=0, GD_FIXED=1, GD_LS=2, NEWTON=3
    };
    MinimizerTypes currentMinimizer = GD_FIXED;
    struct MinimizerState
    {
        Minimizer *minimizer;
        std::string name;
        NVGcolor color;
        std::deque<Vector2d> path = {};
        bool show = true;
    };
    std::map<MinimizerTypes, MinimizerState> minimizers;

    int img;
    ContourPlot plot;

    // user interface
    double cursorPosDown[2];
    std::chrono::high_resolution_clock::time_point lastFrame;

    // plot
    static const int plotN = 200;
    int plotZoom = plotN;
    int plotPan = 0;
    int plotCounter = 0;
    bool plotStatic = true;
    float **functionValues;
};

int main(int, char**)
{
    // If you have high DPI screen settings, you can change the pixel ratio
    // accordingly. E.g. for 200% scaling use `pixelRatio = 2.f`
    float pixelRatio = 1.f;
    IKApp app(1080, 720, "CMM'20 - Inverse Kinematics", pixelRatio);
    app.run();

    return 0;
}
