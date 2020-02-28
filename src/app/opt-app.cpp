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

class RosenbrockFunction : public ObjectiveFunction
{
public:
    virtual double evaluate(const VectorXd& x) const {
        const double &x1 = x[0];
        const double &x2 = x[1];
        return std::pow(a-x1,2.0) + b*std::pow(x2-x1*x1, 2.0);
    }

    virtual void addGradientTo(const VectorXd& x, VectorXd& grad) const {
        const double &x1 = x[0];
        const double &x2 = x[1];
        grad[0] += -2*(a-x1) - 4*b*(x2-x1*x1)*x1;
        grad[1] += 2*b*(x2-x1*x1);
    }

    virtual void addHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const {
        const double &x1 = x[0];
        const double &x2 = x[1];
        hessianEntries.push_back(Triplet<double>(0, 0, 2 - 4*b*(x2-x1*x1) + 8*b*x1*x1));
        hessianEntries.push_back(Triplet<double>(1, 0, -4*b*x1));
        hessianEntries.push_back(Triplet<double>(1, 1, 2*b));
    }

    double a = 1, b = 10;
};

class QuadraticFunction : public ObjectiveFunction
{
public:
    virtual double evaluate(const VectorXd& x) const {
        const Vector2d &x2 = x.segment<2>(0);
        return x2.dot(a.cwiseProduct(x2)) + b.dot(x2) + c;
    }

    virtual void addGradientTo(const VectorXd& x, VectorXd& grad) const {
        const Vector2d &x2 = x.segment<2>(0);
        grad.segment<2>(0) += 2.*a.cwiseProduct(x2) + b;
    }

    virtual void addHessianEntriesTo(const VectorXd& x, std::vector<Triplet<double>>& hessianEntries) const {
        for (int i = 0; i < x.size(); ++i)
            hessianEntries.push_back(Triplet<double>(i, i, 2.*a[i]));
    }

    Vector2d a = {5, 10}, b = {0, 4};
    double c = 0.3;
};

class SineFunction : public ObjectiveFunction
{
public:
    SineFunction() {}

    virtual double evaluate(const VectorXd& x) const {
        double f = 1;
        for (int i = 0; i < x.size(); ++i) {
            f *= sin(x[i]);
        }
        return f;
    }
};

class UnconstrainedOptimizationApp : public Application
{
public:
    UnconstrainedOptimizationApp(int w, int h, const char * title, float pixelRatio = 2.f)
        : Application(title, w, h) {

        clearColor[0] = clearColor[1] = clearColor[2] = 0.15f;
        lastFrame = std::chrono::high_resolution_clock::now();

        // create solvers
        minimizers[RANDOM] = {&random, "Random Minimizer", nvgRGBA(100, 255, 255, 150)};
        minimizers[GD_FIXED] = {&gdFixed, "Gradient Descent Fixed Step Size", nvgRGBA(255, 100, 100, 150)};
        minimizers[GD_LS] = {&gdLineSearch, "Gradient Descent w/ Line Search", nvgRGBA(255, 255, 100, 150)};
        minimizers[NEWTON] = {&newton, "Newton's method",nvgRGBA(255, 100, 255, 150)};
        resetMinimizers();

        // set up `functionValues`, used for plots
        functionValues = new float*[minimizers.size()];
        for (int i = 0; i < minimizers.size(); ++i) {
            functionValues[i] = new float[plotN];
            for (int j = 0; j < plotN; ++j) {
                functionValues[i][j] = 0.f;
            }
        }

        // create iso contours
        plot.size = width/pixelRatio;
        plot.from = {0,0};
        plot.to = {2,2};
        generatePlot(true);
    }

    void process() override {
        // move image if left mouse button is pressed
        if(mouseState.rButtonPressed){
            int dw = (mouseState.lastMouseX - cursorPosDown[0]);
            int dh = (mouseState.lastMouseY - cursorPosDown[1]);
            cursorPosDown[0] = mouseState.lastMouseX;
            cursorPosDown[1] = mouseState.lastMouseY;

            plot.translate(objective(), width, height, dw, dh);
            if(img != -1)
                nvgDeleteImage(vg, img);
            img = nvgCreateImageRGBA(vg, width, height, 0, plot.imgData);

        }

        bool setting_x = false;
        if(mouseState.lButtonPressed){
            setting_x = true;
            auto x = plot.fromScreen(mouseState.lastMouseX, mouseState.lastMouseY);
            resetMinimizers(x);
        }

        // run at 60fps, or in slow mo
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::milliseconds>(now-lastFrame).count() > ((slowMo) ? 320 : 16)){

            // call all minimizers
            if(isMinimize && !setting_x){

                int i = 0;
                for(auto &m : minimizers) {
                    if(m.second.path.size() > 0) {
                        VectorXd x = m.second.path[m.second.path.size()-1];
                        m.second.minimizer->minimize(objective(), x);
                        m.second.path.push_back(x);
                        while(m.second.path.size() > 100) m.second.path.pop_front();
                        // record function values for plots
                        float f = (float)objective()->evaluate(x);
                        functionValues[i++][plotCounter] = f;
                    }
                }
                plotCounter = (plotCounter+1) % plotN;
                if(plotCounter == 0) plotStatic = false;
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
            ImGui::EndMenu();
        }
        if(BeginMenu("Minimizer Settings")){
            const double smin = 0.0, smax = 1.0;
            SliderScalar("GD Fixed: step size", ImGuiDataType_Double, &gdFixed.stepSize, &smin, &smax);
            InputDouble("Newton: regularizer", &newton.reg);
            ImGui::EndMenu();
        }
        EndMainMenuBar();

        Begin("Unconstrained Optimization");

        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "left mouse:  start optimization");
        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "right mouse: move function landscape");
        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "mouse wheel: zoom function landscape");
        TextColored(ImVec4(1.f, 1.f, 1.f, 0.5f), "space bar:   play/pause optimization");

        Separator();

        const char * items[3] = {"Quadratic Function", "Rosenbrock", "Sine Function"};
        if(Combo("Function", (int*)&currentFunction, items, 3)){
            generatePlot(true);
            isMinimize = false;
        }

        Checkbox("play [space]", &isMinimize);
        Checkbox("Slow motion", &slowMo);

        for (auto &m : minimizers) {
            PushID(m.second.name.c_str());
            ColorButton(m.second.name.c_str(), ImVec4(m.second.color.r, m.second.color.g, m.second.color.b, 1.0));
            SameLine();
            Text("%s", m.second.name.c_str());
            SameLine();
            Checkbox("", &m.second.show);
            PopID();
        }

        // plots
        if(CollapsingHeader("Plot"))	{
            Indent();

            const int n = minimizers.size();
            const char ** names = new const char*[n];
            ImColor *colors = new ImColor[n];
            int i = 0;
            int p = plotN - plotZoom;
            for (const auto &m : minimizers) {
                names[i] = m.second.name.c_str();
                colors[i] = ImColor(m.second.color.r, m.second.color.g, m.second.color.b);
                i++;
            }

            int plotNZoom = plotZoom;
            PlotMultiLines("Function values", n, names, colors,
                [](const void *data, int idx)->float { return ((const float*)data)[idx]; },
                ((const void *const *)functionValues), plotN, plotNZoom, ((plotStatic) ? 0 : plotCounter) + plotPan, plot.min, plot.max, ImVec2(0, 150));

            SliderInt("plot zoom", &plotZoom, plotN, 0);
            SliderInt("plot pan", &plotPan, 0, plotN - plotNZoom);
            Unindent();
        }

        End();

    }

    void drawNanoVG() override {

        // draw contour plot
        {
            // draw contour lines
            nvgBeginPath(vg);
            NVGpaint imgPaint = nvgImagePattern(vg, 0.f, 0.f, (width) / pixelRatio, height / pixelRatio, 0.f, img, .8f);
            nvgRect(vg, 0.f, 0.f, (width)/ pixelRatio, height/ pixelRatio);
            nvgFillPaint(vg, imgPaint);
            nvgFill(vg);

            // draw path for each minimizer
            nvgReset(vg);
            for(const auto &mm : minimizers)
            {
                const auto &m = mm.second;
                if(m.path.size() > 0 && m.show){

                    int i=0;
                    for(const auto &angle : m.path){
                        float t = ((float)i+1)/(float)m.path.size();

                        nvgBeginPath(vg);
                        nvgCircle(vg, plot.toScreen(angle[0], 0), plot.toScreen(angle[1], 1), 5);
                        auto color = m.color;
                        color.a = .8f*t;
                        nvgStrokeColor(vg, color);
                        nvgStrokeWidth(vg, 2.f);
                        nvgStroke(vg);
                        color.a = .4f*t;
                        nvgFillColor(vg, color);
                        nvgFill(vg);

                        if(i < m.path.size()-1){
                            nvgBeginPath(vg);
                            nvgMoveTo(vg, plot.toScreen(angle[0], 0), plot.toScreen(angle[1], 1));
                            const auto &b = m.path[i+1];
                            nvgLineTo(vg, plot.toScreen(b[0], 0), plot.toScreen(b[1], 1));
                            color.a = .8f*t;
                            nvgStrokeColor(vg, color);
                            nvgStrokeWidth(vg, 10.f);
                            nvgStroke(vg);
                        }

                        i++;
                    }
                }
            }
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

    const ObjectiveFunction* objective() const {
        switch (currentFunction) {
        case QUADRATIC:
            return &quadraticFunction;
        case ROSENBROCK:
            return &rosenbrockFunction;
        case SINE:
            return &sineFunction;
        }
    }

    void resetMinimizers(const Vector2d &x = Vector2d()){

        // reset random minimizer
        random.searchDomainMax = plot.fromScreen(0,0);
        random.searchDomainMin = plot.fromScreen(width/pixelRatio,height/pixelRatio);
        random.fBest = (x.size() == 0) ? HUGE_VAL : objective()->evaluate(x);
        random.xBest = x;

        // remove all recorded paths
        for (auto &m : minimizers) {
            m.second.path.clear();
            m.second.path.push_back(x);
        }
    }

    void generatePlot(bool recomputeContours = false) {
        plot.generate(objective(), (width), height, recomputeContours);
        if(img != -1)
            nvgDeleteImage(vg, img);
        img = nvgCreateImageRGBA(vg, (width), height, 0, plot.imgData);
    }

private:
    // objective functions
    QuadraticFunction quadraticFunction;
    RosenbrockFunction rosenbrockFunction;
    SineFunction sineFunction;
    enum FunctionTypes {
        QUADRATIC=0, ROSENBROCK=1, SINE=2
    } currentFunction = QUADRATIC;

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
    UnconstrainedOptimizationApp app(1080, 720, "CMM'20 - Unconstrained Optimization", pixelRatio);
    app.run();

    return 0;
}
