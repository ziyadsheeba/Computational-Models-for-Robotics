#include <ObjectiveFunction.h>
#include <colormap.h>

using Eigen::Vector2d;

struct ContourPlot {
    double min, max;
    Vector2d from, to;
    int size;
    int nContours = 20;
    double * data = nullptr;
    unsigned char * imgData = nullptr;
    bool showContourEdges = false;
    int widthOld = -1, heightOld = -1;

    void generate(const ObjectiveFunction *obj, int w, int h, bool recompute = false) {

        if(widthOld != w || heightOld != h){
            if(imgData != nullptr)
                free(imgData);
            imgData = new unsigned char[w*h*4];

            if(data != nullptr)
                free(data);
            data = new double[w*h];
            widthOld = w;
            heightOld = h;
        }

        // find min/max
        if(recompute){
            min = HUGE_VAL; max = -HUGE_VAL;
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    double f = obj->evaluate(fromScreen(i, j, size, size));
                    min = std::min(min, f);
                    max = std::max(max, f);
                }
            }
        }

        double * dataPtr = data;
        unsigned char* px = imgData;
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                // sample function at i,j
                double f = obj->evaluate(fromScreen(i, j, size, size));
                *(dataPtr++) = f;

                // use log-scale, +1 to prevent log(0)
                f = log10(f - min + 1) / log10(max - min + 1);

                // put into 'buckets'
                f = floor(f*(double)nContours) / (double)nContours;

                // map to a color
                float r, g, b;
                colorMapColor(f, r, g, b);
                px[0] = (unsigned char)(255.f*r);
                px[1] = (unsigned char)(255.f*g);
                px[2] = (unsigned char)(255.f*b);
                px[3] = 150;
                px += 4; // advance to next pixel
            }
        }

        if(showContourEdges){
            px = imgData;
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    for (int k = 0; k < 3; ++k) {
                        if((i < w-1 && px[k] != px[4+k]) || (j<h-1 && px[k] != px[4*w+k])){
                            px[0] = 0;
                            px[1] = 0;
                            px[2] = 0;
                            px[3] = 150;
                            break;
                        }
                    }
                    px += 4;
                }
            }
        }
    }

    void translate(const ObjectiveFunction *obj, int w, int h, int dw, int dh, bool recompute = false) {

        if(widthOld != w || heightOld != h){
            generate(obj, w, h, recompute);
        }

        Vector2d dx = Vector2d(dw, -dh)/(double)size;
        from -= dx.cwiseProduct(to-from);
        to -= dx.cwiseProduct(to-from);

        // find min/max
        if(recompute){
            min = HUGE_VAL; max = -HUGE_VAL;
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    double f = obj->evaluate(fromScreen(i, j, size, size));
                    min = std::min(min, f);
                    max = std::max(max, f);
                }
            }
        }

        int wm = std::max(0, dw);
        int wp = std::min(w, w+dw);
        int hm = std::max(0, dh);
        int hp = std::min(h, h+dh);

        double * dataNew = new double[w*h];
        double * dataNewPtr = dataNew;
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {

                if(i >= wm && i < wp && j >= hm && j < hp){
                    int idx = (w*(j-dh) + i-dw);
                    *(dataNewPtr++) = data[idx];
                }
                else{
                    *(dataNewPtr++) = obj->evaluate(fromScreen(i, j, size, size));
                }
            }
        }
        delete data;
        data = dataNew;

        dataNewPtr = dataNew;
        unsigned char* px = imgData;
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                // sample function at i,j
                double f = *(dataNewPtr++);

                // use log-scale, +1 to prevent log(0)
                f = log10(f - min + 1) / log10(max - min + 1);

                // put into 'buckets'
                f = floor(f*(double)nContours) / (double)nContours;

                // map to a color
                float r, g, b;
                colorMapColor(f, r, g, b);
                px[0] = (unsigned char)(255.f*r);
                px[1] = (unsigned char)(255.f*g);
                px[2] = (unsigned char)(255.f*b);
                px[3] = 150;
                px += 4; // advance to next pixel
            }
        }

        if(showContourEdges){
            px = imgData;
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    for (int k = 0; k < 3; ++k) {
                        if((i < w-1 && px[k] != px[4+k]) || (j < h-1 && px[k] != px[4*w+k])){
                            px[0] = 0;
                            px[1] = 0;
                            px[2] = 0;
                            px[3] = 150;
                            break;
                        }
                    }
                    px += 4;
                }
            }
        }
    }

    Vector2d fromScreen(int i, int j, int w, int h) const {
        Vector2d x = {(double)i/(double)w, -(double)j/(double)h};
//        x[0] = ((double)i/(double)w - translation[0])*zoom;
//        x[1] = (-(double)j/(double)h - translation[1])*zoom;

        return from + x.cwiseProduct(to-from);
    }

    template<class S>
    Vector2d fromScreen(S i, S j) const {
        return fromScreen((double)i, (double)j, size, size);
    }

    double toScreen(double s, int dim) const {
        return (s - from[dim])/((to-from)[dim])  * (double)((dim == 0) ? size : -size);
//        return (s/zoom + translation[dim]) * (double)((dim == 0) ? size : -size);
    }
};
