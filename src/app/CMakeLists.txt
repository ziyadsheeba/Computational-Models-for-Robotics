cmake_minimum_required(VERSION 3.5)

project(app)

find_package(Json REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(opt-app
    opt-app.cpp
    ContourPlot.h
)
target_link_libraries(opt-app
    optLib
    guiLib
)

add_executable(ik-app
    ik-app.cpp
    ContourPlot.h
    Linkage.h
)
target_link_libraries(ik-app
    optLib
    guiLib
)

add_executable(test-app
    test.cpp
    Linkage.h
    TestResult.h
)
target_link_libraries(test-app
    optLib
)
