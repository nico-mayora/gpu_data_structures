#include <iostream>

#include "viewer.h"
#include "loader/mitsuba3.h"

int main()
{
    std::cout << "start!";
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("cornell-box");
    const auto world = loader->load();
    auto lf = world->cam->lookFrom;
    std::cout << "cam: " << lf.x << " " << lf.y << " " << lf.z << '\n';

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
