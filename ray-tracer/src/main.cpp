#include <iostream>

#include "viewer.h"
#include "loader/mitsuba3.h"

int main()
{
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("cornell-box");
    const auto world = loader->load();

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
