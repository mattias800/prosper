# Desktop frontend

`main.cpp` owns the SDL window, Vulkan presentation and application lifecycle.
Library UI, settings, gamepad input and FPS overlays live alongside it; guest APIs and
shared rendering implementations belong in their existing core/shared directories.
Stop frontend workers and release SDL-dependent objects before calling SDL_Quit.
Keep performance displays explicit about guest flips, presentations and fresh render work.
