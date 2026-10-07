#pragma once
// list_nav.hpp — which list action the keyboard asks for on one frame.
//
// The table polls IsKeyPressed directly, but a focused InputText (the search box) also eats those
// keys — and IsKeyPressed(Space) fires even while typing a space, which once booted the wrong
// game out from under the user. So the poll is gated on whether text input owns the keyboard,
// and THAT decision is pure and unit-tested here rather than inline in the draw code.
//
// `text_active` is ImGui::GetIO().WantTextInput at the call site: true while any text field
// holds focus. With it set, every key belongs to the field — including Up/Down, which move the
// text caret — so the list takes none.

namespace prosper::frontend {

enum class ListNavMove { none, up, down, home, end };

inline ListNavMove list_nav_move(bool up, bool down, bool home, bool end, bool text_active) {
    if (text_active) return ListNavMove::none;
    if (up) return ListNavMove::up;
    if (down) return ListNavMove::down;
    if (home) return ListNavMove::home;
    if (end) return ListNavMove::end;
    return ListNavMove::none;
}

inline bool list_nav_open(bool enter, bool text_active) {
    return enter && !text_active;
}

} // namespace prosper::frontend
