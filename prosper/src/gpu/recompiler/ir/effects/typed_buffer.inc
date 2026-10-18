// ISA 70648 8.1, 12.14/12.15, Tables 97/99; llvm-mc gfx1030/gfx1013.
// CONFIDENCE: HIGH. D16 format data is packed; CMPSWAP consumes twice its returned width.
constexpr Row kMtbuf[] = {
    {F::MTBUF, 0x00, buffer("tbuffer_load_format_x", MA::Load, 0, 1, 0, false)},
    {F::MTBUF, 0x01, buffer("tbuffer_load_format_xy", MA::Load, 0, 2, 0, false)},
    {F::MTBUF, 0x02, buffer("tbuffer_load_format_xyz", MA::Load, 0, 3, 0, false)},
    {F::MTBUF, 0x03, buffer("tbuffer_load_format_xyzw", MA::Load, 0, 4, 0, false)},
    {F::MTBUF, 0x04, buffer("tbuffer_store_format_x", MA::Store, 1, 0, 0, false)},
    {F::MTBUF, 0x05, buffer("tbuffer_store_format_xy", MA::Store, 2, 0, 0, false)},
    {F::MTBUF, 0x06, buffer("tbuffer_store_format_xyz", MA::Store, 3, 0, 0, false)},
    {F::MTBUF, 0x07, buffer("tbuffer_store_format_xyzw", MA::Store, 4, 0, 0, false)},
    {F::MTBUF, 0x08, buffer("tbuffer_load_format_d16_x", MA::Load, 0, 1, 0, false)},
    {F::MTBUF, 0x09, buffer("tbuffer_load_format_d16_xy", MA::Load, 0, 1, 0, false)},
    {F::MTBUF, 0x0a, buffer("tbuffer_load_format_d16_xyz", MA::Load, 0, 2, 0, false)},
    {F::MTBUF, 0x0b, buffer("tbuffer_load_format_d16_xyzw", MA::Load, 0, 2, 0, false)},
    {F::MTBUF, 0x0c, buffer("tbuffer_store_format_d16_x", MA::Store, 1, 0, 0, false)},
    {F::MTBUF, 0x0d, buffer("tbuffer_store_format_d16_xy", MA::Store, 1, 0, 0, false)},
    {F::MTBUF, 0x0e, buffer("tbuffer_store_format_d16_xyz", MA::Store, 2, 0, 0, false)},
    {F::MTBUF, 0x0f, buffer("tbuffer_store_format_d16_xyzw", MA::Store, 2, 0, 0, false)},
};
