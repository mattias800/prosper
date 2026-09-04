# RenderDoc in-application API header (vendored)

`renderdoc_app.h`, vendored **verbatim** from **RenderDoc** by Baldur Karlsson
(https://github.com/baldurk/renderdoc), version **1.45**. Byte-identical to the copy Fedora's
`renderdoc-devel` package installs at `/usr/include/renderdoc_app.h`.

- **License:** MIT (see `LICENSE`). Copyright (c) 2015-2026 Baldur Karlsson.
- **What it is:** a single self-contained header declaring the ABI of the API that
  `librenderdoc.so` exposes through its one exported entry point, `RENDERDOC_GetAPI`. It contains
  no implementation — the library is loaded at runtime with `dlopen`, never linked.
- **Why vendored rather than `#include <renderdoc_app.h>`:** the struct is an **ABI**, and the whole
  point of using the real header is getting the function-pointer order exactly right — a hand-written
  subset that is one field out calls the wrong pointer and crashes. Vendoring also means the trigger
  compiles everywhere, with no build-time dependency on `renderdoc-devel` being installed. RenderDoc
  versions this struct explicitly (`eRENDERDOC_API_Version_*`) precisely so a consumer can pin one;
  prosper requests **1.4.1**, well below the installed 1.45, so an older RenderDoc still works.
- **Prosper glue lives elsewhere:** the loader, the frame triggers and the capture-path reporting are
  prosper's own code in `frontends/shared/diagnostics/renderdoc_capture.{hpp,cpp}`. This directory is
  the unmodified upstream header.

Do not edit this header — keep it byte-identical to upstream so the vendored copy is auditable and
updatable. To refresh it, copy a newer `renderdoc_app.h` in whole and re-check that the API version
prosper requests is still declared.
