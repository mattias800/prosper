# Graphics state tests

These tests constrain decoding and translation of guest render and pipeline state, plus the
backend behavior that consumes it. Register-extraction tests construct explicit guest register
files or command packets; their expected fields must not be re-derived using the production
extractor's masks. A missing register and an observed zero register are different inputs.

CPU state tests do not establish rendered pixels. Vulkan cases in this folder additionally
exercise actual device execution and have their own registration and capability requirements.
Cross-stage shader compilation tests belong in the sibling `recompiler/` and `execute/` folders.
