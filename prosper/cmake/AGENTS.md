# CMake build contracts

Small shared build mechanisms and opt-in toolchain profiles. Revision generation embeds the source
identity of the linked compiler; toolchain profiles select local tools without installing them or
changing another build directory. Keep runtime source behavior outside this folder.
