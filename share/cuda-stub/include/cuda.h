/* Not a CUDA header. Clang's CUDA driver looks for an SDK installation to pick
 * the launch ABI; this directory is that "installation" and only reports the
 * SDK version. Nothing here is ever included (we compile with -nocudainc). */
#define CUDA_VERSION 12050
