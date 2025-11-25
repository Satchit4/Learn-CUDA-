#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDriverGetVersion_t)(int *);

int main() {
    void *handle = dlopen("libcuda.so.1", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(handle, "cuInit");
    cuDriverGetVersion_t cuDriverGetVersion = (cuDriverGetVersion_t)dlsym(handle, "cuDriverGetVersion");

    if (!cuInit || !cuDriverGetVersion) {
        fprintf(stderr, "Failed to find symbols\n");
        return 1;
    }

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", res);
        return 1;
    }

    int version;
    res = cuDriverGetVersion(&version);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuDriverGetVersion failed: %d\n", res);
        return 1;
    }

    printf("CUDA Driver Version: %d.%d\n", version / 1000, (version % 1000) / 10);
    return 0;
}
