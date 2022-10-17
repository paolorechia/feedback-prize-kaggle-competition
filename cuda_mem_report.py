from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit


def report_cuda_memory(verbose=True):
    nvmlInit()
    device_handle = nvmlDeviceGetHandleByIndex(0)

    info = nvmlDeviceGetMemoryInfo(device_handle)
    if verbose:
        print("--------------------" * 3)
        print("\n\n")
        print("\tCUDA Memory Info:")
        print("\n")
        print(f"\t\tTotal memory: {info.total / 1024 / 1024 / 1024:.2f} GB")
        print(f"\t\tFree memory: {info.free / 1024 / 1024 / 1024 :.2f} GB")
        print(f"\t\tUsed memory: {info.used / 1024 / 1024 / 1024 :.2f} GB")
        print("\n\n")
        print("--------------------" * 3)
    return info
