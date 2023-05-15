import gc
import threading
import time

import psutil
import torch


class PeakCPUMemory:
    def __init__(self):
        self.process = psutil.Process()
        self.peak_monitoring = False

    def peak_monitor(self):
        self.cpu_memory_peak = -1

        while True:
            self.cpu_memory_peak = max(self.process.memory_info().rss, self.cpu_memory_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            if not self.peak_monitoring:
                break

    def start(self):
        self.peak_monitoring = True
        self.thread = threading.Thread(target=self.peak_monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.peak_monitoring = False
        self.thread.join()
        return self.cpu_memory_peak


cpu_peak_tracker = PeakCPUMemory()


def start_measure():
    # Time
    measures = {"time": time.time()}

    gc.collect()
    torch.cuda.empty_cache()

    # CPU mem
    measures["cpu"] = psutil.Process().memory_info().rss
    cpu_peak_tracker.start()

    # GPU mem
    for i in range(torch.cuda.device_count()):
        measures[str(i)] = torch.cuda.memory_allocated(i)
    torch.cuda.reset_peak_memory_stats()

    return measures


def end_measure(start_measures):
    # Time
    measures = {"time": time.time() - start_measures["time"]}

    gc.collect()
    torch.cuda.empty_cache()

    # CPU mem
    measures["cpu"] = (psutil.Process().memory_info().rss - start_measures["cpu"]) / 2**20
    measures["cpu-peak"] = (cpu_peak_tracker.stop() - start_measures["cpu"]) / 2**20

    # GPU mem
    for i in range(torch.cuda.device_count()):
        measures[str(i)] = (torch.cuda.memory_allocated(i) - start_measures[str(i)]) / 2**20
        measures[f"{i}-peak"] = (torch.cuda.max_memory_allocated(i) - start_measures[str(i)]) / 2**20

    return measures


def log_measures(measures, description):
    print(f"{description}:")
    print(f"- Time: {measures['time']:.2f}s")
    for i in range(torch.cuda.device_count()):
        print(f"- GPU {i} allocated: {measures[str(i)]:.2f}MiB")
        peak = measures[f"{i}-peak"]
        print(f"- GPU {i} peak: {peak:.2f}MiB")
    print(f"- CPU RAM allocated: {measures['cpu']:.2f}MiB")
    print(f"- CPU RAM peak: {measures['cpu-peak']:.2f}MiB")
