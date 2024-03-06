import multiprocessing


def worker_function(index):
    result = 0
    for _ in range(1000000):  # Some computation task for testing purposes
        result += 1
    print(f"Worker {index} result: {result}")


if __name__ == "__main__":
    num_cpus = 10
    processes = []

    for i in range(num_cpus):
        process = multiprocessing.Process(target=worker_function, args=(i,))
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All processes have finished.")
