def timed_execution(func, *args, **kwargs):
    """Measures execution time of a function, averaging over N_RUNS repetitions."""
    times = []
    result = None
    print(
        f"Benchmarking '{func.__name__}' for {GLOBAL_PARAMS['N_RUNS']} repetitions..."
    )

    # Run N_RUNS times for timing
    for i in range(GLOBAL_PARAMS["N_RUNS"]):
        # Start time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # Function call
        result = func(*args, **kwargs)

        # Stop time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        times.append(end_time - start_time)
        print(
            f"  Repetition {i+1}/{GLOBAL_PARAMS['N_RUNS']} complete in {times[-1]:.4f}s"
        )

    # Report average time
    avg_time = np.mean(times)
    print(f"Average execution time: {avg_time:.4f}s")

    # Return last model result, average time, and individual timings
    return result, avg_time, times
