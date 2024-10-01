import random
import math
import time
from bisect import bisect_left
import numpy as np
import matplotlib.pyplot as plt

def bucket_sort_quantile_based(data):
    """
    Sorts the data using a bucket sort algorithm where buckets are defined
    by quantiles estimated from a sample of the data.
    """
    n = len(data)
    if n == 0:
        return data

    # Determine the number of buckets (optimal is proportional to sqrt(n))
    num_buckets = max(1, int(math.sqrt(n)))

    # Sample data to estimate quantiles
    sample_size = min(n, max(1000, int(n * 0.01)))  # Sample 1% of data or at least 1000 points
    sample = random.sample(data, sample_size)
    sample.sort()

    # Determine bucket boundaries using quantiles from the sample
    bucket_boundaries = []
    for i in range(1, num_buckets):
        # Calculate the quantile index
        quantile = i / num_buckets
        index = int(quantile * (sample_size - 1))
        bucket_boundaries.append(sample[index])

    # Create empty buckets
    buckets = [[] for _ in range(num_buckets)]

    # Assign elements to buckets based on the boundaries
    for x in data:
        index = bisect_left(bucket_boundaries, x)
        buckets[index].append(x)

    # Sort elements within each bucket and concatenate them
    sorted_data = []
    for bucket in buckets:
        sorted_data.extend(sorted(bucket))

    return sorted_data

def bucket_sort_with_timing(data):
    """
    Sorts the data using bucket_sort_quantile_based and measures the time taken.

    Parameters:
    - data: List of numerical values to sort.

    Returns:
    - A tuple containing:
        - The sorted list.
        - The time taken to sort in seconds.
    """
    start_time = time.perf_counter()
    sorted_data = bucket_sort_quantile_based(data)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return sorted_data, elapsed_time

def run_time_complexity_tests():
    """
    Runs the time complexity tests over various dataset sizes and distributions.
    """
    # Seed the random number generators for reproducibility
    random.seed(42)
    np.random.seed(42)

    dataset_sizes = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    num_trials = 1  # Number of trials per dataset size
    distributions = ['uniform', 'exponential']

    # Dictionary to store average timings
    average_timings = {
        'uniform': [],
        'exponential': []
    }

    for size in dataset_sizes:
        print(f"\nDataset Size: {size}")
        for dist in distributions:
            total_time = 0.0
            for _ in range(num_trials):
                if dist == 'uniform':
                    # Generate uniformly distributed integers between 0 and size
                    data = [random.randint(0, size) for _ in range(size)]
                elif dist == 'exponential':
                    # Generate exponentially distributed real numbers
                    data = np.random.exponential(1, size).tolist()
                else:
                    continue

                # Sort the data and measure the time taken
                _, sort_time = bucket_sort_with_timing(data)
                total_time += sort_time

            avg_time = total_time / num_trials
            average_timings[dist].append(avg_time)
            print(f"  Distribution: {dist}, Average Time: {avg_time:.6f} seconds")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for dist in distributions:
        plt.plot(dataset_sizes, average_timings[dist], marker='o', label=dist.capitalize())
    plt.xlabel('Dataset Size')
    plt.ylabel('Average Sorting Time (seconds)')
    plt.title('Time Complexity of Improved Bucket Sort with Quantile-Based Buckets')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_time_complexity_tests()
