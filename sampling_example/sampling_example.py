import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def heuristic_sampling(mean, stdev, num_samples):
    # this takes in a sample distribution and resamples it baed on our standard deviation heurtic
    num_seed = math.ceil(num_samples/2)
    num_diverse = math.floor(num_samples/2)
    diverse_samples = []
    seed_samples = np.random.normal(loc=mean, scale=stdev, size=num_seed)
    test_samples = np.random.normal(loc=mean, scale=stdev, size=20*num_samples)
    seed_std = np.std(seed_samples, ddof=1)
    # print(f"stdev: {seed_std}")
    mins = []
    # average seed diff
    seed_diffs = []
    min_diffs = []
    for i, s1_sample in enumerate(seed_samples):
        min_diff = []
        temp_diffs = []
        for j, s2_sample in enumerate(seed_samples):
            if i != j:
                seed_diff = diff = abs(s1_sample - s2_sample)
                seed_diffs.append(seed_diff)
                temp_diffs.append(seed_diff)
        min_diff = min(temp_diffs)
        min_diffs.append(min_diff)
            
    min_diffs_set = list(set(min_diffs))
    mean_min_diff = np.mean(min_diffs_set)
    # print(f"mean min diff: {mean_min_diff}")

    # remove half the difference matrix
    seed_diffs_set = list(set(seed_diffs))
    mean_diff = np.mean(seed_diffs_set)
    # print(f"mean diff: {mean_diff}")

    while len(diverse_samples) < num_diverse:
        for sample in test_samples:
            diffs = []
            for seed_sample in seed_samples:
                diff = abs(sample - seed_sample)
                diffs.append(diff)
                if diff >= mean_min_diff:
                    diverse_samples.append(sample)
            mins.append(min(diffs))

    new_samples = seed_sample + diverse_samples

    # plt.figure()
    # plt.hist(mins, bins=30, density=True, alpha=0.6, color='g')
    # plt.axvline(mean_min_diff, color='red', linestyle='dashed', linewidth=2)
    # plt.savefig("./figs/sampling_example/test_sampling.pdf", dpi=300, bbox_inches="tight")
    # plt.close()

    return new_samples, mins, mean_min_diff

if __name__ == "__main__":
    # initialization of random sampling
    num_samples = 800
    num_recursions = 300
    means_rand = [0]
    std_devs_rand = [1]

    for i in range(0, num_recursions):
        samples = np.random.normal(loc=means_rand[0], scale=std_devs_rand[i], size=num_samples)
        means_rand.append(samples.mean())
        std_dev = np.std(samples, ddof=1)
        std_devs_rand.append(std_dev)

    recursion_range = range(0, num_recursions+1)
    recursion_vec = list(recursion_range)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # 1 row, 2 columns

    # First subplot
    axes[0].scatter(recursion_vec, means_rand, color='blue', alpha=0.7)
    axes[0].axhline(np.mean(means_rand), color='orange', linestyle='dashed', linewidth=2)
    axes[0].set_title("Mean over Recursive Random Sampling")
    axes[0].set_xlabel("Recursion")
    axes[0].set_ylabel("Mean")

    # Second subplot
    axes[1].scatter(recursion_vec, std_devs_rand, color='green', alpha=0.7)
    axes[1].set_title("Standard Deviation over Recursive Random Sampling")
    axes[1].set_xlabel("Recursion")
    axes[1].set_ylabel("Standard Deviation")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig("./figs/sampling_example/random_sampling.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # initialization of heuristic sampling
    num_samples = 250
    num_recursions = 300
    means_heur = [0]
    std_devs_heur = [1]
    mean_min_diffs = []

    for i in tqdm(range(0, num_recursions), total=num_recursions):
        samples = np.random.normal(loc=means_heur[0], scale=std_devs_heur[i], size=num_samples)
        means_rand.append(samples.mean())
        std_dev = np.std(samples, ddof=1)
        std_devs_rand.append(std_dev)
        new_samples, mins, mean_min_diff = heuristic_sampling(means_heur[0], std_devs_heur[i], num_samples)
        mean_min_diffs.append(mean_min_diff)
        means_heur.append(new_samples.mean())
        std_dev = np.std(new_samples, ddof=1)
        std_devs_heur.append(std_dev)


    recursion_range = range(0, num_recursions+1)
    recursion_vec = list(recursion_range)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # 1 row, 2 columns

    # First subplot
    axes[0].scatter(recursion_vec, means_heur, color='blue', alpha=0.7)
    axes[0].axhline(np.mean(means_heur), color='orange', linestyle='dashed', linewidth=2)
    axes[0].set_title("Mean over Recursive Heuristic Sampling")
    axes[0].set_xlabel("Recursion")
    axes[0].set_ylabel("Mean")

    # Second subplot
    axes[1].scatter(recursion_vec, std_devs_heur, color='green', alpha=0.7)
    axes[1].set_title("Standard Deviation over Recursive Heuristic Sampling")
    axes[1].set_xlabel("Recursion")
    axes[1].set_ylabel("Standard Deviation")

    # Third subplot
    axes[2].scatter(recursion_vec[1:], mean_min_diffs, color='purple', alpha=0.7)
    axes[2].set_title("Mean of Min Difference between data over recursions")
    axes[2].set_xlabel("Recursion")
    axes[2].set_ylabel("Mean Min Difference")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig("./figs/sampling_example/heuristic_sampling.pdf", dpi=300, bbox_inches="tight")
    plt.close()