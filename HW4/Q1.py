import random


data = [65, 70, 68, 73, 75, 80, 85, 78, 90, 100]

def PartA():
    mean = sum(data) / len(data)
    print(f"Mean: {mean}")
    
    std = 0
    for i in data:
        std += (i - mean) ** 2
    std = (std / (len(data) - 1)) ** 0.5
    print(f"Standard Deviation: {std}")
    
    t = 2.262
    ci_lower = mean - t * (std / (len(data) ** 0.5))
    ci_higher = mean + t * (std / (len(data) ** 0.5))
    print(f"95% Confidence Interval: ({ci_lower}, {ci_higher})")
PartA()
    
def construct_bootstrap(data, n_bootstrap):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(len(data))]
        bootstrap_means.append(sum(sample) / len(sample))
    return bootstrap_means

def PartB():
    bootstrap_means = construct_bootstrap(data, 40)
    bootstrap_means.sort()
    
    lower_idx = int(0.025 * len(bootstrap_means))
    upper_idx = int(0.975 * len(bootstrap_means))
    ci_lower = bootstrap_means[lower_idx]
    ci_higher = bootstrap_means[upper_idx]
    
    print(f"Bootstrap 95% Confidence Interval: ({ci_lower}, {ci_higher})")
PartB()