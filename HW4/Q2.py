import random

data = [65, 70, 68, 73, 75, 80, 85, 78, 90, 100]

def PartA():
    random.seed(42)
    subsets: list[list[int]] = [[] for _ in range(5)]
    i = 0
    while i < len(data):
        target = random.randint(0, 4)
        if len(subsets[target]) < 2:
            subsets[target].append(data[i])
            i += 1
            
    for i, subset in enumerate(subsets):
        print(f"Subset {i + 1}: {subset}")
        
    return subsets
        
def PartB():
    training_scores = []
    subsets = PartA()
    
    for i in range(5):
        testing_set = subsets[i]
        training_set = subsets[:i] + subsets[i+1:]
        training_ints = []
        for l in training_set:
            training_ints.extend(l)

        training_scores.append(sum(training_ints) / len(training_ints))
        print(f"Testing Set {i + 1}: {testing_set}")
        print(f"Training Set {i + 1}: {training_ints}")
        print(f"Average Training Score {i + 1}: {training_scores[-1]}\n")
        
    return training_scores

def PartC():
    subsets = PartA()
    training_scores = PartB()
    difference_list = []
    
    for i in range(5):
        testing_set = subsets[i]
        avg_training_score = training_scores[i]
        differences = []
        for score in testing_set:
            difference = abs(score - avg_training_score)
            differences.append(difference)
        
        print(f"Differences for Testing Set {i + 1}: {differences}")
        mean_difference = sum(differences) / len(differences)
        print(f"Mean Difference for Testing Set {i + 1}: {mean_difference}")
        print()
        difference_list.append(mean_difference)

    return difference_list

def PartD():
    difference_list = PartC()
    overall_mean_difference = sum(difference_list) / len(difference_list)
    print(f"Overall Mean Difference: {overall_mean_difference}")

def PartE():
    mean_all = sum(data) / len(data)
    print(f"Overall Mean of Data: {mean_all}")
    abs_errors = [abs(x - mean_all) for x in data]
    sum_abs_errors = sum(abs_errors)
    print(f"Sum of Absolute Errors predicting by overall mean: {sum_abs_errors}")
    mae = sum(abs_errors) / len(abs_errors)
    print(f"Mean Absolute Error predicting by overall mean: {mae}")
    return mae

PartE()