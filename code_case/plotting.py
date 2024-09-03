import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    # comparing few-shot verse zero-shot diversity scores

    # importing the token data
    token_class_few_full = np.load("./dataset/new_generator/token_class_few_full/distances.npz")
    token_class_few_full_zero = token_class_few_full['array1']
    token_class_few_full_few = token_class_few_full['array2']

    token_class_zero_full = np.load("./dataset/new_generator/token_class_zero_full/distances.npz")
    token_class_zero_full_zero = token_class_few_full['array1']
    
    token_classless_few_full = np.load("./dataset/new_generator/token_classless_few_full/distances.npz")
    token_classless_few_full_zero = token_classless_few_full['array1']
    token_classless_few_full_few = token_classless_few_full['array2']

    token_full_zero_class = np.concatenate((token_class_few_full_zero, token_class_zero_full_zero))

    # importing the tree data
    tree_class_few = np.load("./dataset/new_generator/tree_class_few/distances.npz")
    tree_class_few_zero = tree_class_few['array1']
    tree_class_few_tree = tree_class_few['array2']

    tree_class_zero = np.load("./dataset/new_generator/tree_class_zero/distances.npz")
    tree_class_zero_zero = tree_class_zero['array1']
    
    tree_classless_few = np.load("./dataset/new_generator/tree_classless_few/distances.npz")
    tree_classless_few_zero = tree_classless_few['array1']
    tree_classless_few_tree = tree_classless_few['array2']

    tree_full_zero_class = np.concatenate((tree_class_few_zero, tree_class_zero_zero))


    # plotting
    # token-class few vs zero
    plt.figure(figsize=(12, 10))

    plt.scatter(range(0,len(token_full_zero_class)), token_full_zero_class, color='blue', label='Zero')
    plt.scatter(range(0,len(token_class_few_full_few)), token_class_few_full_few, color='red', label='Few')
    plt.title('Token-Class: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Class Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/token_class_few_vs_zero.png")

    # token-classless few vs zero
    plt.clf()
    plt.figure(figsize=(12, 10))

    plt.scatter(range(0,len(token_classless_few_full_zero)), token_classless_few_full_zero, color='blue', label='Zero')
    plt.scatter(range(0,len(token_classless_few_full_few)), token_classless_few_full_few, color='red', label='Few')
    plt.title('Token-Classless: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Classless Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/token_classless_few_vs_zero.png")

    # tree-classless few vs zero
    plt.clf()
    plt.figure(figsize=(12, 10))

    plt.scatter(range(0,len(tree_classless_few_zero)), tree_classless_few_zero, color='blue', label='Zero')
    plt.scatter(range(0,len(tree_classless_few_tree)), tree_classless_few_tree, color='red', label='Few')
    plt.title('Tree-Classless: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Classless Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/tree_classless_few_vs_zero.png")

    # tree-class few vs zero
    plt.clf()
    plt.figure(figsize=(12, 10))

    plt.scatter(range(0,len(tree_full_zero_class)), tree_full_zero_class, color='blue', label='Zero')
    plt.scatter(range(0,len(tree_class_few_tree)), tree_class_few_tree, color='red', label='Few')
    plt.title('Tree-Class: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Class Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/tree_class_few_vs_zero.png")
