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

    # Define the bin edges for both datasets
    bins = np.linspace(min(min(token_full_zero_class), min(token_class_few_full_few)), max(max(token_full_zero_class), max(token_class_few_full_few)), 30)

    plt.figure(figsize=(12, 10))

    plt.hist(token_full_zero_class, bins="auto", alpha=0.8, color='blue', edgecolor='blue', label='Zero', density=True, histtype='step')
    plt.hist(token_class_few_full_few, bins="auto", alpha=0.8, color='red', edgecolor='red', label='Few', density=True, histtype='step')
    plt.title('Token-Class: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Class Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/token_class_few_vs_zero.png")

    # token-classless few vs zero
    bins = np.linspace(min(min(token_classless_few_full_zero), min(token_classless_few_full_few)), max(max(token_classless_few_full_zero), max(token_classless_few_full_few)), 30)
    plt.clf()
    plt.figure(figsize=(12, 10))

    plt.hist(token_classless_few_full_zero, bins=bins, alpha=0.5, color='blue', edgecolor='black', label='Zero', density=True)
    plt.hist(token_classless_few_full_few, bins=bins, alpha=0.5, color='red', edgecolor='black', label='Few', density=True)
    plt.title('Token-Classless: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Classless Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/token_classless_few_vs_zero.png")

    # tree-classless few vs zero
    bins = np.linspace(min(min(tree_classless_few_zero), min(tree_classless_few_tree)), max(max(tree_classless_few_zero), max(tree_classless_few_tree)), 30)    

    plt.clf()
    plt.figure(figsize=(12, 10))

    plt.hist(tree_classless_few_zero, bins=bins, alpha=0.5, color='blue', edgecolor='black', label='Zero', density=True)
    plt.hist(tree_classless_few_tree, bins=bins, alpha=0.5, color='red', edgecolor='black', label='Few', density=True)
    plt.title('Tree-Classless: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Classless Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/tree_classless_few_vs_zero.png")

    # tree-class few vs zero
    bins = np.linspace(min(min(tree_full_zero_class), min(tree_class_few_tree)), max(max(tree_full_zero_class), max(tree_class_few_tree)), 30)  
    plt.clf()
    plt.figure(figsize=(12, 10))

    plt.hist(tree_full_zero_class, bins=bins, alpha=0.5, color='blue', edgecolor='black', label='Zero', density=True)
    plt.hist(tree_class_few_tree, bins=bins, alpha=0.5, color='red', edgecolor='black', label='Few', density=True)
    plt.title('Tree-Class: Few vs. Zero')
    plt.xlabel('Data Point')
    plt.ylabel('Class Based Diversity Metric')
    plt.legend()
    plt.grid(True)

    plt.savefig("./figs/tree_class_few_vs_zero.png")
