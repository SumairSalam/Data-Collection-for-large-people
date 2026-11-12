# Decision Tree Classification from Scratch

An educational Python project exploring how decision trees use **entropy** and **information gain** to select features and build classification rules.

The implementation is provided in a Jupyter Notebook and uses a small transport-choice dataset.

## Overview

The notebook includes a custom `DicisionTree` class that calculates entropy, evaluates information gain, and constructs a tree represented as nested Python lists.

The example explores transport choices using four features:

| Feature | Example values |
|---------|----------------|
| Gender | male, female |
| Ownership Car | 0, 1, 2 |
| Travel Cost | cheap, standerd, expensive |
| Income level | low, medium, high |
| Target: Transport | bus, train, car |

The dataset contains 10 sample records and is defined directly in the notebook.

> The class name `DicisionTree` and category value `standerd` match the current code.

## Technologies

- Python
- Pandas
- Jupyter Notebook
- Python standard library: `math`

## Repository Contents

| File | Description |
|------|-------------|
| `DS_project.ipynb` | Custom decision tree implementation, sample dataset, and search example |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SumairSalam/Data-Collection-for-large-people.git
cd Data-Collection-for-large-people
```

### 2. Install dependencies

```bash
python -m pip install pandas notebook
```

### 3. Open the notebook

```bash
jupyter notebook DS_project.ipynb
```

Run the cells in order to define the class, create the sample dataset, build the tree, and execute the search example.

## Usage

After running the class definition and dataset cells:

```python
# Separate input features and target labels
y = df["Transport"]
X = df.drop(["Transport"], axis=1)

# Construct the decision tree
DT = DicisionTree(X, y)

# Create a sample to search
SearchValue = pd.DataFrame({
    "Gender": ["male"],
    "Ownership Car": ["0"],
    "Travel Cost": ["expensive"],
    "Income level": ["medium"]
})

# Search the generated tree
DicisionTree.Search(DT, DT.Tree, SearchValue)
```

The notebook prints the generated tree and the matching branch result. The saved example displays `car`.

## Key Concepts

- **Entropy:** Measures uncertainty in the target labels.
- **Information gain:** Measures how much a feature reduces uncertainty.
- **Recursive tree construction:** Builds branches by selecting features and processing subsets.
- **Leaf nodes:** Store class labels at the ends of branches.

## Current Limitations

This is an experimental learning implementation and requires further validation.

- The example uses a small, categorical dataset.
- No train/test split or evaluation metrics are included.
- Continuous-feature thresholds and missing-value handling are not implemented.
- Recursive subset selection, entropy calculations, and search traversal need review before use with other datasets.
- The search method prints results rather than providing a conventional prediction API.

## Possible Improvements

- Correct and validate recursive splitting and branch traversal.
- Add `fit()` and `predict()` methods.
- Evaluate accuracy on a separate test set.
- Compare results with scikit-learn's `DecisionTreeClassifier`.
- Support numerical features and unseen category values.
- Add tests for entropy, information gain, and predictions.

## Author

[Sumair Salam](https://github.com/SumairSalam)
