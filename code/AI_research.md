Question 1

The term "multi-head attention" in Transformers refers to:

✅ C: Running several attention mechanisms in parallel to focus on different parts of the input.

Explanation:
In the Transformer architecture, multi-head attention means the model uses multiple attention heads in parallel. Each head learns to attend to different parts of the input sequence, allowing the model to capture richer contextual relationships. The outputs from all heads are concatenated and linearly transformed.

Question 2

In the context of transformers, the term "scaling" in the scaled dot-product attention refers to:

✅ B: Normalizing the dot products to prevent large gradients and instabilities.

Explanation:
In scaled dot-product attention, the attention score is divided by the square root of the key dimension 𝑑𝑘. This scaling prevents the dot product values from becoming too large when 𝑑𝑘 is large, which would otherwise lead to very small gradients after applying softmax.

Question 3

In the context of neural networks, what is the vanishing gradient problem primarily associated with?

✅ C: Deep networks using sigmoid or tanh activations.

Explanation:
The vanishing gradient problem occurs when gradients become extremely small as they are backpropagated through many layers. This often happens in deep networks with activation functions like sigmoid or tanh, which squash input values into small ranges, making the derivatives very small.

Question 4

Gradient clipping is used to address which of the following issues in training neural networks?

✅ C: Exploding gradients.

Explanation:
Gradient clipping prevents the gradients from growing too large during backpropagation by setting a maximum threshold. This helps stabilize training and is especially useful in RNNs or very deep models prone to exploding gradients.

Question 5

The Bellman equation is fundamental in reinforcement learning because it

✅ B: Defines the relationship between the value of a state and the values of its successor states.

Explanation:
The Bellman equation expresses the recursive relationship between the value of a state and the expected returns (rewards + discounted future values) of subsequent states. It forms the core of dynamic programming and value-based RL methods like Q-learning.

Question 6

In decision trees, the Gini impurity is a measure used to

✅ A: Evaluate the quality of a split.

Explanation:
Gini impurity measures how often a randomly chosen element from a set would be incorrectly labeled if randomly labeled according to the distribution of labels in the subset. Decision tree algorithms (like CART) use it to find the best split.

Question 7

In the context of neural networks, dropout is a regularization technique that

✅ B: Removes random neurons during training to prevent overfitting.

Explanation:
Dropout randomly “drops” (sets to zero) a fraction of neurons during each training step. This prevents the model from relying too heavily on specific neurons, improving generalization and reducing overfitting.

Question 8

Why are optimization problems in deep learning typically non-convex?

✅ C: They involve complex, high-dimensional loss landscapes due to multiple layers and non-linear activations.

Explanation:
Deep neural networks use non-linear activation functions and have many interconnected layers, creating a high-dimensional, non-convex loss surface. This leads to many local minima and saddle points, making optimization more challenging than convex problems like linear regression.

Question 9

Why is the vanishing gradient problem less severe in architectures like ResNets?

✅ B: Residual connections provide a pathway for gradients to flow through the network without diminishing.

Explanation:
ResNets use skip (residual) connections, allowing gradients to bypass layers and flow directly back through the network. This helps preserve gradient magnitude and mitigates the vanishing gradient problem, enabling much deeper networks to train effectively.

Question 10

In PCA, the eigenvectors of the covariance matrix represent:

✅ A: Principal components that capture the maximum variance in data projections.

Explanation:
Principal Component Analysis (PCA) identifies eigenvectors (principal components) that point in the directions of maximum variance in the data. The corresponding eigenvalues represent the amount of variance captured by each component.


## Coding: Implement a Markowitz-style stock portfolio optimizer in Python.

Here’s a clear, complete solution that fulfills all listed requirements (using only pandas and numpy, as allowed):
```
import pandas as pd
import numpy as np

def markowitz_stock_portfolio():
    # 1. Load dataset
    df = pd.read_csv('./data/stock-prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 2. Compute daily percentage returns
    returns = df.pct_change()

    # 3. Compute mean daily returns
    mean_returns = returns.mean()

    # 4. Compute covariance matrix
    cov_matrix = returns.cov()

    # 5. Generate random portfolios
    num_stocks = len(mean_returns)
    num_portfolios = 10000

    results = []
    for _ in range(num_portfolios):
        # Random weights that sum to 1
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)

        # Portfolio mean return
        port_return = np.dot(weights, mean_returns)

        # Portfolio standard deviation (volatility)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = port_return / port_std if port_std != 0 else 0

        results.append(np.concatenate((weights, [port_std, sharpe_ratio])))

    # Convert results to DataFrame
    columns = list(mean_returns.index) + ['std', 'sharpe']
    portfolio_df = pd.DataFrame(results, columns=columns)

    # 6. Portfolio with minimum risk (lowest std)
    min_std_portfolio = portfolio_df.loc[portfolio_df['std'].idxmin()]

    # 7. Portfolio with maximum Sharpe ratio
    max_sharpe_portfolio = portfolio_df.loc[portfolio_df['sharpe'].idxmax()]

    # 8. Create output dictionary
    output = {
        'returns': returns,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'optimal_portfolios': pd.DataFrame(
            [min_std_portfolio, max_sharpe_portfolio],
            index=['min_std', 'max_sharpe']
        )
    }

    return output
```

## Coding: class-based machine learning pipeline design under class imbalance, where you can modify data preprocessing (but not the LinearSVC model itself).

Let’s write a high-scoring implementation that maximizes ROC-AUC.
```
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class ClassifierWithImbalanceClass:
    def __init__(self):
        # Create a pipeline with scaling, SMOTE, and LinearSVC
        self._pipeline = make_pipeline(
            StandardScaler(),
            SMOTE(random_state=42, sampling_strategy='auto'),
            LinearSVC(random_state=42)
        )

    def train(self, X, y):
        # Fit the model using the pipeline
        self._pipeline.fit(X, y)

    def predict(self, X):
        # Predict using the trained pipeline
        return self._pipeline.predict(X)
```

Extra Boost (optional, if allowed)

If you’re allowed to further tune preprocessing, you can add feature selection:
```
from sklearn.feature_selection import SelectKBest, f_classif

self._pipeline = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif, k=150),
    SMOTE(random_state=42),
    LinearSVC(random_state=42)
)

```




