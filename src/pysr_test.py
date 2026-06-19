import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

# Create data with 2 features and 3 categories
X = np.random.uniform(-3, 3, (1000, 2))
category = np.random.randint(0, 3, 1000)

# Parameters for each category
offsets = [0.1, 1.5, -0.5]
scales = [1.0, 2.0, 0.5]

# y = scale[category] * sin(x1) + offset[category]
y = np.array([scales[c] * np.sin(x1) + offsets[c] for x1, c in zip(X[:, 0], category)])
template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={
        "p1": 3,  # length = # of categories
        "p2": 3,
    },  # p1 = offset OR scale, vice versa for p2
    combine="f(x1, x2, p1[category], p2[category])",
)

# 'category + 1' for julia's indexing by 1
X_with_category = np.column_stack([X, category + 1])

model = PySRRegressor(
    expression_spec=template,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin"],
    maxsize=10,
)
model.fit(X_with_category, y)
print(model)
# Predicting on new data
# model.predict(X_test_with_category)
