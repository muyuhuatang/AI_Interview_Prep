🧮 Question 1: Confusion Matrices — Recall > 90%, FPR < 10%

Recall = TP / (TP + FN)
False Positive Rate (FPR) = FP / (FP + TN)

Let’s compute:

Option 1
True	Pred 0	Pred 1
0	82	7
1	19	77

Recall = 77 / (77 + 19) = 0.802 → 80.2%

FPR = 7 / (7 + 82) = 7.9%

✅ FPR < 10%, ❌ Recall < 90% → Fails recall

Option 2
True	Pred 0	Pred 1
0	84	10
1	8	83

Recall = 83 / (83 + 8) = 91.2%

FPR = 10 / (10 + 84) = 10.6%

✅ Recall > 90%, ❌ FPR slightly above 10% → Fails FPR

Option 3
True	Pred 0	Pred 1
0	72	7
1	8	98

Recall = 98 / (98 + 8) = 92.4%

FPR = 7 / (7 + 72) = 8.9%

✅ Recall > 90%, ✅ FPR < 10% → ✅ Passes both thresholds

Option 4
True	Pred 0	Pred 1
0	71	12
1	8	94

Recall = 94 / (94 + 8) = 92.1%

FPR = 12 / (12 + 71) = 14.5%

❌ FPR too high

Option 5
True	Pred 0	Pred 1
0	71	17
1	16	81

Recall = 81 / (81 + 16) = 83.5%

FPR = 17 / (17 + 71) = 19.3%

❌ Fails both

✅ Correct Answer: Option 3

🧩 Question 2: Ensemble Learning

Prompt says:

Please select all appropriate options that should be considered for using ensemble learning.

Let’s analyze each:

❌ “If dataset contains both linear and non-linear relationships, ensemble learning typically results in lower performance” → False, it usually helps.

❌ “Improves overall interpretability” → Ensemble models reduce interpretability.

❌ “Typically creates overfitted models” → Usually reduces overfitting.

✅ “If dataset contains both linear and non-linear relationships, ensemble learning is useful for combining them” → True.

✅ “Can be time intensive to train” → True.

✅ Correct Answer: Option 4 and Option 5

🌳 Question 3: Decision Tree — Impurity Measures

Impurity measures used in decision trees:

Gini Index

Entropy

Classification Error (less common but valid)

✅ Correct Answers: Option 2 (Entropy) and Option 4 (Gini Index)
(Classification Error can be used, but usually not preferred — depends if multiple selections allowed.)

If only one: choose Gini Index (most common default).

🧠 Question 5: Regularization Penalties

“Some coefficients in the model have been zeroed out.”

That behavior corresponds to L1 regularization (Lasso).

✅ Correct Answer: Option 2 (L1 norm)

🧠 Question 6 — Logistic Regression Loss Function

You’re training a binary classifier (logistic regression). Which loss is appropriate?

Option	Loss Function	Correct?	Reason
1	Mean Squared Error	❌	Used for regression, not classification
2	Hinge	❌	Used for SVMs
3	Mean Absolute Error	❌	Used for regression
4	Cross-Entropy	✅	Standard for logistic regression
5	KL Divergence	⚠️ Related but not typically used directly	
6	None of the above	❌	

✅ Correct Answer: Option 4 (Cross-Entropy)

🧮 Question 7 — Simple Neural Network Calculation

Let’s compute step by step (using your image):

Given:

Inputs: 9, 2, 7

Hidden Layer:

𝑓1=9(0.3)+2(0.4)+7(0.1)=2.7+0.8+0.7=4.2

𝑓
2
=
9
(
0.3
)
+
2
(
0.2
)
+
7
(
0.5
)
=
2.7
+
0.4
+
3.5
=
6.6
f
2
	​

=9(0.3)+2(0.2)+7(0.5)=2.7+0.4+3.5=6.6
(both linear activations, so keep as-is)

Output layer:

Input to 
𝑓
3
=
4.2
(
0.1
)
+
6.6
(
0.2
)
=
0.42
+
1.32
=
1.74
f
3
	​

=4.2(0.1)+6.6(0.2)=0.42+1.32=1.74

𝑓
3
f
3
	​

 has a sigmoid activation:

𝜎
(
1.74
)
=
1
1
+
𝑒
−
1.74
≈
0.850
σ(1.74)=
1+e
−1.74
1
	​

≈0.850

✅ Final output: ≈ 0.850




