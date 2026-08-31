# Ingredients, Preparation, and Recipe Steps

**Yifeng Zheng**

## Introduction

This project studies recipes and user ratings collected from Food.com. The cleaned dataset contains 83,782 recipes, with one row per recipe. I focus on the question: **Which recipe characteristics are associated with the number of steps in a recipe, and how well can they predict it?**

I use `n_steps` as the response and as a simple measure of recipe complexity. It counts how many instructions a cook must complete, though it does not measure how difficult each instruction is. The most relevant columns are:

- `n_steps`: number of instructions in a recipe.
- `n_ingredients`: number of ingredients.
- `minutes`: stated preparation time.
- `nutrition`: calories and six percentage-of-daily-value measurements.
- `description`: the author's description of the recipe.
- `rating`: an individual user's score, aggregated into `average_rating` for each recipe.

## Data Cleaning and Exploratory Data Analysis

The interaction table has one row per submitted rating, so one recipe can appear many times. Of all interaction rows, 234,428 match the 83,782 recipes in the supplied recipe table; the others refer to recipes outside this subset. I left-merged the matching ratings onto recipes so every supplied recipe was retained. Ratings equal to 0 are outside the valid 1-5 scale and act as missing-rating placeholders, so I replaced them with `NaN`. I then computed the mean valid rating for each recipe and added that Series back to the recipe table. This restores one row per recipe and prevents heavily reviewed recipes from being counted repeatedly.

I also split the `nutrition` string into seven numerical columns so values such as `calories` and `protein_pdv` can be analyzed. After cleaning, only `name` (1 value), `description` (70 values), and `average_rating` (2,609 values) are missing. I preserve missing `average_rating` for the missingness analysis, do not use `name` in the model, and handle missing descriptions inside the model Pipeline.

| name | minutes | n_steps | n_ingredients | calories | average_rating |
|---|---:|---:|---:|---:|---:|
| 1 brownies in the world best ever | 40 | 10 | 9 | 138.4 | 4.0 |
| 1 in canada chocolate chip cookies | 45 | 12 | 11 | 595.1 | 5.0 |
| 412 broccoli casserole | 40 | 6 | 9 | 194.8 | 5.0 |
| millionaire pound cake | 120 | 7 | 7 | 878.3 | 5.0 |
| 2000 meatloaf | 90 | 17 | 13 | 267.0 | 5.0 |

The number of steps is right-skewed: most recipes have relatively few steps, while a small group is much longer.

<iframe
  src="assets/steps-distribution.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

Mean step count generally rises with ingredient count. The plot includes ingredient counts represented by at least 100 recipes, so isolated rare counts do not dominate the pattern. This is an association, not evidence that adding an ingredient causes another step.

<iframe
  src="assets/ingredient-steps.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

The grouped table gives the same pattern in exact values. Recipes with 1-5 ingredients average 6.56 steps, compared with 16.64 steps for recipes with at least 16 ingredients. Median preparation time and median calories also rise across these groups. I use medians for time and calories because both columns have long right tails and a small number of very large values.

| Ingredient count | Recipes | Mean steps | Median minutes | Median calories |
|---|---:|---:|---:|---:|
| 1-5 | 14,164 | 6.56 | 17 | 200.20 |
| 6-10 | 41,623 | 9.25 | 35 | 288.20 |
| 11-15 | 22,810 | 12.38 | 45 | 368.55 |
| 16+ | 5,185 | 16.64 | 60 | 468.50 |

## Assessment of Missingness

`average_rating` may be **MNAR** if the chance that a recipe receives a valid rating depends on the rating it would have received. For example, recipes that users would rate very positively or negatively may be more likely to prompt a rating than recipes that create little reaction. Page views, search impressions, favorites, and recommendation exposure would help measure whether a recipe had an opportunity to receive a rating; with those observed variables, MAR may become more plausible.

I used permutation tests with the K-S statistic to compare the full numerical distributions when `average_rating` is missing and present. The null hypothesis is that missingness in `average_rating` is independent of the comparison column; the alternative is that the two distributions differ. For `n_steps`, the observed K-S statistic is 0.0760. None of the 1,000 simulated statistics was as large, so the simulation reports p < 0.001 at this resolution. This provides evidence that missingness in `average_rating` depends on the distribution of `n_steps`. For `protein_pdv`, the observed K-S statistic is 0.0175 and the empirical p-value is 0.320, so I fail to reject the null hypothesis. This is a lack of evidence that the distributions differ, not proof of independence.

The conditional step distributions below provide visual context for the first test.

<iframe
  src="assets/missingness-steps.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

## Hypothesis Testing

I tested whether recipes above the median ingredient count have more steps on average.

- **Null hypothesis:** Ingredient-group labels are independent of `n_steps`; recipes with more than 9 ingredients and recipes with at most 9 ingredients have the same `n_steps` distribution, so the observed difference in means is due to chance.
- **Alternative hypothesis:** Recipes with more than 9 ingredients have a larger mean number of steps.
- **Test statistic:** mean steps in the high-ingredient group minus mean steps in the low-ingredient group.
- **Significance level:** 0.05.

The statistic is directional because the alternative specifically says “larger.” The two means are 12.629 and 8.202 steps, an observed difference of 4.427. None of the 5,000 simulated statistics was as large, so p < 0.0002 at this simulation resolution. I reject the null hypothesis. The data provide strong evidence of an association, but this observational test does not establish causation.

<iframe
  src="assets/ingredient-hypothesis.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

## Framing a Prediction Problem

I predict `n_steps`, making this a regression problem. The prediction is made while a recipe listing is being drafted, after its expected time, ingredient count, nutrition information, and description are known, but before its detailed instructions are examined. I evaluate models with RMSE because it measures prediction error in steps and gives extra weight to large mistakes.

The text in `steps` is excluded because it directly reveals the response. `average_rating` is excluded because ratings are generated after publication. Recipe and contributor IDs are also excluded because they identify records rather than describe recipe complexity. I use a fixed 75% training and 25% test split for both models.

## Baseline Model

The baseline is a depth-3 decision tree in a single scikit-learn Pipeline. It uses two original quantitative features, `minutes` and `n_ingredients`; neither needs categorical encoding.

- Training RMSE: 5.573 steps.
- Test RMSE: 5.627 steps.

The similar training and test values do not indicate severe overfitting, but an error of about 5.6 steps is large relative to the median recipe's nine steps. The model is therefore a useful baseline rather than a strong final result.

## Final Model

The final model is also a `DecisionTreeRegressor`. Its Pipeline retains the two baseline features and uses two additional features:

- `calories`, parsed once from `nutrition` during data cleaning and then reused directly by the model, because caloric content may reflect how substantial a recipe is.
- `description_word_count`, because recipes that need more explanation may also require more instructions.

`calories` is created once during general data cleaning. The final Pipeline reuses that cleaned column while performing all model-specific feature assembly and fitting. Before using the test set, I used 5-fold cross-validation on the training set to search `max_depth` in 3, 5, 7, and 10 and `min_samples_leaf` in 1, 10, and 50. The best choice within this grid was `max_depth=7` and `min_samples_leaf=50`.

| Model | Training RMSE | Test RMSE |
|---|---:|---:|
| Baseline | 5.573 | 5.627 |
| Final | 5.346 | 5.476 |

The final test RMSE is 2.68% lower than the baseline test RMSE on the same held-out recipes. The improvement is modest, but it is measured fairly and the added features have a data-based interpretation.

## Fairness Analysis

I compared the final model for recipes with more than 9 ingredients (Group X) and recipes with at most 9 ingredients (Group Y). The cutoff is the median ingredient count in the training data.

- **Null hypothesis:** The model is fair for these groups; their RMSE values are approximately equal, and the observed difference is due to random group labels.
- **Alternative hypothesis:** The model performs worse for Group X, so its RMSE is higher.
- **Test statistic:** RMSE(Group X) minus RMSE(Group Y).
- **Significance level:** 0.05.

I kept the final fitted model and test predictions fixed and shuffled only the group labels. Group X has RMSE 6.455, compared with 4.605 for Group Y. The observed difference is 1.850, and none of the 5,000 simulated statistics was as large, so p < 0.0002 at this simulation resolution. I reject the null hypothesis and find evidence that the model is less accurate for recipes with many ingredients. These recipes also have more variable step counts, which may contribute to the gap, but the test does not identify its cause. A useful next step would be to add features that better describe complex recipes without using the leaked `steps` text.

<iframe
  src="assets/fairness-test.html"
  width="100%"
  height="500"
  frameborder="0"
></iframe>

