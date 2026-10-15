# Amazon Review Sentiment as a Predictor of Economic Trends

Can the tone of millions of Amazon reviews tell us something about the U.S. economy? This team project builds a PySpark NLP pipeline over roughly 30 GB of Amazon reviews, aggregates sentiment and text features by year, and tests whether they predict year-over-year changes in U.S. GDP per capita growth.

*Team project for DSC 232R (Big Data Analytics) in the UC San Diego M.S. in Data Science program.*

**My role:** contributed to the analysis code, wrote the project's original README write-up, and co-wrote the final report.

**[Full report (PDF)](./Amazon_Review_Sentiment_as_a_Predictor_of_Economic_Trends.pdf)** · **[Final notebook](./Amazon_reviews_GDP_analysis_final.ipynb)**

![Modeling pipeline](./figures/Amazon_Data_Pipeline.png)

## Results at a glance

| Model | Train AUC | Train accuracy | Test AUC | Test accuracy |
|---|---|---|---|---|
| Logistic regression (L1, PySpark MLlib) | 1.000 | 1.000 | 0.833 | 0.800 |

- Year-over-year change in average review sentiment correlates modestly with the GDP signal: **r = 0.22** for essential products and **r = 0.18** for luxury products. The gap between the two barely correlates (r = 0.04).
- Sentiment dropped sharply across all categories in 2020, lining up with the COVID-19 downturn, and the model's prediction for 2020 was correct.
- The test set is only **five years (2019–2023)**, so each year moves accuracy by 0.2. Read these numbers as a proof of concept, not a validated forecaster. See [Caveats](#caveats).

## Data

- **Amazon reviews:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), four categories grouped by spending type:
  - *Essential:* Grocery & Gourmet Food, Baby Products
  - *Luxury / discretionary:* Movies & TV, Video Games
- **Economic data:** World Bank U.S. GDP per capita growth (annual %), 1996–2023

Neutral 3-star reviews are dropped to sharpen the sentiment signal. Review volume grows sharply after 2012, so later years carry most of the data.

## Approach

1. **Text features (PySpark ML).** Tokenize, remove stop words, hash into 5,000-dimensional term-frequency vectors (`HashingTF`), and apply IDF weighting.
2. **Sentiment.** Score each review's polarity (−1 to +1) with TextBlob.
3. **Yearly aggregation by group.** For essential and luxury reviews separately: mean polarity, polarity volatility (standard deviation), review count, and mean TF-IDF vector.
4. **Dimensionality reduction.** PCA to 50 components per group (100 total).
5. **Year-over-year features.** Change in essential sentiment, luxury sentiment, and the essential–luxury sentiment gap.
6. **Label.** 1 if U.S. GDP per capita growth rose compared with the previous year, otherwise 0.
7. **Model.** The 103 features (3 YoY + 100 PCA) are standardized and fed to an L1-regularized logistic regression. Training uses 1997–2018 and testing uses 2019–2023, a chronological split. A linear SVM was also trained on the same features.

## Exploration highlights

![Rating distributions](./figures/Comparison_of_Distribution.png)
*Ratings are highly polarized: 5-star and 1-star reviews dominate, and 3-star reviews are rare in every group.*

![Average ratings and review volume](./figures/Average_Ratings_Comparison.png)
*Average rating by year for each group (left) and review volume over time on a log scale (right).*

![Sentiment change vs. GDP](./figures/Correlation_Graph.png)
*Year-over-year sentiment change against the GDP signal, with Pearson correlations (top) and both series over time (bottom).*

## Caveats

These came out of a careful review of the final notebook and are worth knowing before drawing conclusions:

- **Very small test set.** Five years means one wrong prediction costs 0.2 accuracy, and the AUC rests on very few positive/negative pairs.
- **Preprocessing sees the test years.** PCA and `StandardScaler` are fit on all years, including 2019–2023, before the split. With so few rows this leaks information from the test period into the features. Fitting both on 1997–2018 only would make the evaluation strictly forward-looking.
- **Cross-validation didn't support the chosen settings.** In 3-fold cross-validation on the training years, every regularization setting scored an AUC of 0.50 or lower, and the selected setting (`regParam = 0.01`) scored about 0.23. The folds are random rather than chronological, which partly explains this, but the cross-validation results don't by themselves back the test-set score.
- **The label measures acceleration, not growth.** The World Bank series is already a growth rate, so a label of 1 means growth *sped up* versus the prior year, not that the economy grew.
- **SVM results need a re-run.** The SVM cell originally printed the logistic regression's metrics by mistake, which is why the two models once appeared identical. The print statement is fixed; the SVM's own numbers require re-running the notebook.

**Next steps:** fit PCA and scaling on training years only, use time-series cross-validation, aggregate quarterly or monthly for more data points, and compare against a majority-class baseline and traditional indicators like consumer confidence.

## Repository contents

| Path | Description |
|---|---|
| `Amazon_reviews_GDP_analysis_final.ipynb` | Full pipeline: loading, NLP features, aggregation, modeling, evaluation |
| `Data_download.ipynb` | Downloads and unzips the review categories into `datasets/` |
| `Amazon_Review_Sentiment_as_a_Predictor_of_Economic_Trends.pdf` | Final written report |
| `figures/` | Figures used in the report and this README |
| `milestones/` | Earlier course milestones: data exploration, GDP and indicator exploration, first model |
| `requirements.txt` | Python dependencies |

## How to run

Requires Java (for Spark) and a machine with substantial memory; the original run used a 110 GB driver.

```bash
pip install -r requirements.txt
jupyter notebook Data_download.ipynb                        # downloads the review data
jupyter notebook Amazon_reviews_GDP_analysis_final.ipynb    # fetches the GDP CSV and runs the analysis
```

Lower `spark.driver.memory` in the notebook's first cells to fit your machine.
