# ECON 5200 Final Project: Does Job Training Cause Higher Earnings?

## Overview
This project investigates whether participation in the National Supported Work (NSW) job training program caused higher earnings for participants. Using Double Machine Learning (DML), I estimate the causal effect of the program while controlling for selection bias between the treated group and the CPS comparison population.

## Causal Question
Does participation in a job training program cause higher earnings?

## Data
Dehejia-Wahba version of the Lalonde dataset, hosted on NBER:
- NSW treated group: 185 individuals
- CPS comparison controls: 15,992 individuals
- Total: 16,177 observations
- Source: https://users.nber.org/~rdehejia/nswdata.html

## Key Findings
- Naive OLS estimate (no controls): -$8,498 (biased by selection)
- DML estimate (Gradient Boosting): +$989 (95% CI: -296 to 2,274)
- DML estimate (Random Forest): +$1,530 (95% CI: 370 to 2,691)
- Both estimates are consistent with the experimental benchmark of ~$1,800

## Repo Structure
- `notebooks/` — Jupyter notebook with full analysis
- `data/` — NSW and CPS data files (.dta)
- `app.py` — Streamlit dashboard
- `requirements.txt` — Python dependencies

## Streamlit Dashboard
https://bxuojhfnw4dwkkg7skvye3.streamlit.app

## Tools Used
Python, pandas, scikit-learn, econml, statsmodels, matplotlib, seaborn, Streamlit
