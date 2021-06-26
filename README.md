# Survival Analysis Paper List
A list of papers/resources in Survival Analysis that we have read or would like to read.

Last Update Time: 2021.06.25
- [Sections](#Survival-Analysis-Paper-List)
    - [Literature Survey](#Literature-Survey)
	- [ML and DL for Survival Analysis](#ML-and-DL-for-Survival-Analysis)
    - [Time-varying Covariates Models](#Time-varying-Covariates-Models)
    - [Competing Risks Models](#Competing-Risks-Models)
    - [Generalized Survival Analysis Methods](#Generalized-Survival-Analysis-Methods)
	- [Evaluation Metrics](#Evaluation-Metrics)
    - [Temporal Time Process](#Temporal-Time-Process)
    - [Applied Survival Analysis](#Applied-Survival-Analysis)

***

## Literature Survey

|Title|Publisher|Date|Code|Notes|
|----|---|--|---|---|
|[Machine Learning for Survival Analysis: A Survey](https://arxiv.org/abs/1708.04649)|ACM Computing Surveys|2019.02||[Slides](https://dmkd.cs.vt.edu/TUTORIAL/Survival/)|
|[Calibration: the Achilles heel of predictive analytics](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7)|BMC Medicine|2019.12|||
|[A tutorial on calibration measurements and calibration models for clinical prediction models](https://academic.oup.com/jamia/article/27/4/621/5762806)|JAMIA|2020.02|[R](https://github.com/easonfg/cali_tutorial)||

## ML and DL for Survival Analysis
|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|GBMCOX|[The State of Boosting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.22.276&rep=rep1&type=pdf)|Computing Science and Statistics|2008|[R](https://github.com/gbm-developers/gbm)||
|RSF|[Random Survival Forest](https://arxiv.org/pdf/0811.1645.pdf)|The Annals of Applied Statistics|2008|[R](https://kogalur.github.io/randomForestSRC/)||
|MTLR|[Learning Patient-Specific Cancer Survival Distributions as a Sequence of Dependent Regressors](http://www.cs.cornell.edu/~cnyu/papers/nips11_survival.pdf)|NeurIPS|2011|[R](https://cran.r-project.org/web/packages/MTLR/index.html)|[Poster](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.8967&rep=rep1&type=pdf)|
|GBMCI|[A Gradient Boosting Algorithm for Survival Analysis via Direct Optimazation of Concordance Index](https://www.hindawi.com/journals/cmmm/2013/873595/)|Computational and Mathematical Methods in Medicine|2013.09|[R](https://github.com/uci-cbcl/GBMCI)||
|N-MTLR|[Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework](https://arxiv.org/abs/1801.05512)|Arxiv|2018.01|[Python](https://square.github.io/pysurvival/)||
|DeepSurv|[DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1)|BMC Medical Research Methodology|2018.02|[Python](https://github.com/jaredleekatzman/DeepSurv)||
|CRPS|[Countdown Regression: Sharp and Calibrated Survival Predictions](https://arxiv.org/abs/1806.08324)|UAI|2019|[PyTorch](https://github.com/stanfordmlgroup/cdr-mimic)||
|SPIE|[Simultaneous Prediction Intervals for Patient-Specific Survival Curves](https://www.ijcai.org/Proceedings/2019/0828.pdf)|IJCAI|2019|[Python](https://github.com/ssokota/spie)||
|CoxTime / CoxCC|[Time-to-Event Prediction with Neural Networks and Cox Regression](https://jmlr.org/papers/volume20/18-424/18-424.pdf)|JMLR|2019.08|[PyTorch](https://github.com/havakv/pycox)|PyCox 1-3|
|PCHazard / LogisticHazard|[Continuous and Discrete-Time Survival Prediction with Neural Networks](https://arxiv.org/abs/1910.06724)|Arxiv|2019.10|[PyTorch](https://github.com/havakv/pycox)|PyCox 2-3|
|X-CAL|[X-CAL: Explicit Calibration for Survival Analysis](https://arxiv.org/abs/2101.05346)|NeurIPS|2020|[PyTorch](https://github.com/rajesh-lab/X-CAL)||
|Discrete-RPS|[Estimating Calibrated Individualized Survival Curves with Deep Learning](https://www.aaai.org/AAAI21Papers/AAAI-8472.KamranF.pdf)|AAAI|2021.02|[PyTorch](https://github.com/MLD3/Calibrated-Survival-Analysis)||
|DCM|[Deep Cox Mixtures for Survival Regression](https://arxiv.org/pdf/2101.06536.pdf)|NeurIPS Machine Learning for Health Workshop|2021.01|[TensorFlow](https://github.com/chiragnagpal/deep_cox_mixtures)||
|DHBN|[Using Discrete Hazard Bayesian Networks to Identify which Features are Relevant at each Time in a Survival Prediction Model](http://proceedings.mlr.press/v146/kuan21a/kuan21a.pdf)|AAAI Spring Symposium (SP-ACA)|2021.03|[R](https://github.com/kuan0911/ISDEvaluation)||
|TDSA|[Transformer-Based Deep Survival Analysis](http://proceedings.mlr.press/v146/hu21a/hu21a.pdf)|AAAI Spring Symposium (SP-ACA)|2021.03|||

***

## Time-varying Covariates Models
|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|Time-varying Cox|[Time-varying covariates and coefficients in Cox regression models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6015946/)|Annals of Translational Medicine|2018.01|||
|DRSA|[Deep Recurrent Survival Analysis](https://arxiv.org/abs/1809.02403)|AAAI|2019.02|[TensorFlow](https://github.com/rk2900/DRSA)||
|TSNN|[Time-Dependent Survival Neural Network for Remaining Useful Life Prediction](https://link.springer.com/chapter/10.1007/978-3-030-16148-4_34)|PAKDD|2019.03|||
|DRSM|[Deep Parametric Time-to-Event Regression with Time-Varying Covariates](http://proceedings.mlr.press/v146/nagpal21a.html)|AAAI Spring Symposium (SP-ACA)|2021.03|[PyTorch](https://autonlab.github.io/DeepSurvivalMachines/#deep-recurrent-survival-machines)||
***

## Competing Risks Models
|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|DeepHit|[DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit.pdf)|AAAI|2018.02|||
***

## Generalized Survival Analysis Methods
|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|---|--|---|---|
||[Adapting machine learning techniques to censored time-to-event health record data: A general-purpose approach using inverse probability of censoring weighting](https://www.sciencedirect.com/science/article/pii/S1532046416000496)|Journal of Biomedical Informatics|2016.03|[R](https://github.com/docvock/JBI_IPCW_for_ML)||
||[A General Machine Learning Framework for Survival Analysis](https://arxiv.org/abs/2006.15442)|ECML|2020.06|[R](https://github.com/adibender/machine-learning-for-survival-ecml2020)||

***

## Evaluation Metrics
|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|---|--|---|---|
|Brier Score Decomposition|[The Comparison and Evaluation of Forecasters](https://www.jstor.org/stable/2987588)|The Statistician|1983|||
|CRPS Decomposition|[Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems](https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml?tab_body=abstract-display)|Weather and Forecasting|2000.03|||
|IPCW Brier Score|[Assessment and Comparison of Prognostic Classification Schemes for Survival Data](https://pubmed.ncbi.nlm.nih.gov/10474158/)|Statistics in medicine|1999.09|||
|D-Calibration|[Effective Ways to Build and Evaluate Individual Survival Distributions](https://jmlr.org/papers/volume21/18-772/18-772.pdf)|JMLR|2020.06|[R](https://github.com/haiderstats/ISDEvaluation)||
|Administrative Brier Score|[The Brier Score under Administrative Censoring: Problems and Solutions](https://arxiv.org/abs/1912.08581)|Arxiv|2019.12|[PyTorch](https://github.com/havakv/pycox)|PyCox 3-3|
|Graphical-Cal|[Graphical calibration curves and the integrated calibration index (ICI) for survival models](https://onlinelibrary.wiley.com/doi/epdf/10.1002/sim.8570)|Statistics in Medicine|2019.11|[Python](https://lifelines.readthedocs.io/en/latest/lifelines.calibration.html)||
|KSD|[Kernelized Stein Discrepancy Tests of Goodness-of-fit for Time-to-Event Data](https://arxiv.org/abs/2008.08397)|ICML|2020.08|||

***

## Temporal Time Process

|Title|Publisher|Date|Code|Notes|
|----|---|--|---|---|
|[Lecture Notes: Temporal Point Processes and the Conditional Intensity Function](https://arxiv.org/abs/1806.00221)|Arxiv|2018.06|||

***

## Applied Survival Analysis
|Title|Publisher|Date|Code|Notes|
|----|---|--|---|---|
|[Machine-Learning Approaches in COVID-19 Survival Analysis and Discharge-Time Likelihood Prediction Using Clinical Data](https://www.sciencedirect.com/science/article/pii/S2666389920300945)|Patterns|2020.08|[Python](https://github.com/Mnemati/Machine-Learning-Approaches-in-COVID-19-Survival-Analysis)||