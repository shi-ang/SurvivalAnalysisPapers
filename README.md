# Survival Analysis Paper List

A list of papers/resources in Survival Analysis that I have read or would like to read. Should you wish to suggest an addition to this list, please feel free to open an issue.

Last Update Time: 2024.04.06

- Categories
    - [Tutorials/Surveys](#tutorials-and-surveys)
	- [ML/DL Survival Models](#ml-and-dl-for-survival-analysis)
    - [Objective Functions](#objective-functions)
    - [Time-varying Covariates Models](#time-varying-covariates-models)
    - [Explainable Survival Models](#explainable-survival-models)
    - [Competing Risks Models](#competing-risks-models)
    - [Generalized Survival Analysis Methods](#generalized-survival-analysis-methods)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Causal Inference](#causal-inference)
    - [Fairness](#fairness)
    - [Dependent Censoring](#dependent-censoring)
    - [Temporal Time Process](#temporal-time-process)
    - [Applications](#applied-survival-analysis)

*Please note that some papers may belong to multiple categories. However, I've organized them according to their most significant contribution (purely subjective).
***

## Tutorials and Surveys

|Title|Publisher|Date|Code|Notes|
|----|---|--|---|---|
|[Machine Learning for Survival Analysis: A Survey](https://arxiv.org/abs/1708.04649)|ACM Computing Surveys|2019.02||[Slides](https://dmkd.cs.vt.edu/TUTORIAL/Survival/)|
|[Calibration: the Achilles heel of predictive analytics](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7)|BMC Medicine|2019.12|||
|[A tutorial on calibration measurements and calibration models for clinical prediction models](https://academic.oup.com/jamia/article/27/4/621/5762806)|JAMIA|2020.02|[R](https://github.com/easonfg/cali_tutorial)||
|[Survival analysis—time-to-event data and censoring](https://www.nature.com/articles/s41592-022-01563-7)|Nature Methods|2022.04|||
|[Survey: Strategies for Loss-Based Discrete-Time Hazard and Survival Function Estimation](https://ieeexplore.ieee.org/document/9952504)|ICTC|2022.10|||
|[Regression modeling of time-to-event data with censoring](https://www.nature.com/articles/s41592-022-01689-8)|Nature Methods|2022.11|||
|[Factors influencing clinician and patient interaction with machine learning-based risk prediction models: a systematic review](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00241-8/fulltext)|Lancet Digital Health|2024.02|||
***

## ML and DL for Survival Analysis

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|GBMCOX|[The State of Boosting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.22.276&rep=rep1&type=pdf)|Computing Science and Statistics|2008|[R](https://github.com/gbm-developers/gbm)||
|RSF|[Random Survival Forest](https://arxiv.org/pdf/0811.1645.pdf)|The Annals of Applied Statistics|2008|[R](https://kogalur.github.io/randomForestSRC/)||
|MTLR|[Learning Patient-Specific Cancer Survival Distributions as a Sequence of Dependent Regressors](http://www.cs.cornell.edu/~cnyu/papers/nips11_survival.pdf)|NeurIPS|2011|[R](https://cran.r-project.org/web/packages/MTLR/index.html)|[Poster](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.8967&rep=rep1&type=pdf)|
|N-MTLR|[Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework](https://arxiv.org/abs/1801.05512)|Arxiv|2018.01|[Python](https://square.github.io/pysurvival/)||
|DeepSurv|[DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1)|BMC Medical Research Methodology|2018.02|[Python](https://github.com/jaredleekatzman/DeepSurv)||
|DATE & DRAFT|[Adversarial Time-to-Event Modeling](https://arxiv.org/abs/1804.03184)|ICML|2018.07|[TensorFlow](https://github.com/paidamoyo/adversarial_time_to_event)||
|CoxTime / CoxCC|[Time-to-Event Prediction with Neural Networks and Cox Regression](https://jmlr.org/papers/volume20/18-424/18-424.pdf)|JMLR|2019.08|[PyTorch](https://github.com/havakv/pycox)|PyCox 1-3|
|PCHazard / LogisticHazard|[Continuous and Discrete-Time Survival Prediction with Neural Networks](https://arxiv.org/abs/1910.06724)|Arxiv|2019.10|[PyTorch](https://github.com/havakv/pycox)|PyCox 2-3|
|SurvivalQuilts|[Temporal Quilting for Survival Analysis](http://proceedings.mlr.press/v89/lee19a/lee19a.pdf)|AISTATS|2020.04|[Python](https://github.com/chl8856/SurvivalQuilts)||
|SCA|[Survival Cluster Analysis](https://dl.acm.org/doi/pdf/10.1145/3368555.3384465)|ACM CHIL|2020.04|[TensorFlow](https://github.com/paidamoyo/survival_cluster_analysis)||
|VAECox|[Improved survival analysis by learning shared genomic information from pan-cancer data](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i389/5870509)|Bioinformatics|2020.07|[Pytorch](https://github.com/dmis-lab/VAECox)||
|DCM|[Deep Cox Mixtures for Survival Regression](https://arxiv.org/pdf/2101.06536.pdf)|NeurIPS Machine Learning for Health Workshop|2021.01|[TensorFlow](https://github.com/chiragnagpal/deep_cox_mixtures)||
|DHBN|[Using Discrete Hazard Bayesian Networks to Identify which Features are Relevant at each Time in a Survival Prediction Model](http://proceedings.mlr.press/v146/kuan21a/kuan21a.pdf)|AAAI Spring Symposium (SP-ACA)|2021.03|[R](https://github.com/kuan0911/ISDEvaluation)||
|TDSA|[Transformer-Based Deep Survival Analysis](http://proceedings.mlr.press/v146/hu21a/hu21a.pdf)|AAAI Spring Symposium (SP-ACA)|2021.03|||
|DeepQuantreg|[Deep learning for quantile regression under right censoring: DeepQuantreg](https://www.sciencedirect.com/science/article/abs/pii/S0167947321001572)|Computational Statistics and Data Analysis|2021.07|[TensorFlow](https://github.com/yicjia/DeepQuantreg)||
|IWSG|[Inverse-Weighted Survival Games](https://openreview.net/forum?id=j4oYd8SGop)|NeurIPS|2021.12|[PyTorch](https://github.com/rajesh-lab/Inverse-Weighted-Survival-Games)||
|DeepEH|[Deep Extended Hazard Models for Survival Analysis](https://openreview.net/forum?id=GUD7rNkaWKr)|NeurIPS|2021.12|||
|VaDeSC|[A Deep Variational Approach to Clustering Survival Data](https://openreview.net/forum?id=RQ428ZptQfU)|ICLR|2022.03|[TensorFlow](https://github.com/i6092467/vadesc)||
|ODE-Cox|[Survival Analysis via Ordinary Differential Equations](https://www.tandfonline.com/doi/abs/10.1080/01621459.2022.2051519)|JASA|2022|||
|Survival MDN|[Survival Mixture Density Networks](https://arxiv.org/pdf/2208.10759.pdf)|ML4HC|2022.05|[PyTorch](https://github.com/XintianHan/Survival-MDN)||
|SODEN|[SODEN: A Scalable Continuous-Time Survival Model through Ordinary Differential Equation Networks](https://arxiv.org/pdf/2008.08637.pdf)|JMLR|2022|[PyTorch](https://github.com/jiaqima/SODEN)||
|DCS|[Deep Learning-Based Discrete Calibrated Survival Prediction](https://arxiv.org/pdf/2208.08182.pdf)|ICDH|2022.08|[PyTorch](https://github.com/imsb-uke/dcsurv)||
|CQRNN|[Censored Quantile Regression Neural Networks for Distribution-Free Survival Analysis](https://arxiv.org/abs/2205.13496)|NeurIPS|2022.11|[PyTorth](https://github.com/TeaPearce/Censored_Quantile_Regression_NN)|[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/55198.png?t=1669392940.4432118)|
|MSSDA|[Multi-Source Survival Domain Adaptation](https://arxiv.org/pdf/2212.00424.pdf)|AAAI|2023|||
|SurvivalGAN|[SurvivalGAN: Generating Time-to-Event Data for Survival Analysis](https://proceedings.mlr.press/v206/norcliffe23a/norcliffe23a.pdf)|AIStats|2023.02|[PyTorch](https://github.com/vanderschaarlab/survivalgan)||
|DH-MNN|[Metaparametric Neural Networks for Survival Analysis](https://ieeexplore.ieee.org/document/9585306)|TNNLS|2023.08|||
|NSOTree|[Neural Survival Oblique Tree](https://arxiv.org/pdf/2309.13825.pdf)|Arxiv|2023.09|[Code](https://github.com/xs018/NSOTree)||
|NNCDE|[Conditional Distribution Function Estimation Using Neural Networks for Censored and Uncensored Data](https://www.jmlr.org/papers/volume24/22-0657/22-0657.pdf)|JMLR|2023.12|[PyTorch](https://github.com/bingqing0729/NNCDE)||
|NFM|[Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions](https://openreview.net/forum?id=3Fc9gnR0fa)|NeurIPS|2023.12|[PyTorch](https://github.com/Rorschach1989/nfm)||
|Diffsurv|[Differentiable sorting for censored time-to-event data](https://openreview.net/forum?id=gYWjI7wLhc)|NeurIPS|2023.12|[PyTorch](https://github.com/andre-vauvelle/diffsurv)||
|OSST|[Optimal Sparse Survival Trees](https://arxiv.org/pdf/2401.15330.pdf)|AIStats|2024.01|[Python](https://github.com/ruizhang1996/optimal-sparse-survival-trees-public/)||
***
## Objective Functions
|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|GBMCI|[A Gradient Boosting Algorithm for Survival Analysis via Direct Optimazation of Concordance Index](https://www.hindawi.com/journals/cmmm/2013/873595/)|Computational and Mathematical Methods in Medicine|2013.09|[R](https://github.com/uci-cbcl/GBMCI)||
|Survival-CRPS|[Countdown Regression: Sharp and Calibrated Survival Predictions](https://arxiv.org/abs/1806.08324)|UAI|2019|[PyTorch](https://github.com/stanfordmlgroup/cdr-mimic)||
||[Bias in Cross-Entropy-Based Training of Deep Survival Networks](https://ieeexplore.ieee.org/document/9028113)|TPAMI|2020.03|||
|SFM|[Calibration and Uncertainty in Neural Time-to-Event Modeling](https://ieeexplore.ieee.org/document/9244076)|TNNLS|2020.09|[TensorFlow](https://github.com/paidamoyo/calibration_uncertainty_t2e)||
|X-CAL|[X-CAL: Explicit Calibration for Survival Analysis](https://arxiv.org/abs/2101.05346)|NeurIPS|2020|[PyTorch](https://github.com/rajesh-lab/X-CAL)|[Poster](https://nips.cc/virtual/2020/public/poster_d4a93297083a23cc099f7bd6a8621131.html)|
|Discrete-RPS|[Estimating Calibrated Individualized Survival Curves with Deep Learning](https://ojs.aaai.org/index.php/AAAI/article/view/16098)|AAAI|2021.02|[PyTorch](https://github.com/MLD3/Calibrated-Survival-Analysis)||
|KL-Calibration|[Simpler Calibration for Survival Analysis](https://openreview.net/forum?id=bB6YLDJewoK)|ICLR OpenReview|2021.10|||
|SuMo-net|[Survival regression with proper scoring rules and monotonic neural networks](https://proceedings.mlr.press/v151/rindt22a.html)|AIStats|2022.03|[PyTorch](https://github.com/MrHuff/Sumo-Net)||
|DQS|[Proper Scoring Rules for Survival Analysis](https://arxiv.org/pdf/2305.00621.pdf)|ICML|2023.06|[PyTorch](https://github.com/IBM/dqs)|[Poster](https://icml.cc/media/PosterPDFs/ICML%202023/24261.png?t=1687257098.1469564)|
***

## Time-varying Covariates Models

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|SPH|[Survival Prediction by an Integrated Learning Criterion on Intermittently Varying Healthcare Data](https://ojs.aaai.org/index.php/AAAI/article/view/9999)|AAAI|2016.02|||
|Time-varying Cox|[Time-varying covariates and coefficients in Cox regression models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6015946/)|Annals of Translational Medicine|2018.01|||
|DRSA|[Deep Recurrent Survival Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/4407)|AAAI|2019.02|[TensorFlow](https://github.com/rk2900/DRSA)||
|TSNN|[Time-Dependent Survival Neural Network for Remaining Useful Life Prediction](https://link.springer.com/chapter/10.1007/978-3-030-16148-4_34)|PAKDD|2019.03|||
|Dynamic-DeepHit|[Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data](https://ieeexplore.ieee.org/document/8681104)|TBME|2019.04|[PyTorch](https://github.com/chl8856/Dynamic-DeepHit)||
|TSNN & RSNN|[Survival neural networks for time-to-event prediction in longitudinal study](https://link.springer.com/article/10.1007%2Fs10115-020-01472-1)|Knowledge and Information System|2020.05||Extend Paper|
|DRSM|[Deep Parametric Time-to-Event Regression with Time-Varying Covariates](http://proceedings.mlr.press/v146/nagpal21a.html)|AAAI Spring Symposium (SP-ACA)|2021.03|[PyTorch](https://autonlab.github.io/DeepSurvivalMachines/#deep-recurrent-survival-machines)||
|TCSA|[Temporally-Consistent Survival Analysis](https://proceedings.neurips.cc/paper_files/paper/2022/hash/455e1e30edf721bd7fa334fffabdcad8-Abstract-Conference.html)|NeurIPS|2022.11|[Python](https://github.com/spotify-research/tdsurv)|[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/118bd558033a1016fcc82560c65cca5f.png?t=1667814990.0636194)|
|SurvPP|[Survival Permanental Processes for Survival Analysis with Time-Varying Covariates](https://openreview.net/forum?id=CYCzfXn6cZ)|NeurIPS|2023.12|[Python](https://github.com/HidKim/SurvPP)||
***

## Explainable Survival Models

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|SPIE|[Simultaneous Prediction Intervals for Patient-Specific Survival Curves](https://www.ijcai.org/Proceedings/2019/0828.pdf)|IJCAI|2019|[Python](https://github.com/ssokota/spie)||
|SurvLIME|[SurvLIME: A method for explaining machine learning survival models](https://www.sciencedirect.com/science/article/abs/pii/S0950705120304044)|Knowledge-Based Systems|2020.09|[Python](https://github.com/imatge-upc/SurvLIMEpy)||
|AutoScore-Survival|[AutoScore-Survival: Developing interpretable machine learning-based time-to-event scores with right-censored survival data](https://linkinghub.elsevier.com/retrieve/pii/S1532-0464(21)00288-4)|Journal of Biomedical Informatics|2022.01|[R](https://github.com/nliulab/AutoScore-Survival)||
|EXCEL|[Explainable Censored Learning: Finding Critical Features with Long Term Prognostic Values for Survival Prediction](https://arxiv.org/abs/2209.15450)|Arxiv|2022.09|||
|BNN-ISD|[Using Bayesian Neural Networks to Select Features and Compute Credible Intervals for Personalized Survival Prediction](https://ieeexplore.ieee.org/document/10158019)|TBME|2023.07|[PyTorch](https://github.com/shi-ang/BNN-ISD)||
***

## Competing Risks Models

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
||[On pseudo-values for regression analysis in competing risks models](https://pubmed.ncbi.nlm.nih.gov/19051013/)|Lifetime Data Analysis|2009.06|||
|DeepHit|[DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit.pdf)|AAAI|2018.02|||
|SurvTRACE|[SurvTRACE: Transformers for Survival Analysis with Competing Events](https://arxiv.org/abs/2110.00855)|Arxiv|2021.10|[Pytorch](https://github.com/RyanWangZf/SurvTRACE)||
|Deep-CSA|[Deep-CSA: Deep Contrastive Learning for Dynamic Survival Analysis with Competing Risks](https://ieeexplore.ieee.org/abstract/document/9756287)|IEEE Journal of Biomedical and Health Informatics|2022.04|||
|DeepPseudo|[DeepPseudo: Pseudo Value Based Deep Learning Models for Competing Risk Analysis](https://arxiv.org/pdf/2207.05247.pdf)|KDD DSHealth Workshop|2022.08|||
***

## Generalized Survival Analysis Methods

||Title|Publisher|Date|Code|Notes|
|--|----|---|--|---|---|
|Pseudo-observations|[Pseudo-observations in survival analysis](https://pubmed.ncbi.nlm.nih.gov/19654170/)|Statistical Methods in Medical Research|2010|||
||[A doubly robust censoring unbiased transformation](https://pubmed.ncbi.nlm.nih.gov/22550646/)|The International Journal of Biostatistics|2007.03|||
||[Adapting machine learning techniques to censored time-to-event health record data: A general-purpose approach using inverse probability of censoring weighting](https://www.sciencedirect.com/science/article/pii/S1532046416000496)|Journal of Biomedical Informatics|2016.03|[R](https://github.com/docvock/JBI_IPCW_for_ML)||
||[A General Machine Learning Framework for Survival Analysis](https://arxiv.org/abs/2006.15442)|ECML|2020.06|[R](https://github.com/adibender/machine-learning-for-survival-ecml2020)||
|CSA|[Conformalized survival analysis](https://academic.oup.com/jrsssb/article/85/1/24/7008653)|JRSS: Series B|2023.01|[R](https://github.com/zhimeir/cfsurvival)||
***

## Evaluation Metrics

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|---|--|---|---|
|Brier Score Decomposition|[The Comparison and Evaluation of Forecasters](https://www.jstor.org/stable/2987588)|The Statistician|1983|||
|CRPS Decomposition|[Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems](https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml?tab_body=abstract-display)|Weather and Forecasting|2000.03|||
|IPCW Brier Score|[Assessment and Comparison of Prognostic Classification Schemes for Survival Data](https://pubmed.ncbi.nlm.nih.gov/10474158/)|Statistics in Medicine|1999.09|||
|Administrative Brier Score|[The Brier Score under Administrative Censoring: Problems and Solutions](https://arxiv.org/abs/1912.08581)|JMLR|2019.12|[PyTorch](https://github.com/havakv/pycox)|PyCox 3-3|
|Graphical-Cal|[Graphical calibration curves and the integrated calibration index (ICI) for survival models](https://onlinelibrary.wiley.com/doi/epdf/10.1002/sim.8570)|Statistics in Medicine|2019.11|[Python](https://lifelines.readthedocs.io/en/latest/lifelines.calibration.html)||
|D-Calibration|[Effective Ways to Build and Evaluate Individual Survival Distributions](https://jmlr.org/papers/volume21/18-772/18-772.pdf)|JMLR|2020.06|[R](https://github.com/haiderstats/ISDEvaluation)||
|KSD|[Kernelized Stein Discrepancy Tests of Goodness-of-fit for Time-to-Event Data](https://arxiv.org/abs/2008.08397)|ICML|2020.08|||
||[Scoring rules in survival analysis](https://arxiv.org/abs/2212.05260)|Arxiv|2022.12|||
|MAE-PO|[An Effective Meaningful Way to Evaluate Survival Models](https://arxiv.org/pdf/2306.01196.pdf)|ICML|2023.06|[PyTorch](https://github.com/shi-ang/CensoredMAE)|[Poster](https://icml.cc/media/PosterPDFs/ICML%202023/24533.png?t=1688075972.3632987)|
***

## Causal Inference

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|---|--|---|---|
||[Causal inference in survival analysis using pseudo-observations](https://onlinelibrary.wiley.com/doi/10.1002/sim.7297)|Statistics in Medicine|2017.03|||
|CausalTree|[Causal Inference for Survival Analysis](https://arxiv.org/pdf/1803.08218.pdf)|Arvix|2018.03|[R](https://github.com/vikas84bf/causalTree)||
|CSA|[Enabling Counterfactual Survival Analysis with Balanced Representations](https://arxiv.org/abs/2006.07756)|ACM CHIL|2021.03|[Python](https://github.com/paidamoyo/counterfactual_survival_analysis)||
|SurvITE|[SurvITE: Learning Heterogeneous Treatment Effects from Time-to-Event Data](https://arxiv.org/pdf/2110.14001.pdf)|NeurIPS|2021.10|[TensorFlow](https://github.com/chl8856/survITE)||
|CMHE|[Counterfactual Phenotyping with Censored Time-to-Events](https://arxiv.org/abs/2202.11089)|KDD|2022.02|[PyTorch](https://autonlab.github.io/auton-survival/cmhe/)||
|DNMC|[Disentangling Whether from When in a Neural Mixture Cure Model for Failure Time Data](https://proceedings.mlr.press/v151/engelhard22a/engelhard22a.pdf)|AISTATS|2022.03|[TensorFlow](https://github.com/mengelhard/dnmc/)||
|compCATE|[Understanding the Impact of Competing Events on Heterogeneous Treatment Effect Estimation from Time-to-Event Data](https://arxiv.org/pdf/2302.12718v1.pdf)|AISTATS|2023.02|[Python](https://github.com/AliciaCurth/CompCATE)||
***

## Fairness

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|FSRF|[Longitudinal Fairness with Censorship](https://arxiv.org/abs/2203.16024)|AAAI|2022.03|||
|FISA|[Fair and Interpretable Models for Survival Analysis](https://dl.acm.org/doi/10.1145/3534678.3539259)|KDD|2022.08||[Video](https://dl.acm.org/doi/10.1145/3534678.3539259)|
|IFS|[Censored Fairness through Awareness](https://ojs.aaai.org/index.php/AAAI/article/view/26708)|AAAI|2023.03|||
||[Fairness-Aware Processing Techniques in Survival Analysis: Promoting Equitable Predictions](https://link.springer.com/chapter/10.1007/978-3-031-43427-3_28)|ECML-PKDD|2023.09|||
***

## Dependent Censoring

|Keyword|Title|Publisher|Date|Code|Notes|
|---|----|--|--|---|---|
|CopulaDeepSurvival|[Copula-Based Deep Survival Models for Dependent Censoring](https://proceedings.mlr.press/v216/gharari23a/gharari23a.pdf)|UAI|2023.06|[PyTorch](https://github.com/rgklab/copula_based_deep_survival)||
|DCSurvival|[Deep Copula-Based Survival Analysis for Dependent Censoring with Identifiability Guarantees](https://arxiv.org/pdf/2312.15566.pdf)|AAAI|2023.12|[PyTorch](https://github.com/WeijiaZhang24/DCSurvival)||
***

## Temporal Time Process

|Title|Publisher|Date|Code|Notes|
|----|---|--|---|---|
|[Lecture Notes: Temporal Point Processes and the Conditional Intensity Function](https://arxiv.org/abs/1806.00221)|Arxiv|2018.06|||
|[Temporal Point Processes](https://courses.mpi-sws.org/hcml-ws18/lectures/TPP.pdf)|Course Material|2019.01|||
|[Recent Advance in Temporal Point Process: from Machine Learning Perspective](https://thinklab.sjtu.edu.cn/src/pp_survey.pdf)||2019|||
|[Wavelet Reconstruction Networks for Marked Point Processes](https://proceedings.mlr.press/v146/weiss21a.html)|AAAI Spring Symposium (SP-ACA)|2021.03|[Python](https://github.com/jcweiss2/wrnppl/tree/master)||
|[Decoupled Marked Temporal Point Process using Neural Ordinary Differential Equations](https://openreview.net/forum?id=BuFNoKBiMs)|ICLR|2024.01|||
***

## Applied Survival Analysis

|Title|Publisher|Date|Code|Notes|
|----|---|--|---|---|
|[Empirical comparisons between Kaplan-Meier and Nelson-Aalen survival function estimators](https://www.tandfonline.com/doi/abs/10.1080/00949650212847)|Journal of Statistical Computation and Simulation|2002|||
|[Machine-Learning Approaches in COVID-19 Survival Analysis and Discharge-Time Likelihood Prediction Using Clinical Data](https://www.sciencedirect.com/science/article/pii/S2666389920300945)|Patterns|2020.08|[Python](https://github.com/Mnemati/Machine-Learning-Approaches-in-COVID-19-Survival-Analysis)||
|[Application of a novel machine learning framework for predicting non-metastatic prostate cancer-specific mortality in men using the Surveillance, Epidemiology, and End Results (SEER) database](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(20)30314-9/fulltext)|Lancet Digital Health|2021.03|[Python](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/)||
|[Learning accurate personalized survival models for predicting hospital discharge and mortality of COVID-19 patients](https://www.nature.com/articles/s41598-022-08601-6)|Scientific Report|2022.03|[R](https://github.com/kuan0911/ISDEvaluation-covid)||
|[Predicting Time-to-conversion for Dementia of Alzheimer's Type using Multi-modal Deep Survival Analysis](https://arxiv.org/pdf/2205.01188.pdf)|Arxiv|2022.05|||
|[Time-to-event modeling for hospital length of stay prediction for COVID-19 patients](https://www.sciencedirect.com/science/article/pii/S2666827022000603)|Machine Learning with Applications|2022.09|||
|[Metabolomic profiles predict individual multidisease outcomes](https://www.nature.com/articles/s41591-022-01980-3)|Nature Medicine|2022.09|[PyTorch](https://github.com/thbuerg/MetabolomicsCommonDiseases)||
|[Personalized breast cancer onset prediction from lifestyle and health history information](https://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0279174)|PLOS One|2022.12|||
|[SurProGenes: Survival Risk-Ordered Representation of Cancer Patients and Genes for the Identification of Prognostic Genes](https://openreview.net/forum?id=t4COq27gBs)|ICML|2023.06|[TensorFLow](https://github.com/JunetaeKim/SurProGenes)|[Poster](https://icml.cc/media/PosterPDFs/ICML%202023/25067.png?t=1686635672.8107529)|
|[Semi-Parametric Contextual Pricing Algorithm using Cox Proportional Hazards Model](https://openreview.net/forum?id=wkr4r2Cw3i)|ICML|2023.06|[Code](https://github.com/younggeunchoi/CoxContextualPricing)|[Poster](https://icml.cc/media/PosterPDFs/ICML%202023/24954.png?t=1688273092.4951394)|
|[Contrastive Learning of Temporal Distinctiveness for Survival Analysis in Electronic Health Records](https://dl.acm.org/doi/10.1145/3583780.3614824)|CIKM|2023.10|[Python](https://github.com/mohsen-nyb/OTCSurv)||
|[Prototypical Information Bottlenecking and Disentangling for Multimodal Cancer Survival Prediction](https://arxiv.org/pdf/2401.01646.pdf)|ICLR|2024.01|[Python](https://github.com/zylbuaa/PIBD)||
