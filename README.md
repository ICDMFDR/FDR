
# Foundation Model-based Therapy plan Recommendation (FDR)
A Python and R implementation of Foundation Model-based Therapy plan Recommendation (FDR) method.


# Infrastructure used to run experiments:
* OS: Kali GNU/Linux, version 2024.2. (Linux 5.15.167.4-microsoft-standard-WSL2 x86_64)
* CPU: Intel(R) Core(TM) i7-1255U @ 1.70GHz).
* RAM: 16 GB.

# Datasets
We utilized observational data from the TransNEO neoadjuvant breast cancer clinical trial with 147 samples (Sammut et al., 2022) and
the ARTemis clinical trial with 75 samples (Earl et al., 2015), and 2 public clinical datasets for breast cancer research from the GEO (Gene Expression Omnibus) data repository hosted by NCBI (National Center for Biotechnology Information) including GSE25066 (Hatzis et al., 2011) and GSE41998 (Horak et al., 2013)

* TransNEO Multi-omics datasets: The dataset include 147 samples with multi-omics features, binary, categorical and continuous RCB outcome.
* ARTemis Multi-omics dataset: There are 75 samples in this dataset, however we use 72 samples which has the RCB outcomes.
* Combined Multi-omics dataset: We merged data from 2 datasets TransNEO and ARTemis. There are 219 samples which has multi-omics features, treatment arms and the RCB outcomes.
* GSE25066: We used 116 samples that have full treatment arms and treatment outcomes from this dataset. Each sample has 14 clinical features, 2 treatment arms, binary outcome (pCR) and categorical outcome (RCB.category)
* GSE41998: We used 265 samples from this dataset. Each sample has 9 clinical features, 2 treatment arms, binary outcome (pCR) and categorical outcome (RCB.category)
* TransNEO Clinical datasets: We reuse the data from the TransNEO with clinical features only, there are 147 samples which has 8 clinical features, treatment arms and the RCB outcomes.
* ARTemis Clinical dataset: ARTemis dataset with only clinical features, there are 72 samples which has 8 clinical features, treatment arms and the RCB outcomes.

# Baselines
* CT (CausalTree)
• CB (CatBoost)
• NN (Neural Network) MLPRegressor
• LR (Linear Regression
• RF (Random Forest)
• SVR (Support Vector Regression)
• XGB (XGBoost)


# Installation
**Installation requirements for CTR:**

* Python = 3.10
* numpy 1.23.5
* pandas 1.5.3
* tabpfn 2.0.9
* catboost 1.2.7
* xgboost 2.1.4
* scikit-learn 1.5.2
* scipy 1.15.3
* matplotlib 3.7.1
* seaborn 0.12.2
* R-base = 4.4.1
  * causalTree
  * rpart
 
**Detailed Guidelines for Environment Setup using Conda on Linux**

***1. Create a Conda Environment***

Create a new Conda environment named fdr_env with specific versions of Python and R: (bash)
     
    conda create -n fdr_env python=3.10 r-base=4.4 -c conda-forge -y
    conda activate fdr_env
***2. Install Python Packages:***
     Install essential Python packages using pip: (bash)
     
    pip install numpy pandas scikit-learn scipy matplotlib seaborn tabpfn catboost xgboost
***3. Install R packages:***
   Install the r-devtools package via Conda, and required R packages
     
    conda install r-devtools
    
    devtools::install_github("susanathey/causalTree")

    install.packages(c("dplyr", "graph", "rpart"), repos = "https://cloud.r-project.org", dependencies = TRUE)
    




# Reproducing the Paper Results:


**1. Run 7 models for recommendations with 5-fold cross-validation and OOD for all datasets.**

    python REC_5foldsCV.py
    python REC_OOD.py

**2. Run the CausalTree model using R scripts**

    Rscript CT_REC_5foldsCV.R
    Rscript CT_REC_OOD.R
**3. Generate the evaluation results in the paper**

    python model_evaluation.py
