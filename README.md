# Decision-Making Computational Neuroscience Portfolio

## Overview

This repository demonstrates computational neuroscience skills by combining real behavioural data analysis, mechanistic computational modelling, and machine learning applied to human decision-making.

**Repository theme:** Understanding how humans make decisions under conflict using reaction time data, drift diffusion models, and predictive modelling.

## Motivation

This project bridges three core areas of computational neuroscience and cognitive AI:

1. **Neuroscience & Cognition:** Analysing real human behavioural data from conflict tasks (flanker paradigm)
2. **Computational Modelling:** Implementing a drift diffusion model (DDM) to mechanistically explain decision dynamics
3. **AI & Machine Learning:** Using classification algorithms to decode task conditions and participant strategies from behavioural features

The repository showcases:
- Working with real public neuroscience datasets
- Mathematical implementation of cognitive models
- Application of ML to neural/behavioural data
- Clean, reproducible scientific Python code

## Dataset

The project uses the **Hedge et al. (2018)** flanker task dataset from the "The Reliability Paradox" study, available on OSF. This dataset contains:

- Reaction time (RT) and accuracy data from ~200 participants
- Flanker task with congruent and incongruent trials
- Multiple sessions per participant
- Real behavioural data suitable for computational analysis

**Why this dataset?**
- Small enough to run locally (~1-2 MB)
- Programmatically downloadable
- High-quality, published data
- Ideal for DDM fitting and ML decoding

## Repository Structure

```
comp-neuro/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Downloaded raw data
│   ├── processed/        # Cleaned data
│   └── README.md
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_ddm_model.ipynb
│   └── 03_decoding.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Download and load data
│   ├── preprocessing.py  # Clean and filter data
│   ├── features.py       # Extract RT features
│   ├── ddm_model.py      # Drift diffusion model
│   └── decoding.py       # ML classification
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py
└── results/
    ├── figures/
    └── models/
```

## Components

### 1. Real Data Analysis (`notebooks/01_data_analysis.ipynb`)

**What it does:**
- Downloads the Hedge et al. (2018) flanker dataset from OSF
- Preprocesses RT data (removes outliers, filters errors)
- Visualises RT distributions by condition
- Computes summary statistics (mean RT, accuracy, congruency effects)
- Explores individual differences across participants

**Skills demonstrated:**
- Handling real public neuroscience data
- Data cleaning and quality control
- Statistical visualization
- Understanding cognitive task paradigms

### 2. Drift Diffusion Model (`notebooks/02_ddm_model.ipynb`)

**What it does:**
- Implements a 4-parameter DDM from scratch
- Fits model parameters to congruent vs incongruent trials
- Performs parameter recovery and validation
- Visualises decision boundaries and evidence accumulation
- Relates model parameters to behavioural effects

**Mathematical implementation:**
- Drift rate (v): quality of evidence
- Boundary separation (a): speed-accuracy tradeoff
- Non-decision time (t0): sensory/motor delays
- Starting point bias (z): response bias

**Skills demonstrated:**
- Mathematical modelling of cognitive processes
- Parameter estimation and optimization
- Model validation
- Linking models to neural mechanisms

### 3. Machine Learning Decoding (`notebooks/03_decoding.ipynb`)

**What it does:**
- Extracts RT-based features (mean, std, skewness, percentiles)
- Classifies congruent vs incongruent trials from RT patterns
- Predicts participant identity from behavioural signatures
- Performs dimensionality reduction (PCA) for visualization
- Evaluates model performance with cross-validation

**Models used:**
- Logistic Regression (baseline)
- Random Forest (nonlinear classifier)
- SVM with RBF kernel

**Skills demonstrated:**
- Feature engineering from behavioural data
- Proper train/test splits and cross-validation
- Comparison of ML algorithms
- Interpretation of model performance
- Avoiding overfitting and overclaiming

## How to Run

### 1. Clone and setup

```bash
git clone https://github.com/indigojnewark/comp-neuro.git
cd comp-neuro
pip install -r requirements.txt
```

### 2. Download data

```bash
python scripts/download_data.py
```

This downloads the Hedge et al. (2018) flanker dataset from OSF (~1 MB).

### 3. Run notebooks

Open Jupyter and run notebooks in order:

```bash
jupyter notebook
```

1. `01_data_analysis.ipynb` - Explore and visualise the data
2. `02_ddm_model.ipynb` - Fit the drift diffusion model
3. `03_decoding.ipynb` - Train ML classifiers

OR run the full pipeline:

```bash
python scripts/run_pipeline.py
```

## Key Results

### Data Analysis
- Congruent trials: ~450 ms mean RT
- Incongruent trials: ~550 ms mean RT (~100 ms congruency effect)
- Accuracy: >95% congruent, ~85% incongruent
- RT distributions show characteristic right skew

### DDM Fits
- Lower drift rate (v) for incongruent trials (slower evidence accumulation)
- Boundary separation (a) similar across conditions
- Model captures RT distributions and accuracy
- Parameter estimates consistent with published literature

### Decoding Performance
- Condition classification (congruent vs incongruent): ~75% accuracy
- Participant identification: ~45% accuracy (chance = 0.5%)
- Random Forest outperforms linear models
- RT variability and skewness are most predictive features

## Technical Details

**Libraries:**
- `numpy`, `scipy` - Numerical computing and statistics
- `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning
- `requests` - Data downloading from OSF

**Computational requirements:**
- Runtime: ~5-10 minutes total
- Memory: <1 GB
- No GPU required

## What This Repository Demonstrates

✅ **Real data analysis:** Working with published neuroscience datasets, not toy data

✅ **Mathematical modelling:** Implementing cognitive models from equations, not black boxes

✅ **Machine learning:** Proper ML workflow with validation and realistic performance

✅ **Clean code:** Modular structure, docstrings, reproducibility

✅ **Scientific thinking:** Connecting behaviour, models, and neural mechanisms

✅ **MSc readiness:** Demonstrates programming, modelling, and data analysis skills needed for computational neuroscience graduate work

## References

- Hedge, C., Powell, G., & Sumner, P. (2018). The reliability paradox: Why robust cognitive tasks do not produce reliable individual differences. *Behavior Research Methods*, 50(3), 1166-1186.
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, 20(4), 873-922.
- Forstmann, B. U., Ratcliff, R., & Wagenmakers, E. J. (2016). Sequential sampling models in cognitive neuroscience: Advantages, applications, and extensions. *Annual Review of Psychology*, 67, 641-666.

## Author

Built by a neuroscience BSc graduate transitioning into computational neuroscience and cognitive AI for MSc applications.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Dataset from Hedge et al. (2018), available on OSF: https://osf.io/cwzds/
