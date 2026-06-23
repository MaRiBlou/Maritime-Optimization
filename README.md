# Sustainable Maritime Operations and Just-In-Time Arrival Management
**A Multi-criteria Regression Approach:** Predicting cargo vessel fuel consumption and CO2 emissions, optimizing Just-In-Time (JIT) maritime logistics using AIS telemetry, Copernicus environmental data, and Deep Learning / Gradient Boosting architectures.

## Project Overview
This project focuses on reducing the environmental footprint of the maritime industry (Decarbonization) through the Just-In-Time (JIT) Arrival strategy for Container/Cargo Ships. By utilizing raw AIS telemetry and meteorological data, a framework was developed to dynamically predict fuel consumption and CO2 emissions during voyages. The goal is to optimize cruising speed based on weather resistance, replacing the wasteful "hurry-up-and-wait" behavior at anchorages with a more efficient approach speed.


## Installation & Execution

To run the code locally or reproduce the results, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MaRiBlou/Maritime-Optimization.git
   cd Maritime-Optimization

Install the required dependencies:
It is recommended to use a virtual environment. Install the packages listed in the requirements.txt file:

Bash
pip install -r requirements.txt
Execute the code:
The source code is provided as a comprehensively commented Jupyter Notebook. Launch Jupyter and open the main notebook:

Bash
jupyter notebook notebooks/JIT_Arrival_and_Fuel_Consumption_for_Ships.ipynb

## Data Engineering Pipeline

### Data Acquisition & Preprocessing (Danish Open Marine AIS)
Handling raw AIS (Automatic Identification System) data presents a significant computational challenge, with daily volumes reaching ~3GB (millions of records). To prevent Out-Of-Memory (OOM) errors, an automated Chunk Loader & Cleaner was developed. The algorithm reads data sequentially in chunks of 100,000 rows, applying the following preprocessing pipeline on the fly:
* **Targeted Filtering:** Isolates specific vessel types (Cargo and Container ships), rejecting irrelevant maritime traffic.
* **Data Imputation & Cleansing:** Removes records with missing values (NaN) in critical variables (MMSI, Coordinates, SOG, Heading).
* **Temporal Downsampling:** Reduces the frequency to 1-hour resolution per vessel. This effectively removes high-frequency AIS noise, generating smooth, stable time-series data ideal for ML training.

Upon completing daily processing, the cleaned and highly compressed dataset is stored, and raw files are automatically deleted to optimize storage.

### Meteorological Integration (Copernicus Marine)
To accurately assess weather-induced resistance, environmental variables are dynamically fetched from Copernicus based on the vessel's daily coordinates. A bounding box with a 0.5-degree spatial buffer is created around the vessel's route to ensure accurate interpolation. Key variables include:
* **VHM0 (Significant Wave Height):** The most critical parameter. Head waves cause severe speed loss, forcing the engine to consume more fuel to maintain trajectory.
* **VMDR (Wave Mean Direction):** Determines if the wave is trailing (aiding propulsion) or heading (increasing resistance).
* **Wind Speed & Direction (UGRD, VGRD):** Used to calculate aerodynamic loads, heavily impacting Container ships due to their high freeboards.
* **Ocean Currents (UO, VO):** Vector components that either assist or resist the actual Speed Over Ground (SOG).

### Spatio-Temporal Data Fusion
A critical technical milestone of this project is the spatio-temporal fusion of highly heterogeneous data sources. The pipeline does not merely append data; it dynamically merges the continuous vector trajectories of the vessels (AIS) with the multi-dimensional meteorological grids (Copernicus). By applying spatial and temporal interpolation (matching exact timestamps, latitudes, and longitudes within the defined bounding boxes), the environmental constraints (wind, waves, currents) are mapped precisely onto each point of the vessel's route. This fusion creates a robust, multi-modal dataset that perfectly aligns vessel kinematics with environmental dynamics.

### Physics-Informed Target Generation (THETIS-MRV)
To avoid purely theoretical simulations, multi-year (2018-2023) data from the EMSA THETIS-MRV system were utilized. By applying strict vessel filters (Container Ships only) and dropping anomalous records, a realistic baseline for hourly fuel consumption (Tons/Hour) was established. This baseline acts as the Ground Truth. The ML algorithms do not predict arbitrary values; instead, they act as a Virtual Sensor. They learn to dynamically adjust this average baseline based on real-time SOG and local meteorological resistance, mirroring true hydrodynamic laws.

### Outlier Filtering & Dataset Splitting
During initial training on the full 1.4 million records, the Hist-Gradient Boosting model achieved an incredibly low MAE (0.1297 — off by merely ~130 kg of fuel per hour). However, the R-Squared metric artificially collapsed. Due to the cubic relationship inherent in the Physics-Informed model, even minor geographical AIS glitches produced astronomically high fuel consumption targets. By applying strict outlier filtering (SOG <= 25 knots, Fuel <= 40 tons/h), this noise was isolated, resulting in a clean dataset of 844,485 records. The data was then chronologically split into Train (70%), Validation (20%), and Test (10%) sets to prevent data leakage and evaluate the true predictive power of the algorithms.

* **Initial Size:** 1,428,742 rows
* **Cleaned Size:** 844,485 rows
* **Train Set (70%):** 591,139 rows
* **Validation Set (20%):** 168,905 rows
* **Test Set (10%):** 84,441 rows

## Model Evaluation & Physics Interpretation

**Insight:** As this is a Regression problem predicting a continuous variable (Fuel Tons/Hour), performance is evaluated using RMSE, MAE, and R-Squared, accompanied by Actual vs Predicted scatter plots and Loss curves, rather than Classification metrics like Confusion Matrices. 

### Performance Summary Table on the Test Set
<img width="528" height="237" alt="image" src="https://github.com/user-attachments/assets/0d5f580a-3cdc-4ebc-a11e-8254dea90f26" />

### Evaluation Visualizations

**1. Actual vs Predicted Fuel Consumption (Scatter Plots)**
<img width="1790" height="1025" alt="image" src="https://github.com/user-attachments/assets/3ac2c387-c786-4391-9c71-f1646b1f6199" />

**2. Metric Comparison (R² and RMSE)**
<img width="1536" height="536" alt="image" src="https://github.com/user-attachments/assets/8103ee21-5d0e-4681-9cc6-1dfc3cd1ccbf" />

**3. Deep Learning Convergence (LSTM Loss & MAE Curves)**
<img width="1010" height="470" alt="image" src="https://github.com/user-attachments/assets/d73338a2-8a3e-449f-949c-965e301bbfdd" />

### Key Insights
* **The Dominance of Trees and Deep Learning:** Tree-based models (LightGBM, Hist-GB) and Neural Networks successfully managed to "decode" the non-linear hydrodynamic law. LSTM achieved the top overall performance, exploiting its ability to perceive the sequential logic of voyages, though at the cost of increased training time.
* **Linearity vs. Non-Linearity:** The Voting-BRL model (Bayesian Ridge + Lasso), while being an exceptional State-of-the-Art model for annual/voyage-level data, struggled with this hourly, high-frequency dataset (R² = 0.81). Its linear nature fails to fully capture the cubic and quadratic relationships of the physical equations at very low or high speeds.

**Conclusion:** The fact that the top models stopped just short of a perfect R² of 1.0 confirms the success of the methodology: the models successfully learned the true physics of the ship's fuel consumption while effectively ignoring the artificial Gaussian noise (irreducible error) injected into the target variable.
