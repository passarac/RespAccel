# RespAccel
Generate respiratory signals from tri-axal accelerometer data from a chest-worn sensor. Apply cleaning and filtering to identify valid and reliable segments of respiratory data.

---

## Respiratory Waveform Estimation

The overall goal is to turn raw tri-axal accelerometer stream into a clean, respiration-related flow-like waveform during periods when the person is static. The key idea is that, at rest, chest wall motion rotates the gravity vector in the device frame. If you can:

1) track the instantaneous axis of rotation of the chest and
2) measure the small oscillatory angle about that axis,

then the time derivative of that angle closely tracks airflow measured by a nasal cannula. 

This code is based from the method described in the paper by Bates et al. (2010) in *Respiratory Rate and Flow Waveform Estimation from Tri-axial Accelerometer Data*. The paper validated this by showing high correlation with cannula pressure data.

### Big-picture overview of the main steps

1. **Pre-filter and normalise**  
   - Apply low-pass filtering to the tri-axial accelerometer signal to remove high-frequency noise.  
   - Normalise each sample to unit length to focus on direction changes rather than magnitude.

2. **Detect static periods**  
   - Calculate the inter-sample rotation angle:
        θₜ = cos⁻¹(âₜ · âₜ₋₁)
   - Discard segments where $\theta_t$ exceeds the motion threshold, ensuring only breathing-related motion is analysed.

3. **Find and track the breathing rotation axis**  
   - Compute raw rotation axes:  
       rₜ = âₜ × âₜ₋₁ 
   - Align signs using a reference axis from PCA.  
   - Smooth using a Hamming window, optionally weighted by $\theta_t$, to obtain $\bar{r}_t$.

4. **Estimate the mean gravity vector**  
   - Compute a sliding-window mean of the acceleration to estimate $\bar{a}_t$, the local gravity direction.

5. **Calculate the breathing angle**  
   - Determine the signed angular displacement of the chest around $\bar{r}_t$:  
     φₜ = sin⁻¹[ (āₜ × r̄ₜ) · aₜ ]  
6. **Differentiate to obtain a flow-like signal**  
   - Numerically differentiate $\phi_t$ with respect to time:  
     ωₜ = dφₜ/dt
     ωₜ ≈ (φₜ − φₜ₋₁) / Δt 
   - Apply low-pass filtering to reduce noise.  
   - The result ωₜ closely resembles nasal cannula flow, enabling respiratory rate and waveform shape estimation.

---

### Movement handling & data conditioning

**`filter.py`**  
- **Low-pass filtering:** Smooth the raw tri-axial accelerometer data using a zero-phase Butterworth filter to suppress high-frequency noise.  
- **Large motion detection:** Calculate the inter-sample rotation angle  
   θₜ = cos⁻¹(âₜ · âₜ₋₁) 
  and apply an angle threshold to identify static periods.  
- **Purpose:** Restrict further analysis to periods where the accelerometer signal reflects breathing-related chest wall motion, rather than whole-body movements.  
- **Context (Bates et al., 2010):** This matches the paper’s movement detection step using a maximum rotation angle per sample to exclude high-motion segments, improving agreement between accelerometer-derived and nasal cannula waveforms.

---

### Axis of rotation estimation & tracking

**`computeRotationAxes.py`**  
- **Purpose:** Compute the instantaneous rotation axis between consecutive normalised acceleration vectors.  
- **Equation:**  
  rₜ = âₜ × âₜ₋₁ 
  This corresponds to Eq. (2) in Bates et al. (2010).  

**`findReferenceAxis.py`**  
- **Purpose:** Select a consistent global sign/direction for all rotation axes using PCA.  
- **Method:** Identify the principal component of the set of rₜ vectors and use it as the reference axis r_ref. Flip signs so that  
     rₜ′ = {
       rₜ,   if rₜ · r_ref ≥ 0
       −rₜ,  otherwise
      } 
  This corresponds to Eq. (3) in the paper.  

**`trackRotationAxis.py`**  
- **Purpose:** Produce a smoothed, sign-consistent rotation axis r̄ₜ over time.  
- **Method:**  
  1. Apply a sliding Hamming window to rₜ′ values.  
  2. Optionally weight each vector by the inter-sample rotation angle θₜ.  
  3. Average and normalise to unit length:  
     r̄ₜ = normalize( Σᵢ∈Wₜ  H(i) · θₜ₊ᵢ · r′ₜ₊ᵢ )
  This matches Eq. (4) in Bates et al. (2010).

---

### Breathing angle and flow-like signal

**`computeMeanAccel.py`**  
- **Purpose:** Estimate the mean gravity vector āₜ over a short time window.  
- **Method:** Compute a moving average of the normalised acceleration samples to reduce noise.  
- **Context:** This matches Eq. (5) in Bates et al. (2010).

**`computeBreathingAngle.py`**  
- **Purpose:** Calculate the instantaneous breathing angle φₜ around the tracked rotation axis.  
- **Equation:**  
  φₜ = sin⁻¹[ (āₜ × r̄ₜ) · aₜ ]
   - āₜ : mean gravity vector
   - r̄ₜ : smoothed rotation axis
   - aₜ : current acceleration sample
- **Context:** Corresponds to Eq. (6) in Bates et al. (2010).

**`computeRespWaveform.py`**  
- **Purpose:** Convert the breathing angle signal into a flow-like waveform.  
- **Method:** Numerically differentiate φₜ with respect to time, then apply low-pass filtering to suppress noise:  
     ωₜ = dφₜ/dt
- **Context:** This is analogous to estimating respiratory flow rate from angular velocity as validated in Bates et al. (2010).

**End result:** The differentiated and filtered waveform φₜ closely resembles nasal cannula flow, enabling extraction of respiratory rate and waveform shape during static periods.

---

### End-to-end orchestration

**`respWaveformEstimator.py`**  
- **Purpose:** Implements the complete respiration waveform estimation pipeline described in Bates et al. (2010).  
- **Method:**  
   1. **Filter accelerometer data:** Apply low-pass filtering to suppress high-frequency noise.  
   2. **Detect static periods:** Compute inter-sample rotation angles and mask out segments exceeding the motion threshold.  
   3. **Track rotation axis:** Use `trackRotationAxis` to estimate a smoothed, sign-consistent breathing rotation axis r̄ₜ.  
   4. **Compute mean gravity vector:** Apply `computeMeanAccel` to obtain āₜ.  
   5. **Calculate breathing angle:** Use `computeBreathingAngle` to determine φₜ from āₜ, r̄ₜ, and aₜ.  
   6. **Generate respiratory waveform:** Apply `computeRespWaveform` to obtain the angular velocity ωₜ (flow-like signal).  


- **Output:**  
  - A time series ωₜ representing breathing-induced angular velocity, suitable for respiratory rate estimation and waveform shape analysis.  
  - Processed only over periods where the subject is stationary, improving agreement with nasal cannula measurements as reported in Bates et al. (2010).  





## References

Bates, A., Ling, M. J., Mann, J., & Arvind, D. K. (2010). Respiratory rate and flow waveform estimation from tri-axial accelerometer data. _2010 International Conference on Body Sensor Networks (BSN)_, 144–150. https://doi.org/10.1109/BSN.2010.50
>>>>>>> 0dd15ce (Respiratory Waveform Estimation from Accelerometer Data)
