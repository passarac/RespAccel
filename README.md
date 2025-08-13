# RespAccel
Generate respiratory signals from tri-axal accelerometer data from a chest-worn sensor. Apply cleaning and filtering to identify valid and reliable segments of respiratory data.

=======
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
     $$
     \theta_t = \cos^{-1}(\hat{a}_t \cdot \hat{a}_{t-1})
     $$  
   - Discard segments where $\theta_t$ exceeds the motion threshold, ensuring only breathing-related motion is analysed.

3. **Find and track the breathing rotation axis**  
   - Compute raw rotation axes:  
     $$
     r_t = \hat{a}_t \times \hat{a}_{t-1}
     $$  
   - Align signs using a reference axis from PCA.  
   - Smooth using a Hamming window, optionally weighted by $\theta_t$, to obtain $\bar{r}_t$.

4. **Estimate the mean gravity vector**  
   - Compute a sliding-window mean of the acceleration to estimate $\bar{a}_t$, the local gravity direction.

5. **Calculate the breathing angle**  
   - Determine the signed angular displacement of the chest around $\bar{r}_t$:  
     $$
     \phi_t = \sin^{-1}\left[ (\bar{a}_t \times \bar{r}_t) \cdot a_t \right]
     $$  

6. **Differentiate to obtain a flow-like signal**  
   - Numerically differentiate $\phi_t$ with respect to time:  
     $$
     \omega_t = \frac{d\phi_t}{dt}
     $$  
   - Apply low-pass filtering to reduce noise.  
   - The result $\omega_t$ closely resembles nasal cannula flow, enabling respiratory rate and waveform shape estimation.

---

### Movement handling & data conditioning

**`filter.py`**  
- **Low-pass filtering:** Smooth the raw tri-axial accelerometer data using a zero-phase Butterworth filter to suppress high-frequency noise.  
- **Large motion detection:** Calculate the inter-sample rotation angle  
$$
\theta_t = \cos^{-1}(\hat{a}_t \cdot \hat{a}_{t-1})
$$ 
  and apply an angle threshold to identify static periods.  
- **Purpose:** Restrict further analysis to periods where the accelerometer signal reflects breathing-related chest wall motion, rather than whole-body movements.  
- **Context (Bates et al., 2010):** This matches the paper’s movement detection step using a maximum rotation angle per sample to exclude high-motion segments, improving agreement between accelerometer-derived and nasal cannula waveforms.

---

### Axis of rotation estimation & tracking

**`computeRotationAxes.py`**  
- **Purpose:** Compute the instantaneous rotation axis between consecutive normalised acceleration vectors.  
- **Equation:**  
  $$
  r_t = \hat{a}_t \times \hat{a}_{t-1}
  $$  
  This corresponds to Eq. (2) in Bates et al. (2010).  

**`findReferenceAxis.py`**  
- **Purpose:** Select a consistent global sign/direction for all rotation axes using PCA.  
- **Method:** Identify the principal component of the set of $r_t$ vectors and use it as the reference axis $r_{\text{ref}}$. Flip signs so that  
  $$
  r_t' =
  \begin{cases}
    r_t, & \text{if } r_t \cdot r_{\text{ref}} \ge 0 \\
    -r_t, & \text{otherwise}
  \end{cases}
  $$  
  This corresponds to Eq. (3) in the paper.  

**`trackRotationAxis.py`**  
- **Purpose:** Produce a smoothed, sign-consistent rotation axis $\bar{r}_t$ over time.  
- **Method:**  
  1. Apply a sliding Hamming window to $r_t'$ values.  
  2. Optionally weight each vector by the inter-sample rotation angle $\theta_t$.  
  3. Average and normalise to unit length:  
     $$
     \bar{r}_t = \mathrm{normalize} \left( \sum_{i \in W_t} H(i) \, \theta_{t+i} \, r'_{t+i} \right)
     $$  
  This matches Eq. (4) in Bates et al. (2010).

---

### Breathing angle and flow-like signal

**`computeMeanAccel.py`**  
- **Purpose:** Estimate the mean gravity vector $\bar{a}_t$ over a short time window.  
- **Method:** Compute a moving average of the normalised acceleration samples to reduce noise.  
- **Context:** This matches Eq. (5) in Bates et al. (2010).

**`computeBreathingAngle.py`**  
- **Purpose:** Calculate the instantaneous breathing angle $\phi_t$ around the tracked rotation axis.  
- **Equation:**  
  $$
  \phi_t = \sin^{-1} \left[ \left( \bar{a}_t \times \bar{r}_t \right) \cdot a_t \right]
  $$  
  - $\bar{a}_t$: mean gravity vector  
  - $\bar{r}_t$: smoothed rotation axis  
  - $a_t$: current acceleration sample  
- **Context:** Corresponds to Eq. (6) in Bates et al. (2010).

**`computeRespWaveform.py`**  
- **Purpose:** Convert the breathing angle signal into a flow-like waveform.  
- **Method:** Numerically differentiate $\phi_t$ with respect to time, then apply low-pass filtering to suppress noise:  
  $$
  \omega_t = \frac{d\phi_t}{dt}
  $$  
- **Context:** This is analogous to estimating respiratory flow rate from angular velocity as validated in Bates et al. (2010).

**End result:** The differentiated and filtered waveform $\omega_t$ closely resembles nasal cannula flow, enabling extraction of respiratory rate and waveform shape during static periods.

---

### End-to-end orchestration

**`respWaveformEstimator.py`**  
- **Purpose:** Implements the complete respiration waveform estimation pipeline described in Bates et al. (2010).  
- **Method:**  
  1. **Filter accelerometer data:** Apply low-pass filtering to suppress high-frequency noise.  
  2. **Detect static periods:** Compute inter-sample rotation angles and mask out segments exceeding the motion threshold.  
  3. **Track rotation axis:** Use `trackRotationAxis` to estimate a smoothed, sign-consistent breathing rotation axis $\bar{r}_t$.  
  4. **Compute mean gravity vector:** Apply `computeMeanAccel` to obtain $\bar{a}_t$.  
  5. **Calculate breathing angle:** Use `computeBreathingAngle` to determine $\phi_t$ from $\bar{a}_t$, $\bar{r}_t$, and $a_t$.  
  6. **Generate respiratory waveform:** Apply `computeRespWaveform` to obtain the angular velocity $\omega_t$ (flow-like signal).  

- **Output:**  
  - A time series $\omega_t$ representing breathing-induced angular velocity, suitable for respiratory rate estimation and waveform shape analysis.  
  - Processed only over periods where the subject is stationary, improving agreement with nasal cannula measurements as reported in Bates et al. (2010).




## References

Bates, A., Ling, M. J., Mann, J., & Arvind, D. K. (2010). Respiratory rate and flow waveform estimation from tri-axial accelerometer data. _2010 International Conference on Body Sensor Networks (BSN)_, 144–150. https://doi.org/10.1109/BSN.2010.50
>>>>>>> 0dd15ce (Respiratory Waveform Estimation from Accelerometer Data)
