import numpy as np

from .computeBreathingAngle import compute_breathing_angle
from .computeMeanAccel import compute_mean_accel
from .filter import filter_accelerometer_data, detect_large_motions
from .trackRotationAxis import track_rotation_axis
from .computeRespWaveform import *

class RespWaveformEstimator:
    def __init__(self, sampling_freq=12.5, window_size=10, angle_threshold=5e-3):
        self.sampling_freq = sampling_freq
        self.window_size = window_size
        self.angle_threshold = angle_threshold


    def estimateRespWaveform(self, accel_data):
        """
        Estimate respiratory waveform from tri-axial accelerometer data.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 1) Low-pass filter raw accel
        filtered = filter_accelerometer_data(
            accel_data, cutoff=2.0, fs=self.sampling_freq, order=2
        )

        # 2) Normalise filtered accel to unit vectors
        norms = np.linalg.norm(filtered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        filtered_norm = filtered / norms

        # 3) Detect static periods
        static_mask = detect_large_motions(filtered_norm, angle_threshold=self.angle_threshold)
        static_acc = filtered_norm[static_mask]

        if static_acc.size == 0:
            return np.array([])  # No static segments

        # 4) Track rotation axis (longer smoothing window)
        smoothed_axes = track_rotation_axis(
            static_acc, 
            window_size=int(self.sampling_freq * 3),  # ~3 sec
            angle_weight=True
        )

        # Ensure smoothed axes are unit vectors
        sa_norm = np.linalg.norm(smoothed_axes, axis=1, keepdims=True)
        sa_norm[sa_norm == 0] = 1.0
        smoothed_axes = smoothed_axes / sa_norm

        # 5) Compute mean gravity vector āₜ (longer window)
        mean_acc = compute_mean_accel(filtered_norm, window_size=int(self.sampling_freq * 2))
        ma_norm = np.linalg.norm(mean_acc, axis=1, keepdims=True)
        ma_norm[ma_norm == 0] = 1.0
        mean_acc = mean_acc / ma_norm

        # 6) Align lengths
        min_len = min(filtered_norm.shape[0], mean_acc.shape[0], smoothed_axes.shape[0])
        a_t = filtered_norm[:min_len]
        mean_acc = mean_acc[:min_len]
        smoothed_axes = smoothed_axes[:min_len]

        # 7) Compute breathing angle φₜ
        phi_t = compute_breathing_angle(a_t, mean_acc, smoothed_axes)

        # 8) Derive respiratory waveform ωₜ (lower cutoff to smooth noise)
        resp_wave = compute_resp_waveform(phi_t, fs=self.sampling_freq, do_smooth=True, cutoff=0.7)

        # Debug: compare φₜ and ωₜ
        plt.figure(figsize=(10, 4))
        plt.plot(phi_t, label='phi_t (angle)')
        plt.plot(resp_wave, label='omega_t (flow-like)')
        plt.legend()
        plt.title('Debug: Breathing Angle vs Respiratory Waveform')
        plt.show()

        return resp_wave
