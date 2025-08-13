"""
Python re‑implementation of the RESpeck C breathing library.

This module provides a pure‐Python translation of the algorithms originally
implemented in C for computing a breathing signal and breathing rate from
tri‑axial accelerometer data.  The C library was designed for the RESpeck
wearable sensor and exposes a set of functions via a shared library
(`respeck.dll`/`respeck.so`) that are typically accessed from Python via
``ctypes``.  This re‑implementation avoids the need for any compiled
components by reproducing the same state machines and numerical routines
directly in Python.

Usage overview
==============

The top–level entry point is the :class:`RespeckBreathing` class.  You
create a single instance of this class, call :meth:`initBreathing` once
to configure internal buffers and thresholds, and then repeatedly feed
new accelerometer samples via :meth:`updateBreathing`.  After each call
to :meth:`updateBreathing` you can retrieve the latest breathing
metrics:

.. code:: python

   from respeck_breathing import RespeckBreathing

   # Create an instance
   rb = RespeckBreathing()

   # Configure the algorithm (these values mirror those used in the C code)
   rb.initBreathing(
       pre_filter_length=11,
       post_filter_length=12,
       activity_cutoff=0.3,
       threshold_filter_size=50,
       lower_threshold_limit=0.001,
       upper_threshold_limit=0.14,
       threshold_factor=4.2,
       sampling_frequency=12.5,
   )

   # Stream your accelerometer samples (x,y,z) into the algorithm
   for x, y, z in your_data_stream:
       rb.updateBreathing(x, y, z)
       rate = rb.getBreathingRate()      # instantaneous breathing rate
       signal = rb.getBreathingSignal()  # filtered breathing signal
       upper = rb.getUpperThreshold()    # dynamic positive threshold
       lower = rb.getLowerThreshold()    # dynamic negative threshold
       # ... use these values as needed

If you have an external sensor that directly provides a breathing
signal instead of raw accelerometer values, you can call
:meth:`updateBreathingSignal` instead of :meth:`updateBreathing`.  All
other methods remain the same.

Under the hood the implementation replicates the C code line‑for‑line
wherever possible.  Comments from the original sources are preserved as
Python docstrings to aid understanding.  The algorithm maintains
multiple circular buffers to smooth the acceleration data, estimate
rotational motion, derive a breathing waveform, compute dynamic
thresholds via a root‑mean‑square calculation and detect individual
breaths using a simple state machine.  A step counter and activity
predictor are also included to suppress breathing estimation during
periods of vigorous movement.
"""

from __future__ import annotations

import math
from typing import List, Optional

FloatList = List[float]


###############################################################################
# Constants mirroring the original C header files
###############################################################################

# Breathing detection limits
HIGHEST_POSSIBLE_BREATHING_RATE: float = 45.0
LOWEST_POSSIBLE_BREATHING_RATE: float = 5.0
NUMBER_OF_ABNORMAL_BREATHS_SWITCH: int = 3

# Step counter configuration
VECTOR_LENGTH_BUFFER_SIZE: int = 3
STEP_THRESHOLD: float = 1.02
STEP_MIN_DELAY_SAMPLES: int = 2
NUM_STEPS_UNTIL_COUNT_WALKING: int = 6
MAX_ALLOWED_DEVIATION_FROM_MEAN_DISTANCE: float = 6.0
NUM_SAMPLES_UNTIL_STATIC: int = 20

# Activity level buffer
ACTIVITY_LEVEL_BUFFER_SIZE: int = 32

# Mean rotation axis buffer
MEAN_AXIS_SIZE: int = 128

# Breathing rate statistics
BREATHING_RATES_BUFFER_SIZE: int = 50
DISCARD_UPPER_BREATHING_RATES: int = 2
DISCARD_LOWER_BREATHING_RATES: int = 2

# Threshold value type enumeration (mirrors C enum)
POSITIVE: int = 0
INVALID: int = 1
NEGATIVE: int = 2

# Breath state machine enumeration (mirrors C enum)
LOW: int = 0
MID_FALLING: int = 1
MID_UNKNOWN: int = 2
MID_RISING: int = 3
HIGH: int = 4
UNKNOWN: int = 5

# Activity prediction codes
ACTIVITY_STAND_SIT: int = 0
ACTIVITY_WALKING: int = 1
ACTIVITY_LYING: int = 2
ACTIVITY_WRONG_ORIENTATION: int = 3
ACTIVITY_SITTING_BENT_FORWARD: int = 4
ACTIVITY_SITTING_BENT_BACKWARD: int = 5
ACTIVITY_LYING_DOWN_RIGHT: int = 6
ACTIVITY_LYING_DOWN_LEFT: int = 7
ACTIVITY_LYING_DOWN_STOMACH: int = 8
ACTIVITY_MOVEMENT: int = 9


def calculate_vector_length(vector: FloatList) -> float:
    """Calculate the Euclidean length of a 3‑D vector.

    Equivalent to the C function ``calculate_vector_length``.

    :param vector: A list or tuple of three floats.
    :returns: The Euclidean norm of the vector.
    """
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def dot_product(v: FloatList, u: FloatList) -> float:
    """Compute the dot product of two 3‑D vectors."""
    return v[0] * u[0] + v[1] * u[1] + v[2] * u[2]


def cross_product(v: FloatList, u: FloatList) -> List[float]:
    """Compute the cross product ``v x u`` for two 3‑D vectors."""
    return [
        v[1] * u[2] - v[2] * u[1],
        v[2] * u[0] - v[0] * u[2],
        v[0] * u[1] - v[1] * u[0],
    ]


def normalise_vector_to_unit_length(vector: FloatList) -> None:
    """In‑place normalisation of a 3‑D vector to unit length.

    If the vector has zero length it is left unchanged.  This mirrors
    the C function ``normalise_vector_to_unit_length``.
    """
    length = calculate_vector_length(vector)
    if length != 0.0:
        vector[0] /= length
        vector[1] /= length
        vector[2] /= length


###############################################################################
# Circular buffer implementations
###############################################################################

class VectorLengthBuffer:
    """A fixed‑size buffer that computes a running mean of vector lengths.

    This structure stores the magnitudes of the last
    :data:`VECTOR_LENGTH_BUFFER_SIZE` acceleration vectors and exposes
    ``mean_length`` whenever the buffer has been completely filled.  It is
    used by :class:`StepCounter` to smooth the accelerometer magnitude
    signal.
    """

    def __init__(self) -> None:
        self.fill: int = 0
        self.current_position: int = -1
        self.values: List[float] = [0.0] * VECTOR_LENGTH_BUFFER_SIZE
        self.mean_length: float = 0.0
        self.is_valid: bool = False
        self.sum: float = 0.0

    def update(self, new_vector_length: float) -> None:
        """Add a new vector length to the buffer and recompute the mean.

        :param new_vector_length: The magnitude of the latest accelerometer
            sample.
        """
        # Advance the ring buffer index
        self.current_position = (self.current_position + 1) % VECTOR_LENGTH_BUFFER_SIZE

        # Remove the old value from the running sum and insert the new one
        self.sum -= self.values[self.current_position]
        self.values[self.current_position] = new_vector_length
        self.sum += new_vector_length

        # Keep track of how many samples have been seen
        if self.fill < VECTOR_LENGTH_BUFFER_SIZE:
            self.fill += 1

        # Once the buffer is full we can compute a valid mean
        if self.fill < VECTOR_LENGTH_BUFFER_SIZE:
            self.is_valid = False
            return

        self.mean_length = self.sum / VECTOR_LENGTH_BUFFER_SIZE
        self.is_valid = True


class StepCounter:
    """Implementation of the step counting state machine.

    The algorithm tracks acceleration magnitude over a short window to
    detect threshold crossings that indicate steps.  A number of
    consecutive steps with regular timing must occur before the state
    changes to ``WALKING``; otherwise the wearer is considered ``MOVING``
    or ``STATIC``.  Steps counted while walking are accumulated in
    ``minute_step_count`` which can be queried via
    :meth:`getMinuteStepcount`.
    """

    STATIC: int = 0
    MOVING: int = 1
    WALKING: int = 2

    def __init__(self) -> None:
        self.previous_vector_length: float = 0.0
        self.step_distances: List[int] = [0] * (NUM_STEPS_UNTIL_COUNT_WALKING - 1)
        self.samples_since_last_step: int = 0
        self.num_valid_steps: int = 0
        self.num_samples_until_static: int = 0
        self.minute_step_count: int = 0
        self.vector_length_buffer: VectorLengthBuffer = VectorLengthBuffer()
        self.current_state: int = StepCounter.STATIC

    def update(self, accel: FloatList) -> None:
        """Update the step counter with a new acceleration vector.

        The acceleration vector should be a sequence of three floats
        representing the x, y and z axes.  Only when the internal
        :class:`VectorLengthBuffer` is full will the step detection
        logic run.
        """
        # Compute the magnitude of the current acceleration and update the mean buffer
        vector_length = calculate_vector_length(accel)
        self.vector_length_buffer.update(vector_length)

        if not self.vector_length_buffer.is_valid:
            return

        mean_length = self.vector_length_buffer.mean_length
        self.samples_since_last_step += 1

        if (
            mean_length > STEP_THRESHOLD
            and self.previous_vector_length <= STEP_THRESHOLD
            and self.samples_since_last_step > STEP_MIN_DELAY_SAMPLES
        ):
            # A threshold crossing has occurred
            self.num_samples_until_static = NUM_SAMPLES_UNTIL_STATIC

            if self.current_state == StepCounter.STATIC:
                # First step detected after a period of inactivity
                self.num_valid_steps = 1
                self.current_state = StepCounter.MOVING
            elif self.current_state == StepCounter.MOVING:
                # Additional step while moving
                self.num_valid_steps += 1
                # Record the number of samples since the last step
                # When num_valid_steps == 2 this writes to index 0, etc.
                if self.num_valid_steps - 2 >= 0:
                    idx = self.num_valid_steps - 2
                    if idx < len(self.step_distances):
                        self.step_distances[idx] = self.samples_since_last_step
                # Once enough steps have accumulated we determine if walking
                if self.num_valid_steps == NUM_STEPS_UNTIL_COUNT_WALKING:
                    # Calculate the mean distance between steps
                    mean_distance = sum(self.step_distances) / (NUM_STEPS_UNTIL_COUNT_WALKING - 1)
                    valid_walking = True
                    for dist in self.step_distances:
                        if abs(mean_distance - dist) > MAX_ALLOWED_DEVIATION_FROM_MEAN_DISTANCE:
                            valid_walking = False
                            break
                    if valid_walking:
                        self.current_state = StepCounter.WALKING
                        self.minute_step_count += self.num_valid_steps
                    else:
                        # Revert to static if timings are irregular
                        self.current_state = StepCounter.STATIC
            else:
                # Already in walking state: count a new step
                self.minute_step_count += 1

            # Reset step timer
            self.samples_since_last_step = 0
        else:
            # No threshold crossing: if moving or walking then we may decay to static
            if self.current_state != StepCounter.STATIC:
                self.num_samples_until_static -= 1
                if self.num_samples_until_static == 0:
                    self.current_state = StepCounter.STATIC

        self.previous_vector_length = mean_length

    def is_walking(self) -> bool:
        """Return ``True`` if the current state is ``WALKING``."""
        return self.current_state == StepCounter.WALKING

    def is_moving(self) -> bool:
        """Return ``True`` if the current state is ``MOVING``."""
        return self.current_state == StepCounter.MOVING


class ActivityPredictor:
    """Basic posture and movement classifier.

    This class implements the simple threshold‑based activity
    classification used in the RESpeck firmware.  It categorises a
    single acceleration vector into one of several posture or movement
    states based on orientation and the output of the step counter.
    """

    def __init__(self) -> None:
        self.last_prediction: int = -1

    def initialise(self) -> None:
        """Reset the activity predictor to its initial state."""
        self.last_prediction = -1

    def update(self, accel: FloatList, step_counter: StepCounter) -> None:
        """Update the classification based on a new accelerometer sample.

        :param accel: Current acceleration vector.
        :param step_counter: The step counter tracking walking/moving
            status.
        """
        self.last_prediction = self.get_advanced_activity_prediction(
            accel[0], accel[1], accel[2], step_counter.is_walking(), step_counter.is_moving()
        )

    @staticmethod
    def get_advanced_activity_prediction(
        x: float, y: float, z: float, is_walking: bool, is_moving: bool
    ) -> int:
        """Classify the current posture or motion.

        This function mirrors ``get_advanced_activity_prediction`` from
        the C implementation.  It uses simple thresholds on the
        accelerometer orientation (assuming the device is worn on the
        chest) and the output of the step counter.
        """
        if is_walking:
            return ACTIVITY_WALKING
        if is_moving:
            return ACTIVITY_MOVEMENT

        # Lying on the side?
        if x < -0.66129816:
            return ACTIVITY_LYING_DOWN_RIGHT
        elif x > 0.67874145:
            return ACTIVITY_LYING_DOWN_LEFT
        elif y > -0.28865455:
            # Lying either on back or stomach
            if z > 0:
                return ACTIVITY_LYING
            else:
                return ACTIVITY_LYING_DOWN_STOMACH
        else:
            # Not lying: check for leaning forwards/backwards
            if z > 0.43:
                return ACTIVITY_SITTING_BENT_BACKWARD
            elif z < -0.43:
                return ACTIVITY_SITTING_BENT_FORWARD
            return ACTIVITY_STAND_SIT


class ActivityLevelBuffer:
    """Buffer for recent activity levels used to suppress breathing detection.

    The *activity level* is defined as the norm of the difference
    between successive accelerometer vectors.  A high value indicates
    rapid movement; if the maximum activity level over a window of
    :data:`ACTIVITY_LEVEL_BUFFER_SIZE` samples exceeds the configured
    ``activity_cutoff`` then the breathing signal is not updated.
    """

    def __init__(self) -> None:
        self.fill: int = 0
        self.current_position: int = -1
        self.activity_levels: List[float] = [0.0] * ACTIVITY_LEVEL_BUFFER_SIZE
        self.previous_accel: List[float] = [0.0, 0.0, 0.0]
        self.previous_accel_valid: bool = False
        self.max: float = 0.0
        self.is_valid: bool = False

    def update(self, current_accel: FloatList) -> None:
        """Add a new accelerometer vector and update the activity metric.

        :param current_accel: New acceleration vector as a list of
            three floats.  If any component is NaN the buffer is
            invalidated for this iteration.
        """
        # Increment circular buffer index
        self.current_position = (self.current_position + 1) % ACTIVITY_LEVEL_BUFFER_SIZE

        # Reject NaN values (should never occur under normal usage)
        if math.isnan(current_accel[0]) or math.isnan(current_accel[1]) or math.isnan(current_accel[2]):
            self.is_valid = False
            return

        if not self.previous_accel_valid:
            # Initialise the previous acceleration vector
            self.previous_accel = current_accel.copy()
            self.previous_accel_valid = True
            self.is_valid = False
            return

        # Compute the activity level as the Euclidean distance between
        # current and previous acceleration samples
        dx = current_accel[0] - self.previous_accel[0]
        dy = current_accel[1] - self.previous_accel[1]
        dz = current_accel[2] - self.previous_accel[2]
        current_level = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.activity_levels[self.current_position] = current_level

        # Increase the number of filled entries until we reach the
        # configured window size
        if self.fill < ACTIVITY_LEVEL_BUFFER_SIZE:
            self.fill += 1

        # If the buffer isn't full yet, we cannot compute a maximum
        if self.fill < ACTIVITY_LEVEL_BUFFER_SIZE:
            self.is_valid = False
            return

        # Determine the maximum activity level observed over the window
        self.max = self.activity_levels[0]
        for val in self.activity_levels[1:]:
            if val > self.max:
                self.max = val

        # Update previous acceleration for next iteration
        self.previous_accel = current_accel.copy()
        self.is_valid = True


class MeanUnitAccelBuffer:
    """Compute a running mean of accelerometer vectors and normalise it.

    Both the pre‑filter (small window) and the longer mean acceleration
    buffer used when computing the reference orientation are instances
    of this class.  Internally it maintains a circular list of
    acceleration vectors, accumulates their component‑wise sum and
    outputs the (normalised) mean once the buffer has been filled.
    """

    def __init__(self, buffer_size: int) -> None:
        self.sum: List[float] = [0.0, 0.0, 0.0]
        self.current_position: int = -1
        self.fill: int = 0
        self.buffer_size: int = buffer_size
        self.mean_unit_vector: List[float] = [0.0, 0.0, 0.0]
        self.is_valid: bool = False
        # The list of acceleration vectors stored in the buffer
        self.values: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(buffer_size)]

    def update(self, new_accel_data: FloatList) -> None:
        """Add a new acceleration vector to the buffer and recompute the mean.

        :param new_accel_data: Current acceleration vector as a list of
            three floats.
        """
        # Advance circular buffer index
        self.current_position = (self.current_position + 1) % self.buffer_size

        # Subtract the vector being overwritten from the running sum
        old_vec = self.values[self.current_position]
        self.sum[0] -= old_vec[0]
        self.sum[1] -= old_vec[1]
        self.sum[2] -= old_vec[2]

        # Store the new vector and update the running sum
        new_vec = new_accel_data.copy()
        self.values[self.current_position] = new_vec
        self.sum[0] += new_vec[0]
        self.sum[1] += new_vec[1]
        self.sum[2] += new_vec[2]

        # Track how many valid entries are present
        if self.fill < self.buffer_size:
            self.fill += 1

        # If the buffer isn't yet full then we can't emit a valid mean
        if self.fill < self.buffer_size:
            self.is_valid = False
            return

        # Compute the mean vector (not divided by buffer_size to mirror C code)
        self.mean_unit_vector = self.sum.copy()
        # Convert to unit length
        normalise_vector_to_unit_length(self.mean_unit_vector)
        self.is_valid = True


class RotationAxis:
    """Compute the instantaneous rotation axis of the acceleration vector.

    The rotation axis is the cross product of the current and previous
    accelerometer vectors.  Each call to :meth:`update` updates the
    ``previous_accel_data`` and sets ``is_current_axis_valid`` once a
    cross product can be computed.
    """

    def __init__(self) -> None:
        self.previous_accel_data: List[float] = [0.0, 0.0, 0.0]
        self.is_previous_accel_data_valid: bool = False
        self.current_axis: List[float] = [0.0, 0.0, 0.0]
        self.is_current_axis_valid: bool = False

    def update(self, new_accel_data: FloatList) -> None:
        """Update the rotation axis from a new acceleration sample."""
        if not self.is_previous_accel_data_valid:
            # First sample: store it and wait for the next call
            self.previous_accel_data = new_accel_data.copy()
            self.is_previous_accel_data_valid = True
            self.is_current_axis_valid = False
            return
        # Compute cross product of previous and new vectors
        cross = cross_product(self.previous_accel_data, new_accel_data)
        self.current_axis = cross.copy()
        # Update the previous vector
        self.previous_accel_data = new_accel_data.copy()
        self.is_current_axis_valid = True


class MeanRotationAxisBuffer:
    """Maintain a running mean of rotation axes with consistent polarity.

    The rotation axis points from one acceleration vector to the next and
    therefore its sign may flip arbitrarily.  To obtain a consistent
    orientation the dot product between the current axis and a fixed
    reference axis is computed.  If the dot product is negative the
    current axis is inverted before it is accumulated.  Once the
    internal buffer of length :data:`MEAN_AXIS_SIZE` has been filled
    the mean axis is normalised and made available via the ``mean_axis``
    attribute.
    """

    def __init__(self) -> None:
        self.sum: List[float] = [0.0, 0.0, 0.0]
        self.current_position: int = -1
        self.fill: int = 0
        # Storage for the raw axes
        self.accel_buffer: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(MEAN_AXIS_SIZE)]
        self.mean_axis: List[float] = [0.0, 0.0, 0.0]
        self.is_valid: bool = False

    @staticmethod
    def _get_reference_axis() -> List[float]:
        """Return the reference axis used to enforce a consistent sign.

        These values were extracted from the original C code and
        represent an empirically chosen orientation.
        """
        return [0.98499424, -0.17221591, 0.01131468]

    def update(self, new_axis: FloatList) -> None:
        """Add a new rotation axis and recompute the running mean.

        :param new_axis: A 3‑D vector representing the current rotation axis.
        """
        # Advance circular buffer index
        self.current_position = (self.current_position + 1) % MEAN_AXIS_SIZE

        # Subtract the vector being overwritten from the running sum
        old_vec = self.accel_buffer[self.current_position]
        self.sum[0] -= old_vec[0]
        self.sum[1] -= old_vec[1]
        self.sum[2] -= old_vec[2]

        # Copy the new axis into the buffer
        new_vec = new_axis.copy()
        # Enforce consistent polarity with respect to the reference axis
        reference_axis = self._get_reference_axis()
        if dot_product(new_vec, reference_axis) < 0.0:
            new_vec[0] = -new_vec[0]
            new_vec[1] = -new_vec[1]
            new_vec[2] = -new_vec[2]
        self.accel_buffer[self.current_position] = new_vec

        # Update the running sum
        self.sum[0] += new_vec[0]
        self.sum[1] += new_vec[1]
        self.sum[2] += new_vec[2]

        # Track how many valid entries are present
        if self.fill < MEAN_AXIS_SIZE:
            self.fill += 1

        if self.fill < MEAN_AXIS_SIZE:
            self.is_valid = False
            return

        # Compute the mean axis and normalise
        self.mean_axis = self.sum.copy()
        normalise_vector_to_unit_length(self.mean_axis)
        self.is_valid = True


class MeanPostFilter:
    """A simple moving average filter for smoothing scalar values.

    Each :class:`MeanPostFilter` maintains a circular buffer of a
    configurable length and produces the average of all stored values.
    It is used to smooth both the breathing signal and the derived
    breathing angle.
    """

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.values: List[float] = [0.0] * buffer_size
        self.sum: float = 0.0
        self.current_position: int = -1
        self.fill: int = 0
        self.mean_value: float = 0.0
        self.is_valid: bool = False

    def update(self, value: float) -> None:
        """Add a new value to the filter and update the mean."""
        # Advance index
        self.current_position = (self.current_position + 1) % self.buffer_size

        # Subtract the overwritten value and add the new one
        self.sum -= self.values[self.current_position]
        self.values[self.current_position] = value
        self.sum += value

        # Increase the fill level
        if self.fill < self.buffer_size:
            self.fill += 1

        if self.fill < self.buffer_size:
            self.is_valid = False
            return

        # Compute the mean
        self.mean_value = self.sum / self.buffer_size
        self.is_valid = True


class ThresholdBuffer:
    """Root‑mean‑square threshold estimation for the breathing signal.

    Positive and negative portions of the breathing signal are
    accumulated separately to compute independent RMS values.  The
    resulting upper and lower thresholds are updated whenever the
    internal buffer is full.  Invalid samples (NaNs) are ignored.
    """

    def __init__(self, threshold_filter_size: int) -> None:
        self.threshold_filter_size: int = threshold_filter_size
        self.values: List[float] = [0.0] * threshold_filter_size
        # Store which bucket each value belongs to: POSITIVE, NEGATIVE or INVALID
        self.values_type: List[int] = [INVALID] * threshold_filter_size
        self.upper_values_sum_fill: int = 0
        self.lower_values_sum_fill: int = 0
        self.upper_values_sum: float = 0.0
        self.lower_values_sum: float = 0.0
        self.current_position: int = -1
        self.fill: int = 0
        self.is_valid: bool = False
        self.upper_threshold_value: float = float("nan")
        self.lower_threshold_value: float = float("nan")

    def update(self, breathing_signal_value: float) -> None:
        """Update the running RMS thresholds with a new breathing signal sample."""
        # Advance index
        self.current_position = (self.current_position + 1) % self.threshold_filter_size

        # Remove the old value from the appropriate sum
        old_type = self.values_type[self.current_position]
        old_value = self.values[self.current_position]
        if old_type == POSITIVE:
            self.upper_values_sum -= old_value
            self.upper_values_sum_fill -= 1
        elif old_type == NEGATIVE:
            self.lower_values_sum -= old_value
            self.lower_values_sum_fill -= 1
        # If old_type == INVALID no sums are affected

        # Store the new sample
        if math.isnan(breathing_signal_value):
            # Mark this slot as invalid and do not include it in any sum
            self.values_type[self.current_position] = INVALID
        else:
            squared_value = breathing_signal_value * breathing_signal_value
            self.values[self.current_position] = squared_value
            if breathing_signal_value >= 0.0:
                self.values_type[self.current_position] = POSITIVE
                self.upper_values_sum_fill += 1
                self.upper_values_sum += squared_value
            else:
                self.values_type[self.current_position] = NEGATIVE
                self.lower_values_sum_fill += 1
                self.lower_values_sum += squared_value

        # Maintain the count of filled entries
        if self.fill < self.threshold_filter_size:
            self.fill += 1

        # If the buffer isn't full yet, no valid thresholds are available
        if self.fill < self.threshold_filter_size:
            self.is_valid = False
            return

        # Compute RMS of positive values
        if self.upper_values_sum_fill > 0:
            self.upper_threshold_value = math.sqrt(self.upper_values_sum / self.upper_values_sum_fill)
        else:
            self.upper_threshold_value = float("nan")
        # RMS of negative values (note the negative sign)
        if self.lower_values_sum_fill > 0:
            self.lower_threshold_value = -math.sqrt(self.lower_values_sum / self.lower_values_sum_fill)
        else:
            self.lower_threshold_value = float("nan")

        self.is_valid = True


class CurrentBreath:
    """State machine for detecting breaths and estimating breathing rate."""

    def __init__(
        self,
        lower_threshold_limit: float,
        upper_threshold_limit: float,
        sampling_frequency: float,
    ) -> None:
        self.state: int = UNKNOWN
        self.breathing_rate: float = float("nan")
        self.min_threshold: float = lower_threshold_limit
        self.max_threshold: float = upper_threshold_limit
        self.sample_count: int = 0
        self.is_current_breath_valid: bool = False
        self.is_complete: bool = False
        self.sampling_frequency: float = sampling_frequency
        self.first_part_length: int = 0
        self.is_inspiration_above_x: bool = True
        self.count_abnormal_breaths: int = 0

    def reset(self) -> None:
        """Reset the state machine to its initial configuration."""
        self.state = UNKNOWN
        self.breathing_rate = float("nan")
        self.sample_count = 0
        self.is_current_breath_valid = False
        self.is_complete = False
        self.first_part_length = 0
        self.is_inspiration_above_x = True
        self.count_abnormal_breaths = 0

    def _end_breath(self) -> None:
        """Handle the completion of a breath cycle and compute the rate."""
        if self.is_current_breath_valid:
            # An abnormal breath is one where the first part (inspiration or expiration) is longer
            if self.first_part_length > (self.sample_count - self.first_part_length):
                self.count_abnormal_breaths += 1
            else:
                self.count_abnormal_breaths = 0
            # If too many abnormal breaths occur consecutively, flip the expected orientation
            if self.count_abnormal_breaths >= NUMBER_OF_ABNORMAL_BREATHS_SWITCH:
                self.count_abnormal_breaths = 0
                self.is_inspiration_above_x = not self.is_inspiration_above_x
            else:
                # Compute the breathing rate (breaths per minute)
                new_rate = 60.0 * self.sampling_frequency / float(self.sample_count)
                if LOWEST_POSSIBLE_BREATHING_RATE <= new_rate <= HIGHEST_POSSIBLE_BREATHING_RATE:
                    self.breathing_rate = new_rate
                    self.is_complete = True
        # Reset for next breath cycle
        self.sample_count = 0
        self.first_part_length = 0
        self.is_current_breath_valid = True

    def update(self, breathing_signal: float, upper_threshold: float, lower_threshold: float) -> None:
        """Update the breath detection state machine.

        :param breathing_signal: The current filtered breathing signal.
        :param upper_threshold: Positive threshold for detecting breath transitions.
        :param lower_threshold: Negative threshold for detecting breath transitions.
        """
        self.sample_count += 1

        # Invalidate if any values are NaN
        if (
            math.isnan(upper_threshold)
            or math.isnan(lower_threshold)
            or math.isnan(breathing_signal)
        ):
            self.breathing_rate = float("nan")
            self.is_current_breath_valid = False
            return

        # Set initial state if unknown
        if self.state == UNKNOWN:
            if breathing_signal < lower_threshold:
                self.state = LOW
            elif breathing_signal > upper_threshold:
                self.state = HIGH
            else:
                self.state = MID_UNKNOWN

        # Threshold window too narrow: invalid breath
        if (upper_threshold - lower_threshold) < 2.0 * self.min_threshold:
            self.state = UNKNOWN
            self.breathing_rate = float("nan")
            self.is_current_breath_valid = False
            return
        # Threshold window too wide: invalid breath
        if (upper_threshold - lower_threshold) > 2.0 * self.max_threshold:
            self.state = UNKNOWN
            self.breathing_rate = float("nan")
            self.is_current_breath_valid = False
            return

        # State transitions based on the current signal relative to the thresholds
        if self.state == LOW and breathing_signal > lower_threshold:
            self.state = MID_RISING
        elif self.state == HIGH and breathing_signal < upper_threshold:
            self.state = MID_FALLING
        elif (self.state in (MID_RISING, MID_UNKNOWN)) and breathing_signal > upper_threshold:
            self.state = HIGH
            if self.is_inspiration_above_x:
                self._end_breath()
            else:
                # First part of the breath has ended
                self.first_part_length = self.sample_count
        elif (self.state in (MID_FALLING, MID_UNKNOWN)) and breathing_signal < lower_threshold:
            self.state = LOW
            if self.is_inspiration_above_x:
                # First part of the breath has ended
                self.first_part_length = self.sample_count
            else:
                self._end_breath()


class BreathingRateStats:
    """Compute summary statistics of recently detected breathing rates."""

    def __init__(self) -> None:
        self.fill: int = 0
        self.breathing_rates: List[float] = [0.0] * BREATHING_RATES_BUFFER_SIZE
        self.is_valid: bool = False
        self.previous_mean: float = 0.0
        self.current_mean: float = 0.0
        self.previous_variance: float = 0.0
        self.current_variance: float = 0.0
        self.max: float = float("nan")
        self.min: float = float("nan")

    def initialise(self) -> None:
        """Reset all internal statistics."""
        self.fill = 0
        self.is_valid = False

    def update(self, breathing_rate: float) -> None:
        """Add a breathing rate measurement to the buffer.

        Only the first :data:`BREATHING_RATES_BUFFER_SIZE` values are
        stored; subsequent calls after the buffer is full will discard
        new values (mirroring the C implementation).
        """
        if self.fill < BREATHING_RATES_BUFFER_SIZE:
            self.breathing_rates[self.fill] = breathing_rate
            self.fill += 1

    def calculate(self) -> None:
        """Compute the mean, variance and extrema of the collected rates.

        The values on the edges (as specified by
        :data:`DISCARD_LOWER_BREATHING_RATES` and
        :data:`DISCARD_UPPER_BREATHING_RATES`) are discarded before
        calculating the running statistics.  If too few samples are
        available the statistics remain invalid.
        """
        # Sort the collected breathing rates in ascending order
        sorted_rates = self.breathing_rates[: self.fill]
        sorted_rates.sort()

        if self.fill <= (DISCARD_LOWER_BREATHING_RATES + DISCARD_UPPER_BREATHING_RATES):
            self.is_valid = False
            return

        # Initialise running statistics with the first retained value
        start_idx = DISCARD_LOWER_BREATHING_RATES
        first_value = sorted_rates[start_idx]
        self.previous_mean = self.current_mean = first_value
        self.previous_variance = 0.0
        self.max = first_value
        self.min = first_value

        # Iterate through retained values and update running mean and variance
        for i in range(start_idx, self.fill - DISCARD_UPPER_BREATHING_RATES):
            value = sorted_rates[i]
            # Update running mean
            self.current_mean = self.previous_mean + (value - self.previous_mean) / float(self.fill)
            # Update running variance (see Knuth/Welford algorithm)
            self.current_variance = self.previous_variance + (
                value - self.previous_mean
            ) * (value - self.current_mean)

            self.previous_mean = self.current_mean
            self.previous_variance = self.current_variance

            # Update extrema
            if value > self.max:
                self.max = value
            if value < self.min:
                self.min = value

        self.is_valid = True

    def number_of_breaths(self) -> int:
        """Return the total number of stored breathing rates."""
        return self.fill

    def mean(self) -> float:
        """Return the mean breathing rate or NaN if invalid."""
        return self.current_mean if self.is_valid else float("nan")

    def variance(self) -> float:
        """Return the variance of breathing rates or NaN if invalid."""
        if not self.is_valid:
            return float("nan")
        # Variance divides by N-1 (sample variance) in the C code
        return self.current_variance / float(self.fill - 1)

    def standard_deviation(self) -> float:
        """Return the standard deviation of breathing rates or NaN if invalid."""
        var = self.variance()
        return math.sqrt(var) if not math.isnan(var) else float("nan")


###############################################################################
# Breathing measures and top‑level interface
###############################################################################

class BreathingMeasures:
    """Maintain all state required to compute the breathing signal and angle."""

    def __init__(self) -> None:
        self.signal: float = float("nan")
        self.angle: float = float("nan")
        self.is_valid: bool = False
        self.max_act_level: float = float("nan")
        self.is_breathing_initialised: bool = False
        self.activity_cutoff: float = 0.0

        # Sub‑components initialised in ``initialise``
        self.mean_unit_accel_filter: Optional[MeanUnitAccelBuffer] = None
        self.rotation_axis: Optional[RotationAxis] = None
        self.mean_rotation_axis_buffer: Optional[MeanRotationAxisBuffer] = None
        self.mean_unit_accel_buffer: Optional[MeanUnitAccelBuffer] = None
        self.mean_filter_breathing_signal: Optional[MeanPostFilter] = None
        self.mean_filter_angle: Optional[MeanPostFilter] = None
        self.activity_level_buffer: Optional[ActivityLevelBuffer] = None

    def initialise(
        self,
        pre_filter_length: int,
        post_filter_length: int,
        activity_cutoff: float,
    ) -> None:
        """Configure the buffers used to compute the breathing signal.

        :param pre_filter_length: Length of the mean filter applied to
            the raw accelerometer vectors (usually 11).
        :param post_filter_length: Length of the final smoothing filter
            applied to the breathing signal and angle (usually 12).
        :param activity_cutoff: Maximum allowed activity level; if the
            current maximum activity level exceeds this value the
            breathing signal is invalidated.
        """
        self.is_breathing_initialised = False
        self.is_valid = False
        self.signal = float("nan")
        self.angle = float("nan")
        self.max_act_level = float("nan")
        self.activity_cutoff = activity_cutoff

        # Instantiate internal buffers
        self.mean_unit_accel_filter = MeanUnitAccelBuffer(pre_filter_length)
        self.rotation_axis = RotationAxis()
        self.mean_rotation_axis_buffer = MeanRotationAxisBuffer()
        # Long mean of acceleration vectors used to compute the reference axis
        self.mean_unit_accel_buffer = MeanUnitAccelBuffer(128)
        self.mean_filter_breathing_signal = MeanPostFilter(post_filter_length)
        self.mean_filter_angle = MeanPostFilter(post_filter_length)
        self.activity_level_buffer = ActivityLevelBuffer()

        self.is_breathing_initialised = True

    def update(
        self,
        accel: FloatList,
        step_counter: StepCounter,
        activity_predictor: ActivityPredictor,
    ) -> None:
        """Update all breathing measures from a new accelerometer sample.

        This function corresponds closely to ``update_breathing_measures``
        in the C code.  It returns immediately if any prerequisite
        buffers have not yet been filled or if the subject is deemed to
        be moving too vigorously.
        """
        # The breathing signal is considered invalid until the end of this function
        self.is_valid = False
        if not self.is_breathing_initialised:
            return
        # Reject NaN values outright
        if math.isnan(accel[0]) or math.isnan(accel[1]) or math.isnan(accel[2]):
            return
        # Work on a copy of the acceleration vector
        new_accel = accel.copy()
        # Update activity level buffer and activity classification
        assert self.activity_level_buffer is not None
        self.activity_level_buffer.update(new_accel)
        activity_predictor.update(new_accel, step_counter)
        # Wait until the activity level buffer is full
        if not self.activity_level_buffer.is_valid:
            return
        # Use the maximum activity level as a threshold for movement
        self.max_act_level = self.activity_level_buffer.max
        if self.max_act_level > self.activity_cutoff:
            self.signal = float("nan")
            return
        # Discard the signal if the subject is walking
        if step_counter.is_walking():
            self.signal = float("nan")
            return
        # Smooth the acceleration vectors using the pre‑filter
        assert self.mean_unit_accel_filter is not None
        self.mean_unit_accel_filter.update(new_accel)
        if not self.mean_unit_accel_filter.is_valid:
            return
        # Replace the raw vector with the smoothed unit vector
        new_accel = self.mean_unit_accel_filter.mean_unit_vector.copy()
        # Determine the instantaneous rotation axis
        assert self.rotation_axis is not None
        self.rotation_axis.update(new_accel)
        if not self.rotation_axis.is_current_axis_valid:
            return
        current_axis = self.rotation_axis.current_axis
        # Update the long mean of acceleration vectors
        assert self.mean_unit_accel_buffer is not None
        self.mean_unit_accel_buffer.update(new_accel)
        # Update the mean rotation axis buffer
        assert self.mean_rotation_axis_buffer is not None
        self.mean_rotation_axis_buffer.update(current_axis)
        if not self.mean_rotation_axis_buffer.is_valid:
            return
        # Compute the breathing signal: projection of the instantaneous axis onto the mean axis
        mean_axis = self.mean_rotation_axis_buffer.mean_axis
        breathing_signal = dot_product(current_axis, mean_axis) * 120.0
        # Compute the breathing angle
        assert self.mean_unit_accel_buffer.mean_unit_vector is not None
        mean_accel_cross_mean_axis = cross_product(
            self.mean_unit_accel_buffer.mean_unit_vector,
            mean_axis,
        )
        breathing_angle = dot_product(mean_accel_cross_mean_axis, new_accel)
        # Smooth the breathing signal and angle with post filters
        assert self.mean_filter_breathing_signal is not None
        assert self.mean_filter_angle is not None
        self.mean_filter_breathing_signal.update(breathing_signal)
        self.mean_filter_angle.update(breathing_angle)
        # Only update the breathing measures when the post filters are valid
        if not self.mean_filter_breathing_signal.is_valid:
            return
        self.signal = self.mean_filter_breathing_signal.mean_value
        self.angle = self.mean_filter_angle.mean_value
        self.is_valid = True


class RespeckBreathing:
    """Top‑level class exposing a Python interface to the RESpeck algorithm.

    This class aggregates all internal buffers and state machines
    necessary to compute breathing metrics from accelerometer data.
    Its public methods mirror the C functions exported by the original
    shared library.  Create one instance of this class per sensor
    stream and call :meth:`initBreathing` exactly once before
    streaming data via :meth:`updateBreathing`.
    """

    def __init__(self) -> None:
        # Main algorithm components
        self.breathing_measures: BreathingMeasures = BreathingMeasures()
        self.threshold_buffer: Optional[ThresholdBuffer] = None
        self.current_breath: Optional[CurrentBreath] = None
        self.step_counter: StepCounter = StepCounter()
        self.breathing_rate_stats: BreathingRateStats = BreathingRateStats()
        self.activity_predictor: ActivityPredictor = ActivityPredictor()
        # Dynamic threshold factor
        self.th_factor: float = 1.0
        self.upper_threshold: float = float("nan")
        self.lower_threshold: float = float("nan")
        # Flag indicating whether a breath has just ended
        self.is_breath_end: bool = False

    def initBreathing(
        self,
        pre_filter_length: int,
        post_filter_length: int,
        activity_cutoff: float,
        threshold_filter_size: int,
        lower_threshold_limit: float,
        upper_threshold_limit: float,
        threshold_factor: float,
        sampling_frequency: float,
    ) -> None:
        """Initialise or reinitialise all internal buffers and state.

        This method must be called before any calls to
        :meth:`updateBreathing` or :meth:`updateBreathingSignal`.  It
        mirrors the signature of the C function ``initBreathing``.

        :param pre_filter_length: Length of the mean filter used to
            smooth raw acceleration vectors.
        :param post_filter_length: Length of the final smoothing filter
            applied to the breathing signal and angle.
        :param activity_cutoff: Movement threshold above which breathing
            updates are suppressed.
        :param threshold_filter_size: Window length used to compute
            running RMS thresholds for breath detection.
        :param lower_threshold_limit: Minimum allowed threshold
            amplitude (in arbitrary units).
        :param upper_threshold_limit: Maximum allowed threshold
            amplitude (in arbitrary units).
        :param threshold_factor: Scaling factor applied to the RMS
            thresholds when comparing against the breathing signal.
        :param sampling_frequency: Sampling frequency of the incoming
            accelerometer data in Hz.
        """
        # Initialise the breathing signal computation
        self.breathing_measures.initialise(pre_filter_length, post_filter_length, activity_cutoff)
        # Initialise the RMS threshold buffer
        self.threshold_buffer = ThresholdBuffer(threshold_filter_size)
        # Initialise the breath state machine
        self.current_breath = CurrentBreath(lower_threshold_limit, upper_threshold_limit, sampling_frequency)
        # Reset breathing rate statistics
        self.breathing_rate_stats.initialise()
        # Reset step counter and activity predictor
        self.step_counter = StepCounter()
        self.activity_predictor.initialise()
        # Store the threshold factor
        self.th_factor = threshold_factor
        # Reset dynamic thresholds and breath end flag
        self.upper_threshold = float("nan")
        self.lower_threshold = float("nan")
        self.is_breath_end = False

    def updateBreathing(self, x: float, y: float, z: float) -> None:
        """Update the algorithm with a new accelerometer sample.

        :param x: X‑axis acceleration in g (or arbitrary units).
        :param y: Y‑axis acceleration in g (or arbitrary units).
        :param z: Z‑axis acceleration in g (or arbitrary units).
        """
        if self.current_breath is None or self.threshold_buffer is None:
            raise RuntimeError("initBreathing() must be called before updateBreathing().")
        accel = [float(x), float(y), float(z)]
        # Update step counter
        self.step_counter.update(accel)
        # Update breathing measures (signal and angle)
        self.breathing_measures.update(accel, self.step_counter, self.activity_predictor)
        # Update RMS thresholds with the latest breathing signal
        self.threshold_buffer.update(self.breathing_measures.signal)
        # Scale the RMS values by the configured threshold factor
        if math.isnan(self.threshold_buffer.upper_threshold_value):
            self.upper_threshold = float("nan")
        else:
            self.upper_threshold = self.threshold_buffer.upper_threshold_value / self.th_factor
        if math.isnan(self.threshold_buffer.lower_threshold_value):
            self.lower_threshold = float("nan")
        else:
            self.lower_threshold = self.threshold_buffer.lower_threshold_value / self.th_factor
        # Update the breath detection state machine
        self.current_breath.update(
            self.breathing_measures.signal,
            self.upper_threshold,
            self.lower_threshold,
        )
        # If a complete breath has been detected, store its rate
        if self.current_breath.is_complete and not math.isnan(self.current_breath.breathing_rate):
            self.breathing_rate_stats.update(self.current_breath.breathing_rate)
            self.current_breath.is_complete = False
            self.is_breath_end = True
        else:
            self.is_breath_end = False

    def updateBreathingSignal(self, breathing_signal: float) -> None:
        """Update the algorithm with an externally provided breathing signal.

        This method skips the accelerometer processing and should be
        used when you have a direct measurement of the breathing
        waveform (for example from a respiration belt or nasal
        cannula).  The dynamic thresholds and breathing rate detection
        logic are identical to :meth:`updateBreathing`.
        """
        if self.current_breath is None or self.threshold_buffer is None:
            raise RuntimeError("initBreathing() must be called before updateBreathingSignal().")
        # Update threshold buffer
        self.threshold_buffer.update(float(breathing_signal))
        # Scale thresholds
        if math.isnan(self.threshold_buffer.upper_threshold_value):
            self.upper_threshold = float("nan")
        else:
            self.upper_threshold = self.threshold_buffer.upper_threshold_value / self.th_factor
        if math.isnan(self.threshold_buffer.lower_threshold_value):
            self.lower_threshold = float("nan")
        else:
            self.lower_threshold = self.threshold_buffer.lower_threshold_value / self.th_factor
        # Update breath state machine
        self.current_breath.update(breathing_signal, self.upper_threshold, self.lower_threshold)
        if self.current_breath.is_complete and not math.isnan(self.current_breath.breathing_rate):
            self.breathing_rate_stats.update(self.current_breath.breathing_rate)
            self.current_breath.is_complete = False
            self.is_breath_end = True
        else:
            self.is_breath_end = False

    # Getter methods mirroring the C API

    def getBreathingSignal(self) -> float:
        """Return the latest smoothed breathing signal (may be NaN)."""
        return self.breathing_measures.signal

    def getBreathingRate(self) -> float:
        """Return the breathing rate of the most recently detected breath.

        If no complete breath has been detected yet this returns NaN.
        After reading the breathing rate you may wish to call
        :meth:`resetBreathingRate` to distinguish subsequent breaths.
        """
        return self.current_breath.breathing_rate if self.current_breath is not None else float("nan")

    def getUpperThreshold(self) -> float:
        """Return the current scaled upper threshold value."""
        return self.upper_threshold

    def getLowerThreshold(self) -> float:
        """Return the current scaled lower threshold value."""
        return self.lower_threshold

    def getBreathingAngle(self) -> float:
        """Return the latest smoothed breathing angle (may be NaN)."""
        return self.breathing_measures.angle

    def getMinuteStepcount(self) -> int:
        """Return the cumulative number of steps detected in the current minute."""
        return self.step_counter.minute_step_count

    def resetMinuteStepcount(self) -> None:
        """Reset the minute step count to zero."""
        self.step_counter.minute_step_count = 0

    def getActivityLevel(self) -> float:
        """Return the maximum activity level observed over the last window."""
        return self.breathing_measures.max_act_level

    def getActivityClassification(self) -> int:
        """Return the most recent activity classification code."""
        return self.activity_predictor.last_prediction

    def getAverageBreathingRate(self) -> float:
        """Return the mean of stored breathing rates (NaN if insufficient data)."""
        return self.breathing_rate_stats.mean()

    def getStdDevBreathingRate(self) -> float:
        """Return the standard deviation of stored breathing rates (NaN if insufficient data)."""
        return self.breathing_rate_stats.standard_deviation()

    def getNumberOfBreaths(self) -> int:
        """Return the number of breathing rates collected so far."""
        return self.breathing_rate_stats.number_of_breaths()

    def resetBreathingRate(self) -> None:
        """Reset the last breathing rate to NaN.

        This is useful when you wish to be notified of the next breath
        separately: after calling :meth:`getBreathingRate` you can call
        ``resetBreathingRate()`` so that two successive breaths with
        identical rates are not mistaken for a single breath.
        """
        if self.current_breath is not None:
            self.current_breath.breathing_rate = float("nan")

    def resetMedianAverageBreathing(self) -> None:
        """Clear all stored breathing rate statistics."""
        self.breathing_rate_stats.initialise()

    def calculateAverageBreathing(self) -> None:
        """Compute the mean, variance and standard deviation of collected rates."""
        self.breathing_rate_stats.calculate()

    def getIsBreathEnd(self) -> bool:
        """Return True if a breath has just ended in the last update call."""
        return self.is_breath_end
