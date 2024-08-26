import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import sys


###################
##### Defines #####
###################

NOISE_LEVEL = 0.2
SINGLE_SIGNAL_LENGTH = 11        # length of each x
K = 10                          # number of signals
OBSERVED_DATA_LENGTH = 2000      # length of y
SEED_VALUE = 91                # seed value to ensure we get the same random numbers every run

#####################
##### Functions #####
#####################

# Validation Error Check
if (K * SINGLE_SIGNAL_LENGTH * 2) - SINGLE_SIGNAL_LENGTH > OBSERVED_DATA_LENGTH:
    print("Invalid input: It's impossible to have non-overlapping signals.")
    sys.exit(1)

np.random.seed(SEED_VALUE)

def generate_signal(length):
    # Generate a triangle signal using scipy's sawtooth function
    # a time array covering one period of the waveform, with values ranging from 0 to 2π radians
    # 0.5 for DC = 50%
    return sawtooth(np.linspace(0, 1, length) * 2 * np.pi, width=0.5)

def generate_data(signals, noise_level):
    # Initialize observed data
    observed_data = np.zeros(OBSERVED_DATA_LENGTH)

    # Generate random non-overlapping locations for signals
    used_indices = set()
    for signal in signals:
        # Randomly select an index that doesn't overlap with existing signals
        # and ensure that there is at least one signal length space between pairs
        found = False
        while (not found):
            index = np.random.randint(0, OBSERVED_DATA_LENGTH - SINGLE_SIGNAL_LENGTH + 1)
            # Check the distance to the nearest existing signal on both sides
            min_distance_left = all(
                abs(index - existing_index) >= (2 * SINGLE_SIGNAL_LENGTH) + 1 for existing_index in used_indices if
                existing_index < index
            )
            min_distance_right = all(
                abs(index - existing_index) >= (2 * SINGLE_SIGNAL_LENGTH) + 1 for existing_index in used_indices if
                existing_index > index
            )
            found = min_distance_left and min_distance_right

        used_indices.update(range(index, index + SINGLE_SIGNAL_LENGTH))
        observed_data[index:index + SINGLE_SIGNAL_LENGTH] += signal

    # Add Gaussian noise to the observed data
    noise = np.random.normal(scale=noise_level, size=len(observed_data))
    observed_data += noise

    return observed_data


# Generate triangle signals
signals = [generate_signal(SINGLE_SIGNAL_LENGTH) for i in range(K)]

# Generate observed data
observed_data = generate_data(signals, NOISE_LEVEL)

# Print of the observed data
print("Generated Signals:")
for i, signal in enumerate(signals):
    print(f"Signal {i+1}: {signal}")

print("\nObserved Data:", observed_data)

# Plot the observed data
plt.figure(figsize=(10, 6))
plt.plot(observed_data)
plt.title('Observed Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0, OBSERVED_DATA_LENGTH)
plt.ylim(-10, 10)
plt.show()


