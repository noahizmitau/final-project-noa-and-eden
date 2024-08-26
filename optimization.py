import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import grad
from autocorrelation_analysis import (
    first_order_autocorrelation, second_order_autocorrelation, third_order_autocorrelation
)
from synthetic_measurement import signal, observed_data, SINGLE_SIGNAL_LENGTH, OBSERVED_DATA_LENGTH, K
import time

start_time = time.time()

# Constants
L = SINGLE_SIGNAL_LENGTH
N = OBSERVED_DATA_LENGTH
s = K
n2 = L - 1
n3 = (L - 1) * (L - 2) / 2
w1, w2, w3 = 1/2, 1/(2*n2), 1/(2*n3)
gamma_hat = (s * L) / N


# Pre-compute autocorrelations
a_y1 = first_order_autocorrelation(observed_data)
autocorrelation_y2 = np.array([second_order_autocorrelation(observed_data, l1, N) for l1 in range(1, L)])
autocorrelation_y3 = np.array([
    third_order_autocorrelation(observed_data, l1, l2, N)
    for l1 in range(2, L) for l2 in range(1, l1)
])

def objective(x):
    a_x1 = first_order_autocorrelation(x)
    term1 = w1 * (a_y1 - gamma_hat * a_x1) ** 2

    a_x2 = np.array([second_order_autocorrelation(x, l1, L) for l1 in range(1, L)])
    term2 = w2 * np.sum((autocorrelation_y2 - gamma_hat * a_x2) ** 2)

    a_x3 = np.array([
        third_order_autocorrelation(x, l1, l2, L)
        for l1 in range(2, L) for l2 in range(1, l1)
    ])
    term3 = w3 * np.sum((autocorrelation_y3 - gamma_hat * a_x3) ** 2)

    return term1 + term2 + term3

gradient = grad(objective)

# Initial guess for x
initial_guess = np.random.randn(L, 1)
# initial_guess = np.array([1.0] * L)

result = minimize(objective, initial_guess, method='BFGS', jac=gradient, options={'gtol': 1e-12, 'maxiter': 10000})

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("\n")
print("Optimization result:", result)

print("\n")
print("Original signal:", signal)

plt.figure(figsize=(10, 6))
plt.plot(result.x)
plt.title('Reconstructed signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-1, SINGLE_SIGNAL_LENGTH)
plt.ylim(-1, 1)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(signal)
plt.title('Original signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-1, SINGLE_SIGNAL_LENGTH)
plt.ylim(-1, 1)
plt.show()

print(objective(initial_guess))

### error calculations ###

numerator = np.linalg.norm(signal - result.x)
# Calculate the second order norm (Euclidean norm) of X
denominator = np.linalg.norm(signal)
# Calculate the estimated error
estimated_error = numerator / denominator
print("Estimated Error:", estimated_error)
print("observed data length:", N)

