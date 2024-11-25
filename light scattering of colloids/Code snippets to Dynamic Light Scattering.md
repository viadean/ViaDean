# Code snippets to Dynamic Light Scattering

Dynamic Light Scattering (DLS) is an analytical technique used to measure the size distribution of small particles or molecules in suspension or solution. It works by analyzing fluctuations in the intensity of scattered light caused by the Brownian motion of particles. These fluctuations provide data on particle size by correlating them to diffusion coefficients using the Stokes-Einstein equation. DLS is widely used in fields like nanotechnology, pharmaceuticals, and biology due to its precision in determining particle size, polydispersity, and stability in colloidal systems. It is valued for its non-invasive approach, requiring minimal sample preparation while delivering rapid and accurate measurements.

### :cactus:MATLAB snippet

Dynamic Light Scattering (DLS) is a technique used to measure the size distribution of small particles in suspension by analyzing the fluctuations in light intensity caused by Brownian motion. MATLAB is well-suited for this type of analysis, as it offers extensive mathematical and visualization capabilities.

### Basic Overview of DLS:

1. **Concept**: DLS measures the diffusion of particles in a fluid by observing the time-dependent intensity of scattered light.

2. **Data Analysis**: The time correlation function of the intensity signal is used to extract the diffusion coefficient, which can then be related to particle size using the Stokes-Einstein equation.

3. **Mathematical Background**:

   - **Correlation Function** \( $g_2(\tau)$ \): Used to describe how the intensity changes over time.

   - **Diffusion Coefficient** \( $D$ \): Extracted from the correlation function.

   - **Hydrodynamic Radius** \( $R_H$ \): Calculated using the Stokes-Einstein equation:
     $$
     R_H = \frac{k_B T}{6 \pi \eta D}
     $$

     where \( $k_B$ \) is Boltzmann’s constant, \( $T$ \) is temperature, \( $\eta$ \) is the viscosity of the fluid, and \( $D$ \) is the diffusion coefficient.

### Implementing DLS Analysis in MATLAB:

Here's a simplified approach to analyzing DLS data using MATLAB.

#### Step 1: Import Intensity Data

Ensure you have time-series data representing the intensity of the scattered light. This data is typically obtained from a DLS instrument.

```matlab
% Load the intensity data (assumed to be in a .mat or .csv file)
data = load('intensity_data.mat'); % Replace with your file path or import function
intensity = data.intensity; % Adjust based on your file's structure
time = data.time; % Time vector corresponding to intensity data
```

#### Step 2: Calculate the Correlation Function

The correlation function \( $g_2(\tau)$ \) can be estimated using MATLAB's built-in functions.

```matlab
% Compute the correlation function using autocorrelation
normalized_correlation = xcorr(intensity, 'unbiased');
normalized_correlation = normalized_correlation(normalized_correlation >= 0); % Take only non-negative lags

% Normalize the correlation function
g2_tau = normalized_correlation / max(normalized_correlation);

% Plot the correlation function
figure;
plot(time(1:length(g2_tau)), g2_tau);
xlabel('Time Delay (s)');
ylabel('Correlation Function g_2(\tau)');
title('Intensity Correlation Function');
```

#### Step 3: Extract the Diffusion Coefficient

Fit an exponential decay model to the correlation function to find the diffusion coefficient \( $D$ \).

```matlab
% Fit an exponential decay to estimate D (simplified approach)
fit_func = @(b, x) exp(-2 * b * x); % Simple model for fitting
initial_guess = 1e-4; % Adjust based on expected D
params = lsqcurvefit(fit_func, initial_guess, time(1:length(g2_tau)), g2_tau);

D = params; % Extracted diffusion coefficient
disp(['Estimated Diffusion Coefficient: ', num2str(D)]);
```

#### Step 4: Calculate the Hydrodynamic Radius

Use the Stokes-Einstein equation to calculate the hydrodynamic radius \( $R_H$ \).

```matlab
% Constants
kB = 1.380649e-23; % Boltzmann constant (J/K)
T = 298; % Temperature in Kelvin (e.g., room temperature ~25°C)
eta = 0.001; % Viscosity of water in Pa·s (at ~25°C)

% Calculate the hydrodynamic radius
R_H = (kB * T) / (6 * pi * eta * D);
disp(['Hydrodynamic Radius (nm): ', num2str(R_H * 1e9)]); % Convert to nanometers
```

### Visualization

To better understand the data, visualize the correlation function and the exponential decay fit:

```matlab
% Plot the correlation function with the fitted curve
figure;
plot(time(1:length(g2_tau)), g2_tau, 'b-', 'DisplayName', 'Correlation Function');
hold on;
plot(time(1:length(g2_tau)), fit_func(D, time(1:length(g2_tau))), 'r--', 'DisplayName', 'Exponential Fit');
xlabel('Time Delay (s)');
ylabel('Correlation Function g_2(\tau)');
legend('show');
title('DLS Data Analysis');
```

### Enhancements:

- **Noise Filtering**: Preprocess the data to remove noise before calculating the correlation function.
- **Advanced Fitting**: Use non-linear fitting functions or advanced fitting algorithms to improve the accuracy of \( $D$ \).
- **Temperature and Viscosity**: Input exact experimental values for \( $T$ \) and \( $\eta$ \) for precise results.

### Final Notes:

This MATLAB code provides a starting point for analyzing DLS data. For more advanced or commercial applications, consider using DLS analysis software or more comprehensive MATLAB toolboxes that handle complex fitting and correlation calculations.

### :cactus:Python snippet

Dynamic Light Scattering (DLS) is a powerful technique used to measure the size distribution of small particles in suspension or polymers in solution. The basic principle relies on the analysis of the time-dependent fluctuations of light scattered by particles undergoing Brownian motion.

Here's an overview and a basic implementation of how to simulate and process DLS data using Python.

### How Dynamic Light Scattering Works:

- **Brownian Motion**: Particles in a suspension move randomly due to thermal energy.

- **Scattered Light**: A laser beam directed at the suspension will scatter light, which fluctuates in intensity due to the movement of particles.

- **Correlation Function**: The time-dependent fluctuations are analyzed using an autocorrelation function to determine the diffusion coefficient \( $D$ \).

- **Particle Size**: The diffusion coefficient is related to the particle size by the Stokes-Einstein equation:
  $$
  D = \frac{k_B T}{6 \pi \eta R}
  where:
  $$

  - \( $k_B$ \) is the Boltzmann constant,
  - \( $T$ \) is the temperature,
  - \($\eta$ \) is the viscosity of the medium,
  - \( $R$ \) is the hydrodynamic radius of the particle.

### Python Implementation:

You can use libraries like `numpy` and `scipy` to generate synthetic DLS data and analyze it.

#### Step-by-Step Code:

1. Generate synthetic intensity data based on Brownian motion.
2. Calculate the autocorrelation function.
3. Estimate the diffusion coefficient and particle size.

Here's a simple example:

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Generate synthetic DLS data
def generate_synthetic_data(num_points, diffusion_coefficient, noise_level=0.01):
    time = np.linspace(0, 1, num_points)
    g2 = np.exp(-2 * diffusion_coefficient * time) + noise_level * np.random.normal(size=num_points)
    return time, g2

# Fit function for the autocorrelation
def fit_func(t, D):
    return np.exp(-2 * D * t)

# Generate data
num_points = 1000
diffusion_coefficient_true = 0.5
time, g2 = generate_synthetic_data(num_points, diffusion_coefficient_true)

# Fit the data
popt, _ = curve_fit(fit_func, time, g2)
diffusion_coefficient_estimated = popt[0]

# Plot the results
plt.plot(time, g2, label='Synthetic Data')
plt.plot(time, fit_func(time, diffusion_coefficient_estimated), label=f'Fit (D={diffusion_coefficient_estimated:.3f})')
plt.xlabel('Time')
plt.ylabel('g2(t)')
plt.title('DLS Autocorrelation Function')
plt.legend()
plt.show()

# Calculate particle size using the Stokes-Einstein equation
T = 298.15  # Temperature in Kelvin (25°C)
k_B = 1.380649e-23  # Boltzmann constant in J/K
eta = 0.001  # Viscosity of water in Pa.s
R = k_B * T / (6 * np.pi * eta * diffusion_coefficient_estimated)

print(f"Estimated particle radius: {R:.2e} m")
```

### Explanation:

- **`generate_synthetic_data`**: Generates synthetic time-dependent intensity data using an exponential decay model.
- **`fit_func`**: Model function for fitting the autocorrelation data.
- **`curve_fit`**: Used to fit the exponential model to the synthetic data and estimate the diffusion coefficient.
- **Particle Size Calculation**: The Stokes-Einstein equation is used to compute the particle radius.

### Output:

- **Plot**: A plot of the synthetic autocorrelation function \( $g_2(t)$ \) and the fitted curve.
- **Estimated Radius**: The calculated particle radius in meters based on the estimated diffusion coefficient.

### Enhancements:

- **Noise Reduction**: Apply filtering techniques to smooth real experimental data.
- **Advanced Models**: Implement multi-exponential or cumulant analysis for more complex data.
- **Real Data Analysis**: Use experimental DLS data files and parse them for analysis.

### Python Libraries to Consider:

- **`numpy`**: For numerical computations.
- **`scipy`**: For fitting models and handling advanced mathematical operations.
- **`matplotlib`**: For plotting data.
- **`pandas`**: For handling data input/output if you are reading experimental data.

This code provides a foundational approach to understanding and implementing DLS analysis in Python.

### :cactus:R snippet

Dynamic Light Scattering (DLS) is a technique used to measure the size distribution of small particles in suspension by analyzing the fluctuations in the intensity of scattered light. It provides insights into the hydrodynamic radius of particles through the analysis of Brownian motion.

Implementing DLS analysis in R involves:

1. Simulating the scattering data.
2. Applying statistical and mathematical tools to estimate particle size distribution.

### Concept Overview

- **Intensity Fluctuations**: DLS measures how light scattering intensity varies over time as particles move.

- **Correlation Function**: An autocorrelation function \( $G(\tau)$ \) is computed, where \( $\tau$ \) is the lag time. This function characterizes how the intensity of scattered light correlates with itself at different time intervals.

- **Hydrodynamic Radius**: The Stokes-Einstein equation is used to calculate the hydrodynamic radius \( $R_H$ \):
  $$
  R_H = \frac{k_B T}{6 \pi \eta D}
  where:
  $$

  - \( $k_B$ \) is Boltzmann's constant.
  - \( $T$ \) is the absolute temperature.
  - \( $\eta$ \) is the viscosity of the solvent.
  - \( $D$ \) is the diffusion coefficient.

### Simulating DLS Data in R

Here's a simplified example in R to simulate DLS data and calculate the hydrodynamic radius:

```r
# Load necessary libraries
library(ggplot2)

# Parameters for the simulation
set.seed(42)
temperature <- 298.15 # Temperature in Kelvin (25°C)
viscosity <- 0.001 # Viscosity of water in Pa·s (N·s/m²)
boltzmann_constant <- 1.380649e-23 # Boltzmann constant in J/K

# Simulate time points (in microseconds)
time_lags <- seq(1e-6, 1e-3, length.out = 100)

# Simulate the autocorrelation function for particles with a known size
true_radius <- 50e-9 # 50 nm in meters
diffusion_coefficient <- boltzmann_constant * temperature / (6 * pi * viscosity * true_radius)

# Simulate an exponential decay for the autocorrelation function
autocorrelation_function <- exp(-2 * diffusion_coefficient * time_lags)

# Plot the autocorrelation function
ggplot(data = data.frame(Time = time_lags * 1e6, Correlation = autocorrelation_function), 
       aes(x = Time, y = Correlation)) +
  geom_line(color = "blue") +
  labs(title = "Simulated Autocorrelation Function",
       x = "Time Lag (microseconds)",
       y = "Autocorrelation") +
  theme_minimal()

# Calculate the hydrodynamic radius from the diffusion coefficient
calculated_radius <- boltzmann_constant * temperature / (6 * pi * viscosity * diffusion_coefficient)
cat("Calculated hydrodynamic radius (in nm):", calculated_radius * 1e9, "\n")
```

### Explanation:

- **Autocorrelation Function Simulation**: The exponential decay simulates how intensity fluctuations decrease over time.
- **Diffusion Coefficient**: Directly linked to particle size through the Stokes-Einstein equation.
- **Plot**: The autocorrelation function is visualized to show how it decays with increasing time lag.

### Enhancements:

1. **Experimental Data**: Use real DLS data by importing it with functions like `read.csv()` or `read.table()`.
2. **Fitting and Analysis**: Implement non-linear curve fitting using `nls()` to estimate parameters from real data.
3. **Noise Addition**: Add noise to simulate realistic conditions using `rnorm()` for more complex analysis.
4. **Multi-Modal Distribution**: Extend the model to analyze distributions with multiple particle sizes.

### Further Analysis:

- Use the **Correlation Spectroscopy** package for more advanced analyses.
- Apply **Bayesian inference** or **Maximum Likelihood Estimation (MLE)** to fit models to experimental data for precise size distribution analysis.

### :cactus:Julia snippet

Dynamic Light Scattering (DLS), also known as Photon Correlation Spectroscopy or Quasi-Elastic Light Scattering, is a technique used to determine the size distribution of small particles in suspension or polymers in solution by analyzing the time-dependent fluctuations in light scattering.

Here's a guide and example code for implementing a basic model of DLS in Julia.

### Key Concepts of Dynamic Light Scattering:

1. **Brownian Motion**: Particles undergo random motion, and the rate of this motion depends on their size.

2. **Correlation Function**: The time-dependent fluctuations in the intensity of scattered light are analyzed to obtain an autocorrelation function, which helps in determining the particle size distribution.

3. **Stokes-Einstein Equation**: Used to relate the diffusion coefficient to the particle size:
   $$
   D = \frac{k_B T}{6 \pi \eta r}
   $$
   where \( $D$ \) is the diffusion coefficient, \( $k_B$ \) is Boltzmann's constant, \( $T$ \) is temperature, \( $\eta$ \) is the viscosity of the medium, and \( $r$ \) is the particle radius.

### Basic DLS Simulation in Julia

Below is an example of how you could simulate and analyze DLS data using Julia:

```julia
using LinearAlgebra, Plots

# Constants
k_B = 1.380649e-23 # Boltzmann constant (J/K)
T = 298.15 # Temperature (Kelvin, ~25°C)
η = 0.001 # Viscosity of water at ~25°C (Pa·s)

# Function to calculate the diffusion coefficient
function diffusion_coefficient(radius)
    return k_B * T / (6 * π * η * radius)
end

# Function to generate synthetic DLS data (autocorrelation function)
function generate_autocorrelation_function(radius, times)
    D = diffusion_coefficient(radius)
    return exp.(-2 * D * times)
end

# Simulation parameters
particle_radius = 100e-9 # Particle radius in meters (100 nm)
times = LinRange(0, 1e-3, 1000) # Time values in seconds for autocorrelation

# Generate synthetic data
autocorrelation = generate_autocorrelation_function(particle_radius, times)

# Plot the autocorrelation function
plot(times, autocorrelation, xlabel="Time (s)", ylabel="Autocorrelation", 
     title="DLS Autocorrelation Function", lw=2, legend=false)
```

### Explanation:

- **Diffusion Coefficient Calculation**: The `diffusion_coefficient` function calculates \( $D$ \) using the Stokes-Einstein equation.
- **Autocorrelation Function**: The `generate_autocorrelation_function` simulates the exponential decay typically observed in DLS measurements. The decay rate is related to the diffusion coefficient, which depends on the particle size.
- **Plot**: The plot visualizes how the autocorrelation decays over time.

### Steps to Run:

1. Ensure Julia is installed on your system.

2. Install required packages:

   ```julia
   using Pkg
   Pkg.add("Plots")
   ```

3. Run the code in a Julia environment, such as the Julia REPL or Jupyter Notebook.

### Further Enhancements:

1. **Noise Simulation**: Add random noise to simulate real experimental data.
2. **Inverse Laplace Transform**: Use numerical methods to extract the particle size distribution from the autocorrelation function.
3. **Multi-Size Distributions**: Simulate systems with particles of different sizes and see how it affects the autocorrelation.
4. **Advanced Libraries**: Use Julia packages like `LsqFit` or `Distributions` for fitting data and analyzing size distributions.

### Advanced Analysis:

- **Contin Algorithm**: Implement or use pre-existing code for the CONTIN algorithm, which helps deconvolve the autocorrelation function into particle size distributions.
- **Time-Correlation Functions**: Analyze experimental DLS data with higher-order time-correlation analysis to study complex systems or polydisperse samples.

This basic framework provides a starting point for simulating and visualizing DLS data with Julia, useful for learning and educational purposes.
