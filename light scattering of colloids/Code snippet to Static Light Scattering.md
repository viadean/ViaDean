# Code snippet to Static Light Scattering

Static Light Scattering (SLS) is an analytical technique used to measure the molecular weight, size, and interactions of particles in a solution. By detecting the intensity of light scattered at different angles, SLS provides valuable data on the average molecular weight and radius of gyration without the need for calibration standards. This method is particularly useful for characterizing polymers, proteins, and colloids in research and industry. SLS is non-destructive and works by analyzing the angular distribution of scattered light, revealing insights into particle structure and behavior. Its precision makes it a powerful tool in material science and biophysical studies.

### :cactus:MATLAB snippet

Static Light Scattering (SLS) is a technique used to determine the size and shape of particles in a sample based on how they scatter light at various angles. Unlike Dynamic Light Scattering (DLS), SLS measures the intensity of scattered light at different angles but does not focus on time-dependent fluctuations, making it more suitable for analyzing larger or non-Brownian particles.

The intensity of scattered light in SLS can be related to the size of the particles through the **Rayleigh-Debye scattering equation** for spherical particles, or more generally, through models like **Mie Theory**.

Here’s a basic guide to simulating static light scattering in MATLAB, focusing on spherical particles using a simplified Rayleigh or Mie scattering approximation.

### Key Concepts of Static Light Scattering:

1. **Scattering Intensity**: The scattering intensity depends on the particle size, the wavelength of light, and the scattering angle.
2. **Rayleigh Scattering**: For particles much smaller than the wavelength of light, the scattering intensity is inversely proportional to the fourth power of the wavelength.
3. **Mie Theory**: For particles comparable in size to the wavelength of light, Mie theory is used to compute the scattering cross-section and scattering angle dependence.

### Example MATLAB Code for Mie Scattering:

The following MATLAB code demonstrates static light scattering for spherical particles using Mie theory. We'll use the **MieScattering** function from the `MiePlot` MATLAB package, or an approximation for small particles (Rayleigh scattering).

#### Step 1: Install the Mie Theory Function

You can use a Mie scattering function available in the MATLAB File Exchange (e.g., `mie.m`), or you can implement a basic scattering model.

#### Step 2: MATLAB Code Example for Static Light Scattering:

```matlab
% Static Light Scattering Simulation in MATLAB

% Constants
wavelength = 650e-9;  % Wavelength of light (650 nm)
radius = 100e-9;     % Particle radius (100 nm)
angle_range = linspace(0, pi, 180);  % Scattering angle range from 0 to 180 degrees

% Refractive index of the particle (e.g., silica)
n_particle = 1.45;

% Refractive index of the medium (e.g., water)
n_medium = 1.33;

% Size parameter (dimensionless)
size_parameter = (2 * pi * radius) / wavelength;

% Calculate the scattering intensity (simplified model for Rayleigh scattering)
% The Rayleigh scattering intensity is proportional to (sin(θ))^2 / λ^4 for small particles
intensity_rayleigh = (sin(angle_range).^2) / wavelength^4;

% Plot the scattering intensity as a function of angle
figure;
polarplot(angle_range, intensity_rayleigh, 'LineWidth', 2);
title('Rayleigh Scattering Pattern');
ax = gca;
ax.ThetaTickLabel = {'0', '30', '60', '90', '120', '150', '180'};
ax.ThetaTick = 0:30:180;
ax.RLim = [0, max(intensity_rayleigh)*1.2];

% If using Mie scattering, you can implement Mie theory functions or
% use the MiePlot package to calculate the intensity for a more complex
% particle size and refractive index dependency.
```

### Explanation:

- **Wavelength and Radius**: We define the wavelength of the light and the particle size. The `wavelength` is typically in the visible spectrum, while the `radius` is set for a nanoparticle.
- **Size Parameter**: The size parameter \( $x = \frac{2\pi r}{\lambda}$ \) is used to calculate the scattering behavior.
- **Rayleigh Approximation**: The formula used here for Rayleigh scattering is simplified for small particles. The scattering intensity depends on the angle \( $\theta$ \) and wavelength \( $\lambda$ \).
- **Polar Plot**: The `polarplot` function in MATLAB is used to plot the scattering intensity as a function of angle.

### Step 3: Running the Code

1. Copy the MATLAB code above into a new script, say `StaticLightScattering.m`.
2. Run the script in MATLAB.

#### Optional: For More Complex Mie Scattering

If you need more precise Mie scattering calculations, you can use pre-existing Mie theory functions or packages like `MiePlot`, which implements the full Mie theory:

1. Download and add the `MiePlot` or other Mie scattering functions from the MATLAB File Exchange.
2. Replace the Rayleigh scattering part with Mie scattering functions.

Here’s a simple example of using a `mie.m` function that can be found on MATLAB File Exchange:

```matlab
% Using Mie theory for static light scattering
% mie.m function needs to be added from the MATLAB File Exchange

% Call mie function for scattering at different angles
[m, x, q_ext, q_sca, q_back] = mie(wavelength, radius, n_particle, n_medium);

% q_sca is the scattering efficiency; we can plot it vs scattering angle
angles = linspace(0, pi, 180); % Angle in radians
scattering_intensity_mie = q_sca .* sin(angles).^2;

% Plot the Mie scattering intensity as a function of angle
figure;
polarplot(angles, scattering_intensity_mie, 'LineWidth', 2);
title('Mie Scattering Pattern');
```

### Additional Notes:

- **Mie Theory**: Mie scattering involves solving a set of equations that relate the scattering coefficients to the particle size, refractive index, and the wavelength of light. The `mie.m` function typically provides scattering efficiencies such as \( $q_{\text{ext}}$ \) (extinction efficiency), \( $q_{\text{sca}}$ \) (scattering efficiency), and \( $q_{\text{back}}$ \) (backscatter efficiency).
- **Rayleigh vs Mie**: Rayleigh scattering is typically applicable for very small particles (\($ r \ll \lambda$ \)), whereas Mie theory is more suitable for particles comparable in size to the wavelength.

### Conclusion:

This example provides a simple simulation of static light scattering in MATLAB. The Rayleigh approximation is used for small particles, but you can extend this code to use Mie theory for more complex scattering behavior, especially for larger particles. The plot provides insight into how scattering varies with the angle for different particle sizes, helping you understand the scattering profile of your sample.

### :cactus:Python snippet

Static Light Scattering (SLS) is a technique used to study the size, shape, and structure of particles or macromolecules in solution by analyzing how they scatter light. Unlike Dynamic Light Scattering (DLS), which focuses on time-dependent fluctuations due to Brownian motion, Static Light Scattering examines the intensity of scattered light at different angles to extract information about the particle's size and distribution.

In this example, we’ll implement a basic simulation of Static Light Scattering in Python using the Rayleigh-Debye equation. This equation gives the relationship between the scattered light intensity and the size of the particles.

### Key Concepts of Static Light Scattering:

1. **Rayleigh-Debye Scattering**: For small particles (where the particle size is much smaller than the wavelength of light), the intensity \( I(\theta) \) of scattered light is related to the scattering angle \( \theta \) and the particle’s properties.

   The Rayleigh-Debye formula is:
   $$
   I(\theta) \propto \left[ \frac{\sin(\theta)}{\theta} \right]^2
   $$

   where:

   - \($ \theta$ \) is the scattering angle,
   - The intensity \($ I(\theta)$ \) depends on the particle size, concentration, and the optical properties of the medium.

2. **Intensity vs. Angle**: Static Light Scattering typically measures the intensity of light scattered at different angles, allowing for the calculation of parameters like particle size and distribution.

### Python Example: Static Light Scattering Simulation

Below is a Python script using NumPy and Matplotlib to simulate the scattering intensity at different angles for small particles.

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
wavelength = 650e-9  # Wavelength of light in meters (650 nm, red light)
refractive_index = 1.33  # Refractive index of the medium (e.g., water)
particle_radius = 100e-9  # Particle radius in meters (100 nm)

# Function to calculate the scattering angle dependence (Rayleigh-Debye scattering)
def scattering_intensity(theta, wavelength, refractive_index, radius):
    # Size parameter: x = (2 * pi * radius) / wavelength
    size_parameter = (2 * np.pi * radius) / wavelength
    
    # Rayleigh-Debye scattering approximation for small particles
    # Scattering intensity is proportional to (sin(theta) / theta)^2
    intensity = (np.sin(theta) / theta)**2
    return intensity

# Define scattering angles (in radians)
angles = np.linspace(0.01, np.pi, 500)  # Avoid theta=0 for numerical stability

# Calculate the scattering intensity
intensity = scattering_intensity(angles, wavelength, refractive_index, particle_radius)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(angles, intensity, label=f'Radius = {particle_radius*1e9} nm')
plt.xlabel('Scattering Angle (radians)')
plt.ylabel('Scattering Intensity (arbitrary units)')
plt.title('Static Light Scattering Intensity vs. Scattering Angle')
plt.legend()
plt.grid(True)
plt.show()
```

### Explanation:

- **Constants**: The wavelength of the light is set to 650 nm (red light), and the particle radius is set to 100 nm.
- **`scattering_intensity` function**: This function calculates the scattering intensity based on the Rayleigh-Debye approximation. It computes the intensity as proportional to \( $\left(\frac{\sin(\theta)}{\theta}\right)^2$ \), where \( $\theta$ \) is the scattering angle.
- **Angle Range**: We define the range of angles from 0.01 radians to \( $\pi$ \) radians (to avoid division by zero at \( $\theta = 0$ \)).
- **Plotting**: The intensity is plotted as a function of scattering angle. The result will show how the intensity of scattered light changes with the angle, which is key for understanding the particle's scattering properties.

### Steps to Run:

1. **Install Required Libraries**:
   If you don't already have the required libraries, install them using:

   ```bash
   pip install numpy matplotlib
   ```

2. **Save and Run the Code**:
   Save the code in a Python file (e.g., `static_light_scattering.py`) and run it:

   ```bash
   python static_light_scattering.py
   ```

### Further Enhancements:

- **Multiple Particle Sizes**: You can simulate and compare scattering for particles of different sizes by modifying the `particle_radius` and re-running the simulation.
- **Polydisperse Systems**: If you want to model a system with a distribution of particle sizes, you can use a weighted sum of scattering intensities for particles of different sizes.
- **Incorporating Other Factors**: The model can be extended to include effects like particle concentration, refractive index, and different types of scatterers.

### Advanced Analysis:

1. **Size Distribution**: If you have experimental SLS data (scattering intensity vs. angle), you could fit the data using models like the Guinier model (for small angles) or other more complex fitting techniques to estimate particle size distributions.
2. **Multi-Scale Simulations**: For larger particles (when the size becomes comparable to or larger than the wavelength), you would need to use more complex models (e.g., Mie theory) to model the scattering.

This basic implementation gives a foundation for understanding and simulating Static Light Scattering in Python. Further work could involve fitting experimental data or extending the model to account for more complex systems.

### :cactus:R snippet

Static Light Scattering (SLS) is a technique used to analyze the size and shape of particles in a solution or suspension by measuring the intensity of scattered light at different angles. Unlike Dynamic Light Scattering (DLS), which measures the fluctuations in light scattering over time, SLS uses the angular dependence of scattered light to provide information about particle size, molecular weight, and other characteristics.

### Key Concepts of Static Light Scattering (SLS):

1. **Rayleigh and Mie Scattering**: These are two different types of scattering that describe how light interacts with small or large particles, respectively.
2. **Intensity of Scattering**: The intensity of light scattered at an angle depends on the size, shape, and refractive index of the particles.
3. **Zimm Plot**: A commonly used plot in SLS, which plots the inverse of the scattering intensity against the scattering angle to extract the radius of gyration and other molecular parameters.

### Formula for Scattering Intensity:

For SLS, the scattered intensity at a specific angle \( \theta \) is given by the following expression:

$$
I(\theta) = \frac{N}{r} \cdot P(\theta)
$$
where:

- \( $N$ \) is the number of particles,
- \( $r$ \) is the radius of the particle,
- \( $P(\theta)$ \) is the form factor, which depends on the shape of the particle and the scattering angle \( $\theta $\).

For spherical particles, \( $P(\theta)$ \) simplifies to a known function, but for non-spherical particles, it is more complex.

### Example Code for Static Light Scattering in R

Here is a basic R code that simulates the light scattering data and generates a Zimm plot for analysis.

#### Step 1: Simulate the Scattering Data

We'll simulate scattering intensities for spherical particles and fit a Zimm plot to extract the radius of gyration.

```r
# Load necessary libraries
library(ggplot2)

# Constants
lambda <- 633e-9  # Wavelength of light (in meters)
n <- 1.33         # Refractive index of medium (e.g., water)

# Simulate scattering data
scattering_angle <- seq(10, 120, by = 5)  # Angles in degrees
scattering_angle_rad <- scattering_angle * pi / 180  # Convert to radians

# Simulate form factor for spherical particles (for simplicity)
# P(theta) is a simplified form for spherical particles
form_factor <- (1 + 2 * (sin(scattering_angle_rad / 2))^2)

# Assume particle size (radius in nm) and number of particles
radius <- 100  # Particle radius in nm
particle_number <- 1e10  # Number of particles (for simplicity)

# Calculate scattering intensity (simplified version)
intensity <- (particle_number / radius) * form_factor

# Create a data frame for plotting
scattering_data <- data.frame(
  angle = scattering_angle,
  intensity = intensity
)

# Plot the scattering data
ggplot(scattering_data, aes(x = angle, y = intensity)) +
  geom_point() +
  geom_line() +
  labs(title = "Simulated Static Light Scattering Data", 
       x = "Scattering Angle (degrees)", y = "Scattering Intensity")
```

### Explanation:

- **Form Factor**: This is a simple model for spherical particles, where the intensity depends on the scattering angle.
- **Intensity Calculation**: The scattered light intensity is calculated using a simplified formula for spherical particles.
- **Plot**: We plot the intensity as a function of the scattering angle.

#### Step 2: Zimm Plot and Extraction of Particle Size

In practice, the Zimm plot is used to extract the radius of gyration ( $R_g$ ) of particles from the scattering intensity data. The Zimm plot is created by plotting the inverse of the intensity against \( $\sin^2(\theta/2)$ \).

```r
# Create the Zimm plot data by applying the Zimm equation
zimm_plot_data <- data.frame(
  angle = scattering_angle,
  inverse_intensity = 1 / intensity,
  sin_squared = sin(scattering_angle_rad / 2)^2
)

# Plot the Zimm plot
ggplot(zimm_plot_data, aes(x = sin_squared, y = inverse_intensity)) +
  geom_point() +
  geom_line() +
  labs(title = "Zimm Plot", x = "sin²(θ/2)", y = "1/Scattering Intensity")
```

### Explanation:

- **Zimm Plot**: This plot relates the inverse of the scattering intensity to \( $\sin^2(\theta/2)$ \). By fitting this plot, we can estimate the particle size, specifically the radius of gyration.

### Steps to Run:

1. **Install R and RStudio**: If not already installed, install R from [CRAN](https://cran.r-project.org/) and RStudio for an interactive development environment.

2. **Install ggplot2**: To install `ggplot2` for plotting, run the following command in R:

   ```r
   install.packages("ggplot2")
   ```

3. **Run the Code**: Copy the code into an R script or RStudio and execute it.

### Further Enhancements:

1. **Noise Simulation**: Add Gaussian noise to the scattering intensities to simulate real-world experimental data.
2. **Multiple Particle Sizes**: Simulate a mixture of particles with different sizes and observe how it affects the scattering data.
3. **Complex Form Factors**: For non-spherical particles or more complex shapes, implement more complex form factors for \( P(\theta) \).

### Advanced Considerations:

- **Fitting**: To accurately extract particle size, a nonlinear regression or least squares fitting method can be applied to the Zimm plot or other scattering models.
- **Real Data**: For real-world data, SLS measurements typically require high-precision instruments to measure scattering intensities at multiple angles.
- **Polydispersity**: Consider particle size distributions, which can be analyzed using techniques like the CONTIN algorithm.

This example code provides a simple framework for simulating and analyzing Static Light Scattering data using R, helping you understand the fundamental concepts of SLS and its application in characterizing particle size distributions.

