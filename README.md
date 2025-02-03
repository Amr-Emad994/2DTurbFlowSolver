# CFD Turbulent Flow Solver

## Description
This repository contains a Python-based Computational Fluid Dynamics (CFD) solver for simulating turbulent incompressible flows. The solver employs the k-epsilon turbulence model with wall functions and utilizes the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm for pressure-velocity coupling. It is designed for educational and research purposes to demonstrate fundamental CFD techniques.

## Key Features
- **Turbulence Modeling**: k-epsilon model with wall functions for near-wall treatment
- **SIMPLE Algorithm**: Pressure-velocity coupling for incompressible flows
- **Staggered Grid**: Finite volume discretization on a collocated grid
- **Boundary Conditions**: Wall boundary conditions with log-law wall functions
- **Output Handling**: CSV output for velocity, pressure, and turbulence quantities

## Prerequisites
- Python 3.x
- NumPy

## Usage
Run the simulation directly:
```bash
python flowSolver.py
```
When prompted, enter:
- Number of cells in x-direction (e.g., 50)
- Number of cells in y-direction (e.g., 20)
- Number of outer iterations (e.g., 100)

### Example Usage
```bash
$ python flowSolver.py
Enter number of cells in x-direction: 50
Enter number of cells in y-direction: 20
Enter number of outer iterations: 100
```

## Output Files
1. **RESULTS1.csv**: Contains pressure and velocity components at grid points
   - Format: `x, y, z, pressure, velmag, vx, vy, vz`
2. **RESULTS2.csv**: Contains turbulence quantities
   - Format: `x, y, z, k, eps, mut`
3. **wall_profile.txt**: Velocity profile near walls
   - Format: `y/ymax, u/u_max`

## Key Parameters (Adjustable in Code)
- Fluid density (`rho`)
- Turbulence intensity (`ti`)
- Under-relaxation factors (`omegam`, `omegap`, `omegak`, `omegaeps`)
- Turbulence model constants (`cep1`, `cep2`, `cmu`, `sigk`, `sigeps`)
- Domain dimensions (`xmin`, `xmax`, `ymin`, `ymax`)

## References
1. Patankar, S.V., "Numerical Heat Transfer and Fluid Flow"
2. Versteeg, H.K. & Malalasekera, W., "An Introduction to Computational Fluid Dynamics"

## License
This project is licensed under the MIT License

## Disclaimer
**This code is provided as-is for educational purposes. Users are encouraged to validate results and adapt the code to their specific needs.**
