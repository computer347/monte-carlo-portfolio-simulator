# Monte Carlo Simulation Engine

A robust modular data pipeline and simulation engine analyzing Coffee futures and S&P 500 assets via Monte Carlo distributions (GBM, Student-T, Merton Jump-Diffusion).

## Required Dependencies

Install the requirements from the root directory:
```bash
pip install -r requirements.txt
```

## Running the Terminal Pipeline (Tasks 1 & 2)

```bash
python main.py
```
This executes raw fetch validations, parses statistical constraints over a 252 day window, outputs baseline diagnostic charts locally matching statistical diagnostics, and prints simulated parameter behaviors.

## Running the Interactive Web Dashboard (Task 3)

We offer a native Plotly / Dash interactive user interface projecting models locally.

To launch the web dashboard:
```bash
python dashboard/app.py
```
Then navigate to `http://localhost:8050` in your browser.
