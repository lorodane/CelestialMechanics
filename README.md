This is a collection of numerical experiments in the idealized Sitnikov problem.

The Sitnikov problem is a version of the 3-body problem with two equal masses moving on ellipses around their common center of mass and a third massless body moving on the line through the center of mass perpendicular to the plane of the massive bodies. It is perhaps the simplest celestial mechanics system showcasing both regular and chaotic behaviour.

This is part of my (Leonardo Costa Lesage) undergraduate thesis at UPC, as part of the CFIS double bachelor programme, undertaken at the University of Oxford in the Department of Physics, under the supervision of Raymond Pierrehumbert and Eva Miranda. Many thanks to both for their guidance and giving me the chance to work on this topic with them.

## Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for Python environment and package management.

### Setup
To install dependencies and create a virtual environment:
```powershell
uv sync
```

### Running Scripts
Always run Python scripts through `uv` to ensure the correct environment is used:
```powershell
uv run python path/to/script.py
```

### Adding Packages
To add new libraries (e.g., `scipy`):
```powershell
uv add scipy
```
