# FLOSSY

Fit Linear periOd-Spacing patternS interactivelY

(Acronym created before the advent of LLMs)

# Physical context

Period-spacing patterns are an asteroseismic tool that probes the near-core regions of a star. Tassoul 1980 predicts the period of gravity modes to be equidistant (i.e.,  constant ΔP) for a spherical chemically-homogeneous non-rotating star. Keppler and TESS observations have revealed a variety of non-constant patterns in plots of ΔP vs P whose signatures have been associated with the physic inside the star. For instance, Miglio et al. 2008 linked the dips in such patterns to gradients in the star's chemical composition and Van Reeth et al. 2015abc established the relation between the pattern's slope and the interior rotation. The community is currently working on mathematical descriptions (e.g. Townsend 2003, Cunha et al. 2019) and, at the same time, using these patterns as constraints on forward asteroseismic modelling to probe the physics near the stellar core (e.g. Pedersen et al. 2021).

Period-spacing patterns are ultimately identified manually, especially when they are used for forward asteroseismic modelling (Aerts et al. 2018). FLOSSY aims to facilitate this task by offering an interactive tool that lets the user search the ΔP pattern in a previously computed periodogram while offering diagnostic plots as the  ΔP vs P and the period Echelle diagram, both often used to spot and confirm the pattern. FLOSSY assumes that the ΔP changes at a linear rate with respect to P and uses the same parametrization as Li et al. 2019.

# Requirements
You can cover all requirements by installing the following virtual environment using [Conda](https://docs.conda.io/en/latest) and the YML file in the repo.

```
conda env create -f flossy_environment.yml
```


⮕ Alternatively, you can use `pip` to directly install the package along with all its dependences. For it, go to the directory containing the `setup.py` file and run:

```
pip install .
```

If you wish so, you can later uninstall it with:

```
pip uninstall flossy
``` 

# Importing FLOSSY
If you used `pip` to install the package, then `flossy` should be ready to import like any other Python module.

If you just downloaded or cloned the repository, then you need to add its location to the environment variable used by Python to search for modules, so that you can have access to the `flossy` module regardless your location on your machine.

# Run example
# Fit example

![demonstration](flossy.gif)

# Found an issue in the code?
Create an issue on GitHub and include a minimal reproducible example of it.

# Reference paper
https://doi.org/10.1051/0004-6361/202141926