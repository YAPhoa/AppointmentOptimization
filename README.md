# Appointment Optimization
MH4702-Probabilistic Methods in OR Project

This repository contains my group project on the course MH4702-Probabilistic Methods in Operations Research at NTU. Our task is to pick a simple optimization problem in OR as well as provide solution to such problem. Our project submission title is "Exploring simple appointment based queuing system". My main contribution on this project is on the optimization section.

For complete problem description please refer to our [project report](MH4702_Project.pdf)

Our approach on optimizing the said problem is through bayesian optimization. From my experience, bayesian optimization provide a good balance between exploration and exploitation which proven to be very effective in our problem as we are optimizing under uncertainties.

## Required packages :
- pandas
- numpy
- BayesianOptimization (https://github.com/fmfn/BayesianOptimization)

## References
- http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
