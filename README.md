# Content 

This package permits to try algorithms for Best Arm Identification (BAI) and Epsilon Best Arm Identification (epsilon-BAI) mentionned in the papers 
* Optimal Best Arm Identification with Fixed Confidence, Garivier and Kaufmann, COLT 2016
* Non-Asymptotic Sequential Tests for Overlapping Hypotheses and application to near optimal arm identification in bandit models, Garivier and Kaufmann, arXiv:1905.03495
* Fixed-Confidence Guarantees for Bayesian Best-Arm Identification, Shang et al., AISTATS 2020 


# How does it work? 

Choose the parameters of your BAI (resp. epsilon-BAI) problem in mainBAI.jl (resp. mainEpsilonBAI.jl) before running this file.

All the fields that start with capital letters have to be cumstomized for your experiments (e.g. # NUMBER OF SIMULATIONS). 

Experiments will be run in parallel if you open julia with the command julia -p x, where x is (smaller than) the number of CPUs on your machine. 
- choosing typeExp = "Save" will save the results in the results folder 
- results will be displayed in the command window anyways

If you have saved results, running viewResults.jl will help visualizing them (histogram of the number of draws will be printed). 
Name and parameters, specified at the beginning of this file, should match with your saved data
Also, you may want to change the histogramm parameters depending on your problem.
In viewResults.jl, you need to specify whether you are doing BAI or epsilon-BAI.

BAIalgos.jl (resp. EpsilonBAIalgos.jl) contains all algorithms for BAI (resp. epsilon-BAI): before including it, the type of distribution for the arms should be specified
typeDistribution can take the values "Bernoulli" "Gaussian" "Poisson" "Exponential"


# Configuration information

Experiments were run with the following Julia install: 

julia> VERSION
v"1.4.2"

(v1.0) pkg> status
 - Distributions v0.23.11
 - HDF5 v0.12.5
 - PyPlot v2.9.0
 - Distributed 

To install these package, you can run 

>using Pkg; Pkg.add("HDF5"); Pkg.add("Distributions"); Pkg.add("PyPlot"); Pkg.add("Distributed")

# MIT License

Copyright (c) 2020 [Aurélien Garivier and Emilie Kaufmann]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
