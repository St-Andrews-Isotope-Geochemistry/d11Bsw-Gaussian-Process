import sys
sys.path.append("./Code/Data_Manipulation/")

import numpy,json
from importData import importData
from matplotlib import pyplot
from cbsyst import boron_isotopes
from geochemistry_helpers import Sampling,GaussianProcess

d11B4_data = importData("d11B")

## Process d11B4
d11B_x = numpy.arange(0,60,1e-1)
age_bin_spacing = 1
age_bins = numpy.arange(0,100,age_bin_spacing)
age_bins = age_bins[0:-1]+age_bin_spacing/2

d11B4_bin_indices = numpy.digitize(d11B4_data["age"].to_numpy(),age_bins)

ages = []
d11B_minima = []
d11B_maxima = []
for d11B4_bin_index in range(min(d11B4_bin_indices),max(d11B4_bin_indices)+1):
    relevant_data = d11B4_data[d11B4_bin_indices==d11B4_bin_index]
    if len(relevant_data)>0:
        ages += [age_bins[d11B4_bin_index]]

        minimum_point = relevant_data[relevant_data["d11B"]==min(relevant_data["d11B"])]
        maximum_point = relevant_data[relevant_data["d11B"]==max(relevant_data["d11B"])]

        d11B_minima += [list(minimum_point[["d11B","d11B_uncertainty"]].to_numpy()[0])]
        d11B_maxima += [list(maximum_point[["d11B","d11B_uncertainty"]].to_numpy()[0])]

accumulated_distributions = []
for minimum,maximum,age in zip(d11B_minima,d11B_maxima,ages,strict=True):
    iterations = 1000

    epsilon_distributions = [Sampling.Distribution(d11B_x,"Gaussian",(27.2,0.6)).normalise(),Sampling.Distribution(d11B_x,"Gaussian",(26.0,1)).normalise()]
    epsilon_sampler = Sampling.Sampler(d11B_x,"Manual",epsilon_distributions[0].probabilities+epsilon_distributions[1].probabilities,"Monte_Carlo").normalise()
    minimum_sampler = Sampling.Sampler(d11B_x,"Gaussian",(minimum[0],minimum[1]),"Monte_Carlo").normalise()
    maximum_sampler = Sampling.Sampler(d11B_x,"Gaussian",(maximum[0],maximum[1]),"Monte_Carlo").normalise()
    
    epsilon_sampler.getSamples(iterations)
    minimum_sampler.getSamples(iterations)
    maximum_sampler.getSamples(iterations)

    minimum_d11Bsw_samples = maximum_sampler.samples
    maximum_d11Bsw_samples = boron_isotopes.R11_to_d11(boron_isotopes.d11_to_R11(minimum_sampler.samples)*boron_isotopes.d11_to_R11(epsilon_sampler.samples,SRM_ratio=1))

    probabilities = numpy.zeros(numpy.shape(minimum_sampler.probabilities))
    for minimum_sample,maximum_sample in zip(minimum_d11Bsw_samples,maximum_d11Bsw_samples,strict=True):
        distribution = Sampling.Distribution(d11B_x,"Flat",(minimum_sample,maximum_sample)).normalise()
        probabilities += distribution.probabilities
    accumulated_distributions += [Sampling.Distribution(d11B_x,"Manual",probabilities,location=age).normalise(1,by="Maximum")]

json_data = json.dumps(accumulated_distributions,cls=Sampling.MCEncoder,indent=4)
json_data_stripped = json_data.replace('"xxx',"").replace('xxx"',"").replace('xxx',"")
with open("./Data/Output/d11Bsw_from_d11B4.json","w") as file:
    file.write(json_data_stripped)

a = 5