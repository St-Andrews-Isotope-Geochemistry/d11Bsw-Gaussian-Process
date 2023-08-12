import sys
sys.path.append("./Code/Data_Manipulation/")

import numpy,json
from importData import importData
from matplotlib import pyplot
# from cbsyst import boron_isotopes
from geochemistry_helpers import Sampling,GaussianProcess

# Resolution of d11Bsw
d11Bsw_x = numpy.arange(-50,100,1e-1)

# Load in boron data from Rae et al., Annual Reviews paper
d11B4_data = importData("d11B")

# Load in Henehan data
d11Bsw_henehan = importData("henehan")

# Load in Gutjahr data
d11Bsw_gutjahr = importData("gutjahr")

# Load in the Greenop data
d11Bsw_greenop = importData("greenop")

# Load in Anagnostou data
d11Bsw_anagnostou = importData("anagnostou")

# Load in d11Bsw limitations from d11B4 (using epsilon constraint)
with open("./Data/Output/d11Bsw_from_d11B4.json","r") as file:
    d11Bsw_from_d11B4_data = json.loads(file.read())

# Collate non-Gaussian constraints
d11Bsw_central_constraints = d11Bsw_gutjahr
non_gaussian_constraints = d11Bsw_anagnostou+d11Bsw_greenop+d11Bsw_henehan
non_gaussian_constraints_outlier = []
for distribution in non_gaussian_constraints:
    normal_addition = Sampling.Distribution(distribution.bin_edges,"Gaussian",(distribution.mean,distribution.standard_deviation)).normalise()
    # inflated = distribution.approximateGaussian(inflation=3).normalise()
    non_gaussian_constraints_outlier += [Sampling.Distribution(distribution.bin_edges,"Manual",(0.5*distribution.probabilities+0.5*normal_addition.probabilities)).normalise()]
# non_gaussian_constraints_outlier = [Sampling.Distribution(distribution.bin_edges,"Manual",distribution.probabilities+numpy.max(distribution.probabilities)/10) for distribution in non_gaussian_constraints]
# These need to be approximated into Gaussians
non_gaussian_constraints_inflated = []
for distribution in d11Bsw_anagnostou:
    non_gaussian_constraints_inflated += [distribution.approximateGaussian(inflation=1.5)]
for distribution in d11Bsw_greenop:
    non_gaussian_constraints_inflated += [distribution.approximateGaussian(inflation=3)]
for distribution in d11Bsw_henehan:
    non_gaussian_constraints_inflated += [distribution.approximateGaussian(inflation=1.5)]

# Store the ages at which we'll query the GP
non_gaussian_age = numpy.array([constraint.location for constraint in non_gaussian_constraints])
bounds_age = numpy.array([bound["location"] for bound in d11Bsw_from_d11B4_data])

# Set the interpolation ages
interpolation_locations = [d11B4_data["age"].to_numpy(),bounds_age,non_gaussian_age,numpy.arange(0,100,0.1)]

 
## Create constraints
# Create a list of things that will be used to guide the GP
d11Bsw_constraints = []

# First - the most direct estimates
# Hardcode modern value
d11B4_modern_constraint = Sampling.Distribution(d11Bsw_x,"Gaussian",(39.61,0.01),location=0).normalise()

# Assemble into a single list
d11Bsw_constraints += [d11B4_modern_constraint]
d11Bsw_constraints += d11Bsw_central_constraints
d11Bsw_constraints += non_gaussian_constraints_inflated

# Second, the non-Gaussian ones
# d11Bsw_ensemble_inflated = [distribution.approximateGaussian(inflation=2) for distribution in d11Bsw_ensemble]
# d11Bsw_anagnostou_inflated = [distribution.approximateGaussian(inflation=2) for distribution in d11Bsw_anagnostou]
# non_gaussian_constraints_inflated = d11Bsw_anagnostou_inflated+d11Bsw_ensemble_inflated

# To evaluate the probability of rejection we have three criteria:
# - Is it below any minima or above any set maxima? (then definitely reject)
# - Is the gradient above what is considered reasonable? (then definitely reject)
# - Has approximating the non-Gaussians as Gaussians overweighted this particular area? (then possibly reject)

# First the minima and maximum
# Transform d11Bsw_from_d11B4 into distributions
d11Bsw_bounds = []
for distribution_data in d11Bsw_from_d11B4_data:
    d11Bsw_bounds += [Sampling.Distribution(numpy.array(distribution_data["bin_edges"]),"manual",numpy.array(distribution_data["probabilities"]),location=distribution_data["location"])]

# Then set the reasonable gradient
rate_of_change_limit_function = lambda t : (0.8/100)*t + 0.2
rate_of_change_limit = rate_of_change_limit_function(interpolation_locations[-1][:-1])

# Finally determine scaling for non-Gaussian constraints based on supremum for possible rejection
scaled_non_gaussian_constraints = []
acceptance_distributions = []
for distribution,inflated in zip(non_gaussian_constraints_outlier,non_gaussian_constraints_inflated,strict=True):
    ratio = (distribution.probabilities-numpy.min(distribution.probabilities))/(inflated.probabilities)
    scaling = numpy.nanmax(ratio[numpy.isfinite(ratio)])
    scaled_probabilities = (inflated.probabilities*scaling)+numpy.min(distribution.probabilities)
    scaled_non_gaussian_constraints += [Sampling.Distribution(distribution.bin_edges,"manual",scaled_probabilities,location=distribution.location)]

    if inflated.location<30:
        acceptance_probabilities = 0.8+(0.2*(distribution.probabilities)/(scaled_probabilities))
    else:
        acceptance_probabilities = 0.2+(0.8*(distribution.probabilities)/(scaled_probabilities))
    acceptance_distributions += [Sampling.Distribution(distribution.bin_edges,"manual",acceptance_probabilities,location=distribution.location)]


# Now we're ready to set up the GP
d11Bsw_gp = GaussianProcess().constrain(d11Bsw_constraints).setKernel("rbf",(2,10)).query(interpolation_locations)

iteration = 0
# best_samples = [numpy.empty((0,len(location_group))) for location_group in interpolation_locations]
# probabilities = numpy.empty((0,))
viable_samples = Sampling.MarkovChain()
while len(viable_samples)<10000:
    # Get a number of samples
    d11Bsw_gp.getSamples(10000)

    # Determine whether these are below minima or above maximum
    d11Bsw_from_d11B4_gp = GaussianProcess().constrain(d11Bsw_bounds).assignSamples([d11Bsw_gp.samples[-3]])
    sample_likelihood = d11Bsw_from_d11B4_gp.getSampleLikelihood(keep_separate=True,logspace=False)
    
    # Determine whether gradient is reasonable
    rate_of_change = numpy.diff(d11Bsw_gp.samples[-1])/0.1
    rate_of_change_likelihood = numpy.all(rate_of_change<rate_of_change_limit,axis=1)

    non_gaussian_gp = GaussianProcess().constrain(acceptance_distributions).assignSamples([d11Bsw_gp.samples[-2]])
    non_gaussian_likelihood = non_gaussian_gp.getSampleLikelihood(keep_separate=True,logspace=False)
    
    d11Bsw_likelihood = sample_likelihood*rate_of_change_likelihood*non_gaussian_likelihood

    random_values = numpy.random.rand(len(d11Bsw_likelihood))

    accept = random_values<=d11Bsw_likelihood
    viable_indices = numpy.where(accept)

    for viable_index in viable_indices[0]:
        sample = Sampling.MarkovChainSample()
        sample = sample.addField("d11Bsw",[d11Bsw_sample_group[viable_index,:] for d11Bsw_sample_group in d11Bsw_gp.samples],precision=5)
        viable_samples = viable_samples.addSample(sample)    

    iteration += 1
    print(str(iteration)+" iterations done")
    print(str(len(viable_samples))+" viable samples")

# Load in the Greenop data
d11Bsw_ensemble = importData("greenop")

# Load in Anagnostou data
d11Bsw_anagnostou = importData("anagnostou")
non_gaussian_constraints = d11Bsw_anagnostou+d11Bsw_ensemble

d11Bsw_constraints = []
d11Bsw_constraints += [d11B4_modern_constraint]
d11Bsw_constraints += d11Bsw_central_constraints
d11Bsw_constraints += non_gaussian_constraints

d11Bsw_gp = GaussianProcess().constrain(d11Bsw_constraints).setKernel("rbf",(2,10)).query(interpolation_locations)
d11Bsw_gp.fromMCMCSamples(viable_samples.accumulate("d11Bsw"))
d11Bsw_gp.toJSON("./Data/Output/d11Bsw_slow.json")
