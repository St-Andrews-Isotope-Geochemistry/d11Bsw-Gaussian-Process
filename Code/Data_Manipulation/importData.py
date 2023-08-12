import pandas,numpy
from geochemistry_helpers import Sampling

def importData(name):
    if name.lower()=="d11b":
        data = pandas.read_excel("./Data/Input/Rae_2021_Cenozoic_CO2_Precalculated.xlsx",sheet_name="d11B_data",header=0,usecols="C,D,E,F,I,N,P,Q,T,V,W",names=["age","salinity","temperature","depth","d11B_uncertainty","exclude","calcium","magnesium","d11Bsw","d11B","pH"])
        data = data[data["exclude"]==0].dropna(subset=["age","d11B"])
        data["age"] = data["age"]/1e3
        return data
    elif name.lower()=="henehan":
        data = pandas.read_excel("./Data/Input/Constraints.xlsx",sheet_name="Henehan",header=1,usecols="A:D",names=["age","value","uncertainty","type"])
        
        d11Bsw_x = numpy.arange(0,100,1e-2)
        constraint = Sampling.Distribution(d11Bsw_x,"Flat",((data["value"]-data["uncertainty"]/2).values[0],(data["value"]+data["uncertainty"]/2).values[0]),location=data["age"].values[0]).normalise()
        return [constraint]
    elif name.lower()=="gutjahr":
        data = pandas.read_excel("./Data/Input/Constraints.xlsx",sheet_name="Gutjahr",header=1,usecols="A:D",names=["age","value","uncertainty","type"])
        
        d11Bsw_x = numpy.arange(0,100,1e-2)
        constraint = Sampling.Distribution(d11Bsw_x,"Gaussian",(data["value"].values[0],data["uncertainty"].values[0]),location=data["age"].values[0]).normalise()
        return [constraint]
    elif name.lower()=="greenop":
        ages = pandas.read_excel("./Data/Input/Greenop_2017_Ensemble.xlsx",sheet_name="results",header=0,usecols="B:W",nrows=1)
        data = pandas.read_excel("./Data/Input/Greenop_2017_Ensemble.xlsx",sheet_name="results",header=1,usecols="B:W")

        d11Bsw_x = numpy.arange(0,100,1e-2)
        d11Bsw_constraints = []
        for age,column in zip(ages.to_numpy()[0],data,strict=True):
            constraint = Sampling.Distribution.fromSamples(data[column].dropna(),bin_edges=d11Bsw_x).normalise()
            constraint.location = age
            d11Bsw_constraints += [constraint]
        return d11Bsw_constraints
    elif name.lower()=="anagnostou":
        data_max = pandas.read_excel("./Data/Input/Constraints.xlsx",sheet_name="Anagnostou_Ensemble",header=13,usecols="B:E",names=["53.0Ma","45.6Ma","44.4Ma","37.0Ma"])
        data_min = pandas.read_excel("./Data/Input/Constraints.xlsx",sheet_name="Anagnostou_Ensemble",header=16,usecols="G:J",names=["53.0Ma","45.6Ma","44.4Ma","37.0Ma"])
        
        data_min = data_min.dropna()

        d11Bsw_x = numpy.arange(0,100,1e-3)
        distributions = []

        probabilities = 0
        for d11Bsw_maximum in data_max["53.0Ma"].dropna().values:
            distribution = Sampling.Distribution(d11Bsw_x,"Flat",(data_min["53.0Ma"].values[0],d11Bsw_maximum)).normalise()
            probabilities += distribution.probabilities        

        distributions += [Sampling.Distribution(d11Bsw_x,"Manual",probabilities,location=53.0).normalise()]

        probabilities = 0
        for d11Bsw_maximum in data_max["45.6Ma"].dropna().values:
            distribution = Sampling.Distribution(d11Bsw_x,"Flat",(data_min["45.6Ma"].values[0],d11Bsw_maximum)).normalise()
            probabilities += distribution.probabilities        

        distributions += [Sampling.Distribution(d11Bsw_x,"Manual",probabilities,location=45.6).normalise()]

        probabilities = 0
        for d11Bsw_maximum in data_max["44.4Ma"].dropna().values:
            if d11Bsw_maximum>data_min["44.4Ma"].values[0]:
                distribution = Sampling.Distribution(d11Bsw_x,"Flat",(data_min["44.4Ma"].values[0],d11Bsw_maximum)).normalise()
                probabilities += distribution.probabilities        

        distributions += [Sampling.Distribution(d11Bsw_x,"Manual",probabilities,location=44.4).normalise()]

        probabilities = 0
        for d11Bsw_maximum in data_max["37.0Ma"].dropna().values:
            distribution = Sampling.Distribution(d11Bsw_x,"Flat",(data_min["37.0Ma"].values[0],d11Bsw_maximum)).normalise()
            probabilities += distribution.probabilities        

        distributions += [Sampling.Distribution(d11Bsw_x,"Manual",probabilities,location=37.0).normalise()]

        return distributions
    elif name.lower()=="strontium":
        data = pandas.read_excel("./Data/Input/StrontiumLithiumOsmium.xlsx",sheet_name="Strontium",header=1,usecols="B,E,F",names=["age","strontium","strontium_uncertainty"])
        data = data.dropna()
        return data
    elif name.lower()=="lithium":
        data = pandas.read_excel("./Data/Input/StrontiumLithiumOsmium.xlsx",sheet_name="Lithium",header=1,usecols="B,E,F",names=["age","lithium","lithium_uncertainty"])
        data.loc[data["lithium_uncertainty"]==0,"lithium_uncertainty"] = 0.1 # One row has zero uncertainty!!
        return data
    elif name.lower()=="osmium":
        data = pandas.read_excel("./Data/Input/StrontiumLithiumOsmium.xlsx",sheet_name="Osmium",header=1,usecols="A,H:I",names=["age","osmium","osmium_uncertainty"])
        assumed_mean = data["osmium_uncertainty"].dropna().mean()
        data["osmium_uncertainty"][data["osmium_uncertainty"]==0] = assumed_mean
        data["osmium_uncertainty"].fillna(assumed_mean)
        data = data.dropna()
        return data
    else:
        raise(ValueError("Unknown input name"))