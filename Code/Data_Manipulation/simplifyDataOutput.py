import numpy,pandas
import geochemistry_helpers
import numpy,pandas,openpyxl

# Markov chain
# markov_chain = geochemistry_helpers.MarkovChain().fromJSON("./Data/Output/markov_chain.json").round(2)

d11Bsw_gp = geochemistry_helpers.GaussianProcess().fromJSON("./Data/Output/d11Bsw_slow_rate_limited.json")
d11Bsw_x = d11Bsw_gp.queries[0][0].bin_edges

median = d11Bsw_gp.quantile(0.5,group=-1)
quantile_5 = d11Bsw_gp.quantile(0.05,group=-1)
quantile_95 = d11Bsw_gp.quantile(0.95,group=-1)

interpolated_dataframe = pandas.DataFrame(d11Bsw_gp.samples[-1])
interpolated_dataframe.columns = d11Bsw_gp.query_locations[-1]

metrics_dataframe = pandas.DataFrame()
metrics_dataframe["age"] = d11Bsw_gp.query_locations[-1]
metrics_dataframe["median"] = median
metrics_dataframe["5% quantile"] = quantile_5
metrics_dataframe["95% quantile"] = quantile_95

workbook = openpyxl.Workbook(write_only=True)
interpolated_sheet = workbook.create_sheet(title="Interpolated")
metrics_sheet = workbook.create_sheet(title="Metrics")

interpolated_sheet.append(list(interpolated_dataframe.columns))
for index,row in interpolated_dataframe.iterrows():
    interpolated_sheet.append(list(row))

metrics_sheet.append(list(metrics_dataframe.columns))
for index,row in metrics_dataframe.iterrows():
    metrics_sheet.append(list(row))

workbook.save("./Data/Output/d11Bsw_metrics.xlsx")

a = 5



a = 5