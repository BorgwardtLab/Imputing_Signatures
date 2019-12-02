library(scmamp)

#first set directory to location of this script:
setwd('/Users/mimoor/Desktop/localwork/signatures/GP_Signatures/scripts/')

#data(data_gh_2008)
#plotCD(data.gh.2008, alpha=0.01)

data <- read.table(
  '../results/full_tabular_results.csv',
  sep=',',
  header=TRUE,
  #row.names=1
)
row_names <- rownames(data)
rownames(data) <- c()

plotCD(data, alpha=0.01)
