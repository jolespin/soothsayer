# install_r_packages.r
source("http://bioconductor.org/biocLite.R")
# WGCNA
biocLite(c("AnnotationDbi", "impute", "GO.db", "preprocessCore", "dynamicTreeCut", "WGCNA"), suppressUpdates=TRUE)
# edgeR
biocLite("edgeR", suppressUpdates=TRUE)
# APE
biocLite("ape", suppressUpdates=TRUE)
# metagenomeSeq
biocLite("metagenomeSeq", suppressUpdates=TRUE)
# phyloseq
biocLite("phyloseq", suppressUpdates=TRUE)
# ggtree
devtools::install_github("GuangchuangYu/ggtree")
# philr
devtools::install_github('jsilve24/philr')
