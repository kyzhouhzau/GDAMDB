This folder contains the GWAS summary data related to the intereseted disease, and the example is **GWASsummaryData_example.txt** which can be replaced by other GWAS summary data. 

Before data fusion, the SNPs in this file need to be mapped to specific genes. In our work, we applied Bedtools with the mapping rule that the p-value of the gene equals to the SNP with highest p-value. The result should be procedded as the same format as **sorted_IGAP.csv**, which is the example mapping output.
