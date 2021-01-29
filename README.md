# Data Collection
The **MutationTypeData** folder contains the abstracts related to the interested disease in json files that are downloaded from PubTator. The json files need to be processed to BIO format, the example of which is located at **BERT_multi_task/data/BIO_example.txt**

`python MutationTypeData/json2bio.py --input MutationTypeData/ --output BERT_multi_task/data/BIO_example.txt`

The **MutationAssociationData** contains the GWAS summary data related to the intereseted disease, and the example is **GWASsummaryData_example.txt** which can be replaced by other GWAS summary data. 

Before data fusion, the SNPs in this file need to be mapped to specific genes. In our work, we applied **bedtools** with the mapping rule that the p-value of the gene equals to the SNP with highest p-value. The result should be procedded as the same format as **sorted_IGAP.csv**, which is the example mapping output.


# GDAMDB
The purpose of this repository is to demonstrate the workflow of 
GDAMDB and NOT to implement it note to note and at the same time I will
 try not to miss out on the major bits discussed in the paper.
 For that matter, I'll be using the Flowers dataset.

![avatar](picture/workflow.png)

## Extract mutation triples from PubMed: 
To extract mutation type fdg from a targeted text resource in terms of a specific disease d.  
**BERT_multi_task/** contains the Mutation Type Retireval Module, which is to extract mutation triples from PubMed.  

The input file is the BIO format file output from **json2bio.py**, **data/BIO_example.txt**. After runing the the Mutation Type Retireval Module, the result is outputed to **output/**, where there is an example output file **pubtator_Alzheimer'sdisease.txt** whith the format of *(GeneID LOF/GOF/COM/REG DiseaseMeshID)*.

*Note:   
LOF -- loss of function mutation;   
GOF -- gain of function mutation;   
COM -- complex, loss of function mutation and gain of function mutation;   
REG -- mutation without direction.  
In the subsequent data fusion process, only GOF mutations and LOF mutations are utilized.*  

**Usage:**  
`bash zky_run_join_multi.sh`

 
## Synchronization Filter: 
To filter top n genes g which shows significance both in literature and GWAS research. 
AGACGraphicalModel/generate_IGAP.py
input: MutationAssociationData/GWASsummaryData_example.txt 
       pubtator_Alzheimer'sdisease.txt/// pubtator_example.txt
       
output: IGAP.txt
 
## Mutation Data Bridging: 
To bridge all pdg and fdg and predict new gene disease associations.  
AGACGraphicalModel/inference_fusion_vvv.py

