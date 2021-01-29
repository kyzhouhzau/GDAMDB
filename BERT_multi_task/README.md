This folder contains the Mutation Type Retireval Module, which is to extract mutation triples from PubMed.

The input file is the BIO format file output from **json2bio.py**, **data/BIO_example.txt**. After runing the the Mutation Type Retireval Module, the result is outputed to **output/**, where there is an example output file **pubtator_Alzheimer'sdisease.txt** whith the format of *(GeneID LOF/GOF/COM/REG DiseaseMeshID)*.

*Note:   
LOF -- loss of function mutation;   
GOF -- gain of function mutation;   
COM -- complex, loss of function mutation and gain of function mutation;   
REG -- mutation without direction.  
In the subsequent data fusion process, only GOF mutations and LOF mutations are utilized.*  

**Usage:**  
`bash zky_run_join_multi.sh`
