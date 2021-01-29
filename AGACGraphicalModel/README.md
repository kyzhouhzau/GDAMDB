This folder contains the Synchronization Filter and Mutation Data Bridging model.

**Preparation:**
Move the literature extraction result **pubtator_Alzheimer'sdisease.txt** and the GWAS processed result **sorted_IGAP.CSV** to **./data/**

**Synchronization Filter**   
To filter top n genes g which shows significance both in literature and GWAS research.   
usage:
`python generate_IGAP.py [threshold] [use_p_norm] [norm_value] [file_name]`
**\[threshold]:** The threshold of p-value. The Gene with greater p-value of the threshold will be saved;  
**\[use_p_norm]:** True, p-value greater than threshold are assigned with 0.5;  
**\[norm_value]:** False, p-value unchange;  
**\[file_name]:** sorted_IGAP.csv ,the processed GWAS file.  
The output file is located in **data/sorted_IGAP.txt**

**Mutation Data Bridging model**
usage:  
`python inference_fusion_vvv.py [initlambda] [threshold] [times] [filter_count] [rounders] [hidden_factors] [inputfile] [outputfolder]`  
**\[initlambda]:** 240, the hyper-parameter;  
**\[threshold]:** 5e-8, The threshold of p-value;  
**\[times]:** 500, xxx;  
**\[filter_count]:**  
**\[rounders]:**  100, the number of the iteration rounds.
**\[hidden_factors]:**  
**\[inputfile]:** data/sorted_IGAP.txt, the result file from Synchronization Filter.
**\[outputfolder]:** generate/IGAP_Wilcoxon/, the outputfolder.



