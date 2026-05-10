# NeST-VNN: A visible neural network model for drug response prediction
NeST-VNN is an interpretable neural network-based model that predicts
cell response to a drug. the first explainable data-driven method 
for cancer therapeutic response prediction, in which cell structure 
is modeled using a hierarchical map of tumor cell systems.
This framework integrates information across multiple levels of 
cancer cell biology to understand drug response, and can serve 
to identify and explain biomarkers for clinical application.

NeST-VNN characterizes each cell line using its genotype;
the feature vector for each cell is a binary vector representing
mutational status and copy number variations of the genes 
used in clinical panels like Foundation Medicine (n=718).

Related publications (please cite both if you use the repo):
1) Park, S., Silva, E., Singhal, A. et al. A deep learning model of tumor cell architecture elucidates response and resistance to CDK4/6 inhibitors. Nat Cancer (2024). https://doi.org/10.1038/s43018-024-00740-1
2) Zhao, Singhal, et al. Cancer Mutations Converge on a Collection of Protein Assemblies to Predict Resistance to Replication Stress. Cancer Discov 1 March 2024; 14 (3): 508–523. https://doi.org/10.1158/2159-8290.CD-23-0641

# Environment set up for training and testing

# Example using CBioPortal data
1. Download from cBioPortal
```
python scripts/cbioportal_download.py breast_msk_2025
```

2. Transform to NeST-VNN format (interactive endpoint selection)
```
python scripts/transform_to_nest_vnn.py breast_msk_2025
```

3. Train
```
bash scripts/train.sh breast_msk_2025 binary_os_status binary
```

4. Predict (generates hidden embeddings for explainability)
```
bash scripts/predict.sh breast_msk_2025 binary_os_status binary
```

5. Annotate hierarchy
```
bash scripts/annotate.sh breast_msk_2025 binary_os_status binary 4
```