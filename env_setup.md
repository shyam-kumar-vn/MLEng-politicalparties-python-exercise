Databricks Setup:

1.	Login to workspace by clicking on the Databricks India Chicklet in your okta
2.	Workspace : Get below repo into your home folder.
3.	Compute: Create compute using “Min Cluster Policy MLE”. Choose ML DBR
4.	Catalog: Create Schema with your name under catalog: mle_batch_catalog_2025_q2. Add all your table , models under this schema only
5.	While creating workflows, model serving endpoint, mlflow experiments; prefix with mle_your-name eg. mle_nikhil_mane_training
