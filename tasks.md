Tasks:

1. Fix the errors: 
a.	Repo Path: src/text_loader/loader.py
b.	The data loader module is responsible for downloading the dataset and performing the required pre-processing steps 
c.	It’s currently failing, and the user has to fix the tests to get it to pass.

2. Train Machine Learning model: 
a.	Train Model: Develop a machine learning model capable of classifying political parties based on the provided dataset. You can create notebook for training
b.	Lifecycle Management: After training, version your models using databricks ecosystem, to manage the model lifecycle efficiently. Save models to unity catalog
Tip: Use instance of class test_loader.loader.Dataloader to get processed data

3. Create batch training workflow using Databricks ecosystem:
a.	Create databricks Training workflows 
b.	Use feature store / unity delta tables for for storing featurisation outputs
c.	use mlflow for experiment tracking and model registration. Save models to unity catalog
Tip: Workflow Tasks - loader/featurisation -> training and model registration


4. Create model-inference-endpoint: 
Create Databricks serving endpoint for the models that you have versioned to get a Prediction. 
Tip: you can Use Databricks serving UI or deploy programmatically

5. Batch inference workflow: 
a.	Create batch inference workflow which can either use databricks serving endpoint OR registered Model for batch inference. Use same data for inference


USE mlflow==2.22
