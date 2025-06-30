# Tweet Prediction

The election is coming! We want to build an application that can parse a Tweet and determine which political party to associate with the Tweet, Republican or Democrat. As a consulting machine learning engineer you are tasked with taking the initial efforts of building the application and making it functional.  

This repository contains the necessary setup and code base to help guide you in creating this application.

## 🚀 **Latest Updates**

- **Model**: Using **Logistic Regression** for political party classification
- **Production Ready**: Full ML pipeline with training, deployment, and batch inference workflows
- **Enhanced Pipeline**: Complete workflow with model promotion and evaluation

## 🏗️ **Architecture Overview**

The project implements a complete ML pipeline using:
- **Logistic Regression** for model training and inference
- **Databricks Unity Catalog** for model management
- **MLflow** for experiment tracking and model versioning
- **Delta Lake** for feature storage
- **FastAPI** for serving endpoints
- **Batch inference** workflows for scalable predictions

# Project Setup

## Pre-requisites

Please make sure you have the following software installed

- Python (3.10 or 3.11)

## Install all python dependencies

Before you install dependencies make sure you add your python path to the makefile.

```bash
make install
```

## Run tests & checks

The unit tests can be run either by using the make commands (given in the makefile) or by using the commands from the respective packages.
For example, unit tests can be executed using,

```bash
make test
```

**Note: this command should execute with failed unit tests**

# Gearing Up for the Pairing Session

Please be sure to complete the below tasks before the pairing session.

1. Get a high-level understanding of the dataset
2. Have your coding environment ready by installing python and dependencies.
3. Ensure that you are able to run all commands mentioned in this README (except for pytest errors)

**Please note that you DO NOT have to complete the code/tasks inside the `src/` folder. It is meant to be done together during a pairing session.**
