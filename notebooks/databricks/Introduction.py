# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Adobe Bring Your Own ML (BYOML) on Databricks
# MAGIC
# MAGIC Welcome to this example of integrating Adobe Experience Platform (AEP) with Databricks.
# MAGIC In this series of notebooks, we'll walk you through the process of how to create move
# MAGIC data between the two platforms, as well as how to create a custom ML model
# MAGIC so you can improve your advertising results using AI.
# MAGIC
# MAGIC The notebooks are structured according to weeks, to give an indication of the pace you 
# MAGIC might go through them as part of a learning activity or workshop. However, feel free to
# MAGIC move at your own pace if you need more time or less.
# MAGIC
# MAGIC ## Example context
# MAGIC
# MAGIC The scenario used as a theme across these notebooks is that we are executing advertising
# MAGIC campaigns and want to create a new segment to use for advertising campaigns using machine
# MAGIC learning. The types of events we are collecting via AEP include direct mail events, email
# MAGIC events, and various clickstream activities. The final event we're trying to target is a
# MAGIC subscription to our service. The model we'll create is a propensity model which predicts 
# MAGIC a users propensity to subscribe based on various features associated with the events leading
# MAGIC up to a subscription. That model is then used to create our new segment.
# MAGIC
# MAGIC ## Initial setup
# MAGIC
# MAGIC To help you get started, we've provided a [Run Me]($./RunMe) notebook which will configure 
# MAGIC a cluster for you to use, with libraries configured that will be used across the notebooks. 
# MAGIC It also sets up a job in Databricks Workflows that you can use if you want an easy way to
# MAGIC run all the notebooks one after another. Finally, there are some images that are used in 
# MAGIC the notebooks, and this takes care of moving those images to a location from which they
# MAGIC can be displayed in the notebook.
# MAGIC
# MAGIC We also create an initial configuration file which you'll need to populate with several
# MAGIC parameters specific to your AEP instance. Be sure to edit that file to populate it with
# MAGIC the values as explained in the repository's readme file.
# MAGIC
# MAGIC ## Course organization
# MAGIC
# MAGIC ### [Week 1: Generate and ingest dataset]($./Week1Notebook)
# MAGIC
# MAGIC In the first week we demonstrate how to configure the required AEP field group
# MAGIC and schema objects required in preparation of it receiving the dataset. Then, we
# MAGIC generate the dataset as a Delta table on Databricks. Finally, we ingest the data
# MAGIC in batches into a dataset object in the AEP platform, and do some initial analysis
# MAGIC within the platform.
# MAGIC
# MAGIC ### [Week 2: Prepare dataset for ML]($./Week2Notebook)
# MAGIC
# MAGIC During week two, we dive deeper with our analysis and move on to feature engineering.
# MAGIC We show how to create a parameterized query to featurize the event data, and then show
# MAGIC how to set up a flow to move that data into cloud storage via a DLZ.
# MAGIC
# MAGIC ### [Week 3: Train and register ML model in MLflow Model Registry]($./Week3Notebook)
# MAGIC
# MAGIC In week three, we pick up the data from cloud storage and show how we'd ingest features
# MAGIC back into Databricks Feature Store. Then we show how we can further explore the features
# MAGIC from within Databricks using the built-in profiling capabilities. Next, we show how to
# MAGIC train the model both as a one-off run, as well as perform a distributed hyperparameter sweep 
# MAGIC using hyperopt with Spark trials. With a tuned model in hand, we then show how to register
# MAGIC the model in the MLflow Model Registry.
# MAGIC
# MAGIC ### [Week 4: Score model to predict user propensity]($./Week4Notebook)
# MAGIC
# MAGIC Afterwards, in week four, we show how to use the model from the registry to score new 
# MAGIC batches of data to predict their subscription propensity. 
# MAGIC
# MAGIC ### [Week 5: Use model results to create an audience segment]($./Week5Notebook)
# MAGIC
# MAGIC Once the model is created and we've scored a batch of data we have a propensity score
# MAGIC for all our contacts. From there, we can determine a set of thresholds to use to define
# MAGIC and audience to target with our advertising campaigns.
# MAGIC
# MAGIC ### [Week 6: Set up monitoring]($./Week6Notebook)
# MAGIC
# MAGIC Now that our model is working and our pipeline is adjusting our defined audience segment
# MAGIC we need to monitor its performance over time. This notebook shows how you can do that
# MAGIC with the new gated public preview of Databricks Lakehouse Monitoring.
# MAGIC
# MAGIC ## Additional Files 
# MAGIC
# MAGIC In addition to the [Run Me]($./RunMe) notebook, there are a couple of notebooks 
# MAGIC with additional code referenced in the weekly notebooks.
# MAGIC
# MAGIC ### [Common Include]($./CommonInclude)
# MAGIC
# MAGIC This contains some notebook setup code and common functions used across multiple notebooks.
# MAGIC You can check this out to learn how particular helper functions are implemented and for 
# MAGIC details about environment configuration.
# MAGIC
# MAGIC ### [Simulation Helpers]($./SimulationHelpers)
# MAGIC
# MAGIC This course includes logic to simulate user experience events and this notebook contains
# MAGIC all the functions which implements that logic. Check this out for details on how the 
# MAGIC simulation is implemented.
