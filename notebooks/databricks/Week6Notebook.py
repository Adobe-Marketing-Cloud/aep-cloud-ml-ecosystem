# Databricks notebook source


# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **Databricks Gated Public Preview**
# MAGIC
# MAGIC This feature is in Public Preview in the following regions: eu-west-1, us-east-1, us-east-2, us-west-2, ap-southeast-2.
# MAGIC
# MAGIC To sign up for access, [fill out this form](https://docs.google.com/forms/d/1wV5JxbFwyjxFJ9V4ZF4PinSxlIZCO0gtjInuHUNwSr8/viewform?edit_requested=true).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The goal of this notebook is to showcase how you can set up, in your own environment,
# MAGIC a [monitoring pipeline](https://docs.databricks.com/en/lakehouse-monitoring/index.html) 
# MAGIC to keep track of drift metrics and data profile statistics against
# MAGIC the model we've just created.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week6-LakehouseMonitoringOverview.png)
# MAGIC
# MAGIC We'll go through a couple of different steps:
# MAGIC - Committing to predictions based on our propensity threshold.
# MAGIC - Joining in ground truth labels as they're collected (simulated for now).
# MAGIC - Creating the monitoring tables and associated dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup
# MAGIC
# MAGIC As with the previous notebooks, there are several configuration steps that
# MAGIC need to be taken before we can proceed. These are all encapsulated in our 
# MAGIC [Common Include]($./CommonInclude) notebook, so let's run that first.

# COMMAND ----------

# MAGIC %run ./CommonInclude

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Predict Based on our Propensity Threshold
# MAGIC
# MAGIC First, we need the propensity threshold we determined in the prior notebook. 
# MAGIC The monitoring tools need predictions to evaluate model performance, rather than
# MAGIC the probabilities we've been calling predictions up until this point. This should
# MAGIC match the value we used to create our audience. We want the lower threshold. We didn't
# MAGIC include those above the upper threshold in the audience because it doesn't make since to
# MAGIC target people who we already think are sure to convert. However, from a model perspective,
# MAGIC those are still people who we think will subscribe.
# MAGIC
# MAGIC

# COMMAND ----------

propensity_lower_threshold = 0.2

# COMMAND ----------

scored_table_name = "propensity_model_output"
prediction_df = (
    spark.table(scored_table_name)
    .withColumnRenamed("prediction", "propensity")
    .withColumn("prediction", F.when(F.col("propensity") > propensity_lower_threshold, 1).otherwise(0)))
display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Join in Ground Truth Labels
# MAGIC
# MAGIC As we collect our ground truth labels, we can reflect those in our monitored results. 
# MAGIC Outside of a simulation, we would define whatever criteria we saw fit to determine at
# MAGIC which time we'd say whether the campaign was effective or not and let the state of the 
# MAGIC subscriber at that time go from NULL to 0 or 1 as the case may be. Those would then be
# MAGIC left joined into the monitor table, so that the predictions are revealed over time.
# MAGIC Rather than attempt to simulate that here, we cheat a little bit and just join back in
# MAGIC our original labels from the initial data generation notebook in week 1.
# MAGIC
# MAGIC Note that for performance reasons, you'll want change data feed enabled on this and
# MAGIC any other table you want to enable for monitoring.

# COMMAND ----------

ground_truth_df = (
    spark.table("user_propensity_features")
    .select("userId", "subscriptionOccurred"))

labeled_df = (
    prediction_df
    .join(ground_truth_df, ["userId"], "left"))

# if a baseline hasn't already been captured, let's create one
baseline_table_name = "propensity_model_output_baseline"
if not spark.catalog.tableExists(baseline_table_name):
    labeled_df.write.saveAsTable(baseline_table_name)

labeled_table_name = "propensity_model_output_with_labels"
monitored_table_name = f"{catalog_name}.{database_name}.{labeled_table_name}"

monitored_df = (
    labeled_df
    .withColumn("model_version", F.col("model_version").cast("string"))
    .withColumn("prediction", F.col("prediction").cast("string"))
    .withColumn("subscriptionOccurred", F.col("subscriptionOccurred").cast("string"))
)

(monitored_df.write
    .option("delta.enableChangeDataFeed", "true")
    .mode("append")
    .saveAsTable(labeled_table_name))

monitored_df = spark.table(labeled_table_name)

display(monitored_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. Create the Monitor (incl. Tables and Dashboard)
# MAGIC
# MAGIC Now we can create the monitor. This will create the background object which represents the overall
# MAGIC monitoring object itself, along with the associated profile metrics and drift metrics tables. It
# MAGIC also triggers generation of the DBSQL dashboard that helps us visualize the content of those two
# MAGIC monitoring tables to better understand our model performance.
# MAGIC
# MAGIC You can find out more about this method and others in Databricks Lakehouse Monitoring 
# MAGIC [API docs](https://api-docs.databricks.com/python/lakehouse-monitoring/latest/databricks.lakehouse_monitoring.html#databricks.lakehouse_monitoring.create_monitor).

# COMMAND ----------

from databricks import lakehouse_monitoring as lm

info = lm.create_monitor(
    table_name=monitored_table_name,
    profile_type=lm.InferenceLog(
        problem_type="classification",
        prediction_col="prediction",
        timestamp_col="timestamp",
        granularities=["1 day", "1 month"],
        model_id_col="model_version",
        label_col="subscriptionOccurred",
    ),
    output_schema_name=f"{catalog_name}.{database_name}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC It takes a minute or two for the monitor to be created.

# COMMAND ----------

import time

# Wait for monitor to be created
while info.status == lm.MonitorStatus.PENDING:
  info = lm.get_monitor(table_name=monitored_table_name)
  time.sleep(10)

assert(info.status == lm.MonitorStatus.ACTIVE)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Once its created, it will take another 5 to 10 minutes for the initial refresh job to complete.
# MAGIC When its done, you can find it in the quality tab for our monitored table in Unity Catalog.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week6-QualityTab.png)

# COMMAND ----------

# A metric refresh will automatically be triggered on creation
refreshes = lm.list_refreshes(table_name=monitored_table_name)
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (lm.RefreshState.PENDING, lm.RefreshState.RUNNING):
  run_info = lm.get_refresh(table_name=monitored_table_name, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert(run_info.state == lm.RefreshState.SUCCESS)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Once that's complete, you can view the monitoring dashboard and explore the tables
# MAGIC via the links above.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week6-DashboardTopScreenshot.png)
# MAGIC
# MAGIC Since we just created it, most of the time series oriented plots will just have a single point, if
# MAGIC we produce additional inferences over time and update the monitor then we'd see a shift.

# COMMAND ----------

# MAGIC %md
# MAGIC # Next Steps
# MAGIC
# MAGIC From here you could extend the monitor with additional custom metrics. For instance, in 
# MAGIC addition to the label, we could also capture additional business metrics and include
# MAGIC those on the generated dashboards. We could also use the model id for online evaluation
# MAGIC of a champion model vs. a challenger model and run other types of A/B tests and whatever
# MAGIC experiments we wanted to pursue from a business perspective.
# MAGIC
# MAGIC Another handy thing we can do with the monitoring infrastructure as it's configured in 
# MAGIC DBSQL is set up automated alerts. We can use these to send us emails based on whatever
# MAGIC thresholds we defined, send Slack messages, or even trigger new retraining jobs or other
# MAGIC automated workflows.
