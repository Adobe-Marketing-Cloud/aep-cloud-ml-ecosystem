# Databricks notebook source
# MAGIC %md
# MAGIC # Scope of Notebook
# MAGIC
# MAGIC The goal of this notebook is to use the propensity data that is now part of our profiles to create some segments for targeting high-value users, and to activate this audience to a 3rd party ad platform. 
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week5-Workflow.png)
# MAGIC
# MAGIC We'll go through several steps:
# MAGIC - Determining an appropriate **propensity interval** by analyzing the scoring output.
# MAGIC - Creating a **segment** based on the propensity scores.
# MAGIC - Activating that segment into a **destination**.

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup
# MAGIC
# MAGIC As with the previous notebook, this notebook requires some configuration data to properly authenticate to your Adobe Experience Platform instance. You should be able to find all the values required above by following the Setup section of the **README**.
# MAGIC
# MAGIC The next cell will be looking for your configuration file under your **ADOBE_HOME** path to fetch the values used throughout this notebook. See more details in the Setup section of the **README** to understand how to create your configuration file.
# MAGIC
# MAGIC Some imports and utility functions that will be used throughout this notebook are provided in the [Common Include]($./CommonInclude) notebook since they'll also be used in all the other notebooks. Also, if you haven't already done so, please go run the [RunMe Notebook]($./RunMe) the create a cluster that has the required libraries installed.

# COMMAND ----------

# MAGIC %run ./CommonInclude

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Creating a Segment to Target Users based on Propensities
# MAGIC
# MAGIC Because the propensity data is already in the Unified Profile, we can now create a segment to target people based on propensities. But it's not immediately obvious what a good value for the upper and lower bound of our target audience should be, so we need to look at the scoring data a bit to understand it better.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Reading the scored data
# MAGIC
# MAGIC To that end we need to read the output of the scoring data that we wrote to the Data Landing Zone previously. We use the regular container `dlz-user-container`, since this is where we wrote the data.

# COMMAND ----------

from aepp import flowservice

flow_conn = flowservice.FlowService()

# Note that this overrides the general DLZ destination container defined in CommonInclude.
dlz_credentials = flow_conn.getLandingZoneCredential()
dlz_container = dlz_credentials["containerName"]
dlz_sas_token = dlz_credentials["SASToken"]
dlz_storage_account = dlz_credentials["storageAccountName"]
dlz_sas_uri = dlz_credentials["SASUri"]
print(f"Reading from container {dlz_container}")

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we're ready to read the data. We're using Spark since it could be pretty large as we're not doing any sampling. Spark needs the following properties to be able to authenticate using SAS:
# MAGIC - `fs.azure.account.auth.type.$ACCOUNT.dfs.core.windows.net` should be set to `SAS`.
# MAGIC - `fs.azure.sas.token.provider.type.$ACCOUNT.dfs.core.windows.net` should be set to `org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider`.
# MAGIC - `fs.azure.sas.fixed.token.$ACCOUNT.dfs.core.windows.net` should be set to the SAS token retrieved earlier.
# MAGIC
# MAGIC Let's put that in practice and create a Spark dataframe containing the entire featurized data:

# COMMAND ----------

from adlfs import AzureBlobFileSystem
from fsspec import AbstractFileSystem

def read_remote_scores():
    azure_blob_fs = AzureBlobFileSystem(account_name=dlz_storage_account, sas_token=dlz_sas_token)
    export_time = get_export_time(azure_blob_fs, dlz_container, import_path, scoring_dataset_id)
    print(f"Using featurized data export time of {export_time}")

    spark.conf.set(f"fs.azure.account.auth.type.{dlz_storage_account}.dfs.core.windows.net", "SAS")
    spark.conf.set(f"fs.azure.sas.token.provider.type.{dlz_storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
    spark.conf.set(f"fs.azure.sas.fixed.token.{dlz_storage_account}.dfs.core.windows.net", dlz_sas_token)

    protocol = "abfss"
    input_path = f"{protocol}://{dlz_container}@{dlz_storage_account}.dfs.core.windows.net/{import_path}/{scoring_dataset_id}/exportTime={export_time}/"

    remote_scoring_df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(input_path))
    
    return remote_scoring_df

# COMMAND ----------

# MAGIC %md
# MAGIC Before we start working with it, let's ingest it as a Delta table and use that as the basis for our subsequent analysis. We'd only need to do this process once.

# COMMAND ----------

scoring_table_name = "week5_scoring_input"

if not spark.catalog.tableExists(scoring_table_name):
    remote_scoring_df = read_remote_scores()
    remote_scoring_df.write.saveAsTable(scoring_table_name)

df = spark.table(scoring_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC We can verify it matches what we had written out in the second weekly assignment:

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC And also do a sanity check on the data to make sure it looks good:

# COMMAND ----------

df = df.fillna(0)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1.2 Extra synthetic data visualizations
# MAGIC
# MAGIC Let's take a moment here to also think about how we can go about joining prediction output 
# MAGIC from AEP with data we may have available to us outside of AEP. We created some extra synthetic data
# MAGIC to go along with the main in our Week3 Notebook. For instance, let's do a few examples to understand 
# MAGIC if there are any correlations between highly likely subscribers and the other features associated 
# MAGIC with these profiles.

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import functions as F

# Read in extra synthetic data table
extra_synth_data = spark.table("extra_synthetic_data")

# Read in propensity features table
propensity_features = spark.table("user_propensity_features")

# Join predictions from AEP to local features and other data outside of AEP
extra_synth_df = (
    df
    .join(spark.table("extra_synthetic_data"), "userId")
    .join(spark.table("user_propensity_features"), "userId"))

display(extra_synth_df)

# COMMAND ----------

# MAGIC %md
# MAGIC To further demonstrate the idea of combining these datasets, let's create a heatmap to 
# MAGIC investigate some of the relationships in the data.

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Here, we'll demonstrate processing this visualization using Pandas
full_df = extra_synth_df.toPandas()

# Group by 'DeviceType' and calculate the mean of 'Predicted Subscription status'
grouped_data = full_df.groupby(['eventType', 'devicePlatform'], as_index=False).agg({'prediction': 'mean'})

# Create a DataFrame suitable for a heatmap
heatmap_data = grouped_data.pivot('eventType', 'devicePlatform', 'prediction')

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Avg Likeliness to Subscribe'})

plt.title('Likeiness to Subscribe Based on Device Type Used for Digital Interactions')
plt.xlabel('Device Type')
plt.ylabel('Interaction Type');

# COMMAND ----------

# MAGIC %md
# MAGIC We can also look at how subscription likeliness varies across states if we wanted to consider geo-targetting in the future. Based off the visual below you can see VT and SC have the highest concentration of likeliness to subscribe.

# COMMAND ----------

import plotly.graph_objects as go

grouped_by_state = full_df.groupby(['state'], as_index=False).agg({'prediction': 'mean'})

fig = go.Figure(data=go.Choropleth(
    locations = grouped_by_state['state'],
    z = grouped_by_state['prediction'],
    locationmode = 'USA-states', 
    colorscale = 'Blues',
    colorbar_title = 'Likeliness to Subscribe'
)) 

fig.update_layout(
    title = 'Likeliness to Subscribe Prediction by State',
    geo_scope = 'usa'
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There are a variety of different data sources you can bring to the table by taking
# MAGIC advantage of the combined capabilities of AEP and Databricks in this way. This 
# MAGIC particular example just scratches the surface of what's possible to hopefully
# MAGIC inspire your own combinations of datasets to amplify your AI powered marketing
# MAGIC campaigns. Now, let's move on to take a look at how to analyze the predictions to
# MAGIC choose thresholds for our advertising segments.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Figuring out the right threshold via a propensity-reach graph
# MAGIC
# MAGIC In order to determine a suitable interval of propensities for targeting, we need to understand the distribution of the propensity scores across all our profiles.  There's a few different ways we can digest that information, but the very first step is to create a histogram with N bins. Because we may have scored a lot of profiles we do this computation via Spark's `histogram_numeric` function to make sure it is distributed.

# COMMAND ----------

num_buckets = 20
df_histogram = (
    df.selectExpr(f"explode(histogram_numeric(prediction, {num_buckets})) as histogram")
    .selectExpr("round(histogram.x, 2) as propensity_bucket", "histogram.y as reach"))
df_histogram.printSchema()
display(df_histogram)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a histogram it should only be a few rows of data, with each row being a bin, so we can just convert it to a `pandas` dataframe to bring it back locally.

# COMMAND ----------

df_graph = df_histogram.toPandas()
display(df_graph.head())

# COMMAND ----------

# MAGIC %md
# MAGIC We can just plot the histogram as-is, and each point will show us how many profiles are in that particular propensity bin.

# COMMAND ----------

df_graph.plot(x="propensity_bucket", y="reach", title="Number of profiles at a propensity bucket");

# COMMAND ----------

# MAGIC %md
# MAGIC This is still not quite what we want, because typically we'll want to target profiles who have a propensity either above or below a particular threshold. We can get that by computing the **cumulative sum** using two different methods:
# MAGIC - If we do the cumulative sum from the **smallest bucket to the largest bucket** ("left to right"), then any point in the resulting graph shows us the reach if we target all profiles with a propensity **below** a particular threshold.
# MAGIC - If we do the cumulative sum from the **largest bucket to the smallest bucket** ("right to left"), then any point in the resulting graph shows us the reach if we target all profiles with a propensity **above** a particular threshold.

# COMMAND ----------

df_graph["reach_inferior_or_equal"] = df_graph["reach"].cumsum()
df_graph["reach_superior_or_equal"] = df_graph.loc[::-1, "reach"].cumsum()[::-1]
df_graph["reach_inferior"] = df_graph["reach_inferior_or_equal"].shift(1)
df_graph["reach_superior"] = df_graph["reach_superior_or_equal"].shift(-1)
display(df_graph)

# COMMAND ----------

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df_graph.plot(x="propensity_bucket", y="reach_superior_or_equal", 
              title="Reach above a propensity", ax=ax[0])
df_graph.plot(x="propensity_bucket", y="reach_inferior_or_equal", 
              title="Reach below a propensity", ax=ax[1]);

# COMMAND ----------

# MAGIC %md
# MAGIC These graphs are useful to help us define broad propensity-based segments where we only look at profiles with a propensity above or below a threshold. However, that's still not enough to help us define a complete interval for our segment.
# MAGIC
# MAGIC For that, we would ideally like to represent this as a 3-dimensional plot where:
# MAGIC - On the **X axis** we have the **lower bound** of the interval.
# MAGIC - On the **Y axis** we have the **upper bound** of the interval.
# MAGIC - On the **Z axis** you have the **reach** corresponding to that interval.
# MAGIC
# MAGIC To get there, the first step is to create a function that can tell us the reach given a lower and upper bound:

# COMMAND ----------

df_indexed = df_graph.fillna(0).set_index("propensity_bucket")
display(df_indexed)

# COMMAND ----------

def reach_between(df, propensity_from, propensity_to, total_pop):
    if propensity_from > propensity_to:
        return 0.0
    return (
        total_pop
        - df.loc[propensity_from]["reach_inferior"]
        - df.loc[propensity_to]["reach_superior"])

total_population = df_graph["reach"].sum()
total_population

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can create a 2-dimensional array which will represent the reach for each of the bins of the interval. For that we simply iterate over all the bins in both dimensions, and pass it to our reach computation function.

# COMMAND ----------

import numpy as np

dim = len(df_indexed)

z = np.zeros((dim, dim))
x = df_graph["propensity_bucket"].values
for index_from, propensity_from in enumerate(x):
    for index_to, propensity_to in enumerate(x):
        reach_x_y = reach_between(df_indexed, propensity_from, propensity_to, total_population)
        z[index_from][index_to] = reach_x_y

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can make the 3-dimensional plot as a surface plot to visualize things easily in an interactive plot. We also want to generate the corresponding segment rule on the fly for each point, so for that we need to first get the tenant ID since the propensity XDM field was nested under the tenant ID.

# COMMAND ----------

from aepp import schema

schema_conn = schema.Schema()

tenant_id = schema_conn.getTenantId()
tenant_id

# COMMAND ----------

import plotly.graph_objects as go

titlecolor = "black"
bgcolor = "white"

layout = go.Layout(
    autosize=False,
    width=1500,
    height=800,
    title="Propensity-reach Segment Topology",
    showlegend=True,
    scene=dict(
        xaxis_title_text="Propensity To",
        yaxis_title_text="Propensity From",
        zaxis_title_text="Reach",
        aspectmode="manual",
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1),
    ),
    paper_bgcolor=bgcolor,
    plot_bgcolor=bgcolor,
)

trace = go.Surface(
    x=x,
    y=x,
    z=z,
    hovertemplate=f"""
  Propensity From: %{{y}}<br>
  Propensity To: %{{x}}<br>
  Reach: %{{z}}<br>
  Rule: _{tenant_id}.propensity >= %{{y}} and _{tenant_id}.propensity <= %{{x}}
  """,
)

fig = go.FigureWidget(data=[trace], layout=layout)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC You can now use this plot to find an interval that looks interesting. For example we can take a point at the inflexion point so we can target profiles with a decently high propensity without necessarily catching our entire set of profiles, as can be seen below:
# MAGIC
# MAGIC ![Surface](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week5-Surface.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Targeting via a Propensity Segment
# MAGIC
# MAGIC Now that we know the characteristics of the audience we want to target, the next and final step is to turn this into an actual audience, make sure it is populated, and activate it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Creating a Propensity Segment
# MAGIC
# MAGIC We've identified the upper and lower bound of interest in the previous 3-dimensional plot, so at that point we can copy/paste the corresponding segment rule and plug it into the cell below, so we can use this as the basis for our segment:

# COMMAND ----------

segment_rule = f"_{tenant_id}.propensity >= 0.2 and _{tenant_id}.propensity <= 0.93"

# COMMAND ----------

def get_segment_by_name(segment_conn, name):
    for segment in segment_conn.getSegments():
        if segment["name"] == name:
            return segment
    return None

def get_or_create_segment(segment_conn, segment_spec):
    name = segment_spec["name"]
    segment_res = get_segment_by_name(segment_conn, name)
    if segment_res is None:
        segment_res = segment_conn.createSegment(segment_spec)
    return segment_res

# COMMAND ----------

from aepp import segmentation

segment_conn = segmentation.Segmentation()

segment_spec = {
    "name": f"[CMLE][Week5] People with a moderate-to-high propensity to subscribe (created by {username})",
    "profileInstanceId": "ups",
    "description": "People who have a moderate-to-high propensity to subscribe",
    "expression": {"type": "PQL", "format": "pql/text", "value": segment_rule},
    "schema": {"name": "_xdm.context.profile"},
    "payloadSchema": "string",
    "ttlInDays": 60,
}

segment_res = get_or_create_segment(segment_conn, segment_spec)

segment_id = segment_res["id"]
segment_link = get_ui_link(tenant_id, "segment/browse", urllib.parse.quote(segment_id, safe="a"))
display_link(segment_link, segment_spec["name"])

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the segment has been created, but it does not mean it will get populated in realtime. If you've clicked in the UI on `Add all segments to schedule` it should be evaluated and populated eventually (up to 24 hours). 
# MAGIC
# MAGIC If you do not want to wait for that you can trigger a segmentation job on-demand just for this segment:

# COMMAND ----------

job_res = segment_conn.createJob([segment_id])
job_id = job_res["id"]
job_id

# COMMAND ----------

# MAGIC %md
# MAGIC This can still take a few minutes to run so we just keep checking the status and wait for the segmentation job to complete:

# COMMAND ----------

import time

finished = False
while not finished:
    job_info = segment_conn.getJob(job_id)
    job_status = job_info["status"]
    if job_status in ["SUCCEEDED", "FAILED"]:
        total_time = job_info["metrics"]["totalTime"]["totalTimeInMs"] / 1000
        qualified_profiles = job_info["metrics"]["segmentedProfileCounter"][segment_id]
        print(f"Segmentation job completed in {total_time} secs with {qualified_profiles} profiles")
        break
    print(f"Job not yet finished, status is {job_status}")
    time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC After the segmentation job is complete for that segment, you should be able to see the population of your segment reflected accurately in the UI at the link below. In our example, based on the propensity interval we selected earlier, we can verify it matches the reach we expected from the 3-dimensional plot:

# COMMAND ----------

segment_link = get_ui_link(tenant_id, "segment/browse", segment_id)
display_link(segment_link, f"Segment ID {segment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ![Segment](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week5-Segment.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Activating the Segment
# MAGIC
# MAGIC Now we're ready to activate the segment and the profiles associated to it to a destination. This step is more easily accomplished via the UI, and you can follow [this guide](https://experienceleague.adobe.com/docs/experience-platform/destinations/ui/activate/activate-batch-profile-destinations.html?lang=en) to go through the different steps needed for activation.
# MAGIC
# MAGIC You will need to choose a destination for it. You can use any pre-defined destination that you might already have setup, or if you need a dummy destination you can again use the Data Landing Zone to simply use it for validation purposes.
