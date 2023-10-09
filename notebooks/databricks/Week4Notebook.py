# Databricks notebook source
# MAGIC %md
# MAGIC # Scope of Notebook
# MAGIC
# MAGIC The goal of this notebook is to showcase how you can use, in your own environment, a pre-trained model along with some profile data extracted from the Adobe Experience Platform to generate propensity scores and ingest those back to enrich the Unified Profile.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week4-Workflow.png)
# MAGIC
# MAGIC We'll go through several steps:
# MAGIC - **Reading the featurized data** from the Data Landing Zone
# MAGIC - Generating the **scores**
# MAGIC - Creating a **target dataset**
# MAGIC - Creating a **dataflow** to deliver data in the right format to that dataset.

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
# MAGIC # 1. Generating Propensity Scores

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Reading the Featurized Data from the Data Landing Zone
# MAGIC
# MAGIC In the second weekly assignment we had written our featurized data into the Data Landing Zone, and then on the third assignment we just read a sampled portion of it for training our model. At that point we want to score all of the profiles, so we need to read everything.
# MAGIC
# MAGIC The featurized data exported into the Data Landing Zone is under the format **cmle/egress/$DATASETID/exportTime=$EXPORTTIME**. We know the dataset ID which is in your config under `featurized_dataset_id` so we're just missing the export time so we know what to read. To get that we can simply list files in the DLZ and find what the value is.
# MAGIC
# MAGIC Now we use some Python libraries to authenticate and issue listing commands so we can get the paths and extract the time from it.

# COMMAND ----------

from adlfs import AzureBlobFileSystem
from fsspec import AbstractFileSystem

azure_blob_fs = AzureBlobFileSystem(account_name=dlz_storage_account, sas_token=dlz_sas_token)

export_time = get_export_time(azure_blob_fs, dlz_container, export_path, featurized_dataset_id)
print(f"Using featurized data export time of {export_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we're ready to read this data. We're using Spark since it could be pretty large as we're not doing any sampling. Spark needs the following properties to be able to authenticate using SAS:
# MAGIC - `fs.azure.account.auth.type.$ACCOUNT.dfs.core.windows.net` should be set to `SAS`.
# MAGIC - `fs.azure.sas.token.provider.type.$ACCOUNT.dfs.core.windows.net` should be set to `org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider`.
# MAGIC - `fs.azure.sas.fixed.token.$ACCOUNT.dfs.core.windows.net` should be set to the SAS token retrieved earlier.
# MAGIC
# MAGIC Let's put that in practice and create a Spark dataframe containing the entire featurized data:

# COMMAND ----------

spark.conf.set(f"fs.azure.account.auth.type.{dlz_storage_account}.dfs.core.windows.net", "SAS")
spark.conf.set(f"fs.azure.sas.token.provider.type.{dlz_storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set(f"fs.azure.sas.fixed.token.{dlz_storage_account}.dfs.core.windows.net", dlz_sas_token)

protocol = "abfss"
input_path = f"{protocol}://{dlz_container}@{dlz_storage_account}.dfs.core.windows.net/{export_path}/{featurized_dataset_id}/exportTime={export_time}/"

dlz_input_df = spark.read.parquet(input_path).na.fill(0)
dlz_input_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC We can verify it matches what we had written out in the second weekly assignment:

# COMMAND ----------

dlz_input_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC And also do a sanity check on the data to make sure it looks good:

# COMMAND ----------

display(dlz_input_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Since we already created the Feature Store table previously, here we just need to
# MAGIC update and that we need to now with our new data to prepare it for scoring. 
# MAGIC To do that, we'll now use the fresh data from the DLZ to write to the feature 
# MAGIC store so that we merge in any inserts and updates as a single Delta transaction.

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

# We need an instance of the feature store client to use the API.
fs = feature_store.FeatureStoreClient()
feature_table_name = "user_propensity_features"

# Merge the new data into the feature table
fs.write_table(name=feature_table_name, df=dlz_input_df)

# Compose and display a link to the feature table.
feature_table_link = f"/explore/data/{catalog_name}/{database_name}/{feature_table_name}"
display_link(feature_table_link, "User Propensity Features in Unity Catalog")

# Finally, let's look at a dataframe read in from the feature table.
feature_df = fs.read_table(feature_table_name)
display(feature_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Scoring the Profiles

# COMMAND ----------

# MAGIC %md
# MAGIC For scoring we need 2 things:
# MAGIC 1. The batch of **data** to score (primary keys, along with any features not in the feature table).
# MAGIC 2. The URI of the **trained model** that will be used to do the scoring.
# MAGIC
# MAGIC We just created a dataframe containing the first one, and in the previous weekly assignment we created a production model that can operate on this data. Since we used the Feature Store API to log the model, Databricks will automatically take care of looking up the features appropriate for the model. Furthermore, if we decide to shift our feature table to looking at user features over time, it'll even take care of automatically performing the point in time lookup against the feature table.
# MAGIC
# MAGIC If we look at the data we should see a new column called `prediction` which corresponds to the score generated by the model for this particular profile based on all the features computed earlier.
# MAGIC
# MAGIC Note that the `score_batch` function is also handling setting up a Spark UDF for us as well, so the inference is happening across the cluster as it looks up the features from the feature table and applies our model using a vectorized UDF.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from datetime import datetime

client = mlflow.MlflowClient()

current_timestamp = datetime.now()
model_uri = f"models:/{model_name}@Champion"
model_version = client.get_model_version_by_alias(model_name, "Champion")

batch_df = dlz_input_df.select("userId")
scored_df = (
    fs.score_batch(model_uri, batch_df, result_type="array<double>")
    .withColumn("prediction", F.col("prediction")[1])
    .withColumn("model_version", F.lit(model_version.version))
    .withColumn("timestamp", F.lit(current_timestamp)))

scored_table_name = "propensity_model_output"
scored_df.write.mode("overwrite").saveAsTable(scored_table_name)
scored_df = spark.table(scored_table_name)
display(scored_df)

# COMMAND ----------

# MAGIC %md
# MAGIC When you think about bringing the scored profiles back into the Adobe Experience Platform, we don't need to bring back all the features. In fact, we only really need 2 columns:
# MAGIC - The user ID, so we know in the Unified Profile to which profile this row corresponds.
# MAGIC - The score for this user ID.

# COMMAND ----------

from pyspark.sql.functions import (udf, col, lit, create_map, array, struct, current_timestamp)
from itertools import chain

df_to_ingest = scored_df.select("userId", "prediction").cache()
df_to_ingest.printSchema()

# COMMAND ----------

df_to_ingest.count()

# COMMAND ----------

display(df_to_ingest)

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we have the scored profiles and exactly what we need to bring back into Adobe Experience Platform. But we're not quite ready to write the results yet, there's a bit of setup that needs to happen first:
# MAGIC - We need to create and configure a destination **dataset** in Adobe Experience Platform where our data will end up.
# MAGIC - We need to setup a **data flow** that will be able to take this data, convert it into an XDM format, and deliver it to this dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1.3 Inspect Lineage in Unity Catalog
# MAGIC
# MAGIC Since we registered our model in Unity Catalog, our team members will be able to find it
# MAGIC and use it if they're granted access. They'll also be able to determine which tables were
# MAGIC used to build the model and which notebooks were used to produce it, as shown in the
# MAGIC screenshot below.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week4-ModelLineageTable.png)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC By clicking on the `Lineage Graph` button, they can also view the chain 
# MAGIC of dependencies for the model visually.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week4-ModelLineageGraph.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Bringing the Scores back into Unified Profile

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Create ingestion schema and dataset

# COMMAND ----------

# MAGIC %md
# MAGIC The first step is to define where this propensity data we are creating as the output of our model should end up in the Unified Profile. We need to create a few entities for that:
# MAGIC - A **fieldgroup** that will define the XDM for where propensity scores should be stored.
# MAGIC - A **schema** based on that field group that will tie it back to the concept of profile.
# MAGIC - A **dataset** based on that schema that will hold the data.
# MAGIC
# MAGIC As for the structure itself it's pretty simple, we just need 2 fields:
# MAGIC - The **propensity** itself as a decimal number.
# MAGIC - The **user ID** to which this propensity score relates.
# MAGIC
# MAGIC Let's put that in practice and create the field group. Note that because we are creating custom fields here, they need to be nested under the tenant ID corresponding to your organization.

# COMMAND ----------

from aepp import schema

schema_conn = schema.Schema()

tenant_id = schema_conn.getTenantId()
tenant_id

# COMMAND ----------

fieldgroup_spec = {
  	"type": "object",
	"title": f"[CMLE][Week4] Fieldgroup for user propensity (created by {username})",
	"description": "This mixin is used to define a propensity score that can be assigned to a given profile.",
	"allOf": [{
		"$ref": "#/definitions/customFields"
	}],
	"meta:containerId": "tenant",
	"meta:resourceType": "mixins",
	"meta:xdmType": "object",
	"definitions": {
      "customFields": {
        "type": "object",
        "properties": {
          f"_{tenant_id}": {
            "type": "object",
            "properties": {
              "propensity": {
                "title": "Propensity",
                "description": "This refers to the propensity of a user towards an outcome.",
                "type": "number"
              },
              "userid": {
                "title": "User ID",
                "description": "This refers to the user having a propensity towards an outcome.",
                "type": "string"
              }
            }
          }
        }
      }
	},
	"meta:intendedToExtend": ["https://ns.adobe.com/xdm/context/profile"]
}

fieldgroup_res = get_or_create_fieldgroup(schema_conn, fieldgroup_spec)
fieldgroup_id = fieldgroup_res["$id"]
fieldgroup_link = get_ui_link(tenant_id, "schema/mixin/browse", urllib.parse.quote(fieldgroup_id, safe="a"))
display_link(fieldgroup_link, fieldgroup_res['title'])

# COMMAND ----------

# MAGIC %md
# MAGIC From this field group ID we can add it to a brand new schema that will be marked for profiles.

# COMMAND ----------

schema_title = f"[CMLE][Week4] Schema for user propensity ingestion (created by {username})"
schema_res = get_or_create_profile_schema(
    schema_conn,
    title=schema_title,
    mixinIds=[fieldgroup_id],
    description="Schema generated by CMLE for user propensity score ingestion",
)

schema_id = schema_res["$id"]
schema_alt_id = schema_res["meta:altId"]
schema_link = get_ui_link(tenant_id, "schema/mixin/browse", urllib.parse.quote(schema_id, safe="a"))
display_link(schema_link, schema_title)

# COMMAND ----------

# MAGIC %md
# MAGIC Because we eventually intend for these scores to end up in the Unified Profile, we need to specify which field of the schema corresponds to an identity so it can resolve the corresponding profile. In our case, the `userid` field is an ECID and we mark it as such.

# COMMAND ----------

identity_type = "ECID"
descriptor_obj = {
    "@type": "xdm:descriptorIdentity",
    "xdm:sourceSchema": schema_id,
    "xdm:sourceVersion": 1,
    "xdm:sourceProperty": f"/_{tenant_id}/userid",
    "xdm:namespace": identity_type,
    "xdm:property": "xdm:id",
    "xdm:isPrimary": True,
}
descriptor_res = get_or_create_descriptor(schema_conn, descriptor_obj)
descriptor_res

# COMMAND ----------

# MAGIC %md
# MAGIC And of course that schema needs to be enabled for Unified Profile consumption, so it can be added to the profile union schema.

# COMMAND ----------

enable_res = schema_conn.enableSchemaForRealTime(schema_alt_id)
enable_res

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we're ready to create the dataset that will hold our propensity scores. This dataset is based on our schema we just created and nothing more.

# COMMAND ----------

from aepp import catalog

cat_conn = catalog.Catalog()

ingestion_dataset_res = cat_conn.createDataSets(
    name=f"[CMLE][Week4] Dataset for user propensity ingestion (created by {username})",
    schemaId=schema_id,
)

ingestion_dataset_id = ingestion_dataset_res[0].split("/")[-1]
ingestion_dataset_id

# COMMAND ----------

# MAGIC %md
# MAGIC And similarly that dataset needs to be enabled for Unified Profile consumption, so that any batch of data written to this dataset is automatically picked up and processed to insert into the individual profiles and create new fragments.

# COMMAND ----------

# TODO: this is currently failing due to invalid content type, need to fix in aepp, see https://github.com/pitchmuc/aepp/issues/15
# for now just enable in the UI...
cat_conn.enableDatasetProfile(ingestion_dataset_id)

# COMMAND ----------

# MAGIC %md
# MAGIC You should be able to see your dataset in the UI at the link below, and it should match the new schema created as shown in the following screenshot.

# COMMAND ----------

ingestion_dataset_link = get_ui_link(tenant_id, "dataset/browse", ingestion_dataset_id)
display_link(ingestion_dataset_link, f"Dataset ID {ingestion_dataset_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ![Dataset](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week4-ScoringDataset.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Setup ingestion data flow

# COMMAND ----------

# MAGIC %md
# MAGIC Now that all the dataset and schema setup is completed, we're ready to define our Data Flow. The Data Flow defines the contract between the source and destination dataset.
# MAGIC
# MAGIC For the purposes of this notebook we will be using the [Data Landing Zone (DLZ)](https://experienceleague.adobe.com/docs/experience-platform/sources/api-tutorials/create/cloud-storage/data-landing-zone.html?lang=en) as the source filesystem under which the scoring results will be written. Every Adobe Experience Platform has a DLZ already setup as an [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs) container. We'll be using that as a delivery mechanism for the featurized data, but this step can be customized to delivery this data to any cloud storage filesystem.
# MAGIC
# MAGIC To setup the delivery pipeline, we'll be using the [Flow Service for Destinations](https://developer.adobe.com/experience-platform-apis/references/destinations/) which will be responsible for picking up the featurized data and dump it into the DLZ. There's a few steps involved:
# MAGIC - Creating a **source connection**.
# MAGIC - Creating a **target connection**.
# MAGIC - Creating a **transformation**.
# MAGIC - Creating a **data flow**.
# MAGIC
# MAGIC Note that, although we already got DLZ credentials earlier in this notebook, there were for a different container where all the destination data is written (`dlz-destination`), but here we want to get credentials for a different container corresponding to your user drop zone (`dlz-user-container`).
# MAGIC
# MAGIC For that, again we use `aepp` to abstract all the APIs:

# COMMAND ----------

from aepp import flowservice

flow_conn = flowservice.FlowService()

dlz_credentials = flow_conn.getLandingZoneCredential()
dlz_container = dlz_credentials["containerName"]
dlz_sas_token = dlz_credentials["SASToken"]
dlz_storage_account = dlz_credentials["storageAccountName"]
dlz_sas_uri = dlz_credentials["SASUri"]
print(dlz_container)

# COMMAND ----------

# MAGIC %md
# MAGIC The **source connection** is responsible for connecting to your cloud storage account (in our case here, the Data Landing Zone) so that the resulting Data Flow will know from where data needs to be picked up.
# MAGIC
# MAGIC For reference, here is a list of all the connection specs available for the most popular cloud storage accounts (these IDs are global across every single customer account and sandbox):
# MAGIC
# MAGIC | Cloud Storage Type    | Connection Spec ID                   | Connection Spec Name
# MAGIC |-----------------------|--------------------------------------|----------------------
# MAGIC | Amazon S3             | ecadc60c-7455-4d87-84dc-2a0e293d997b | amazon-s3
# MAGIC | Azure Blob Storage    | d771e9c1-4f26-40dc-8617-ce58c4b53702 | google-adwords
# MAGIC | Azure Data Lake       | b3ba5556-48be-44b7-8b85-ff2b69b46dc4 | adls-gen2
# MAGIC | Data Landing Zone     | 26f526f2-58f4-4712-961d-e41bf1ccc0e8 | landing-zone
# MAGIC | Google Cloud Storage  | 32e8f412-cdf7-464c-9885-78184cb113fd | google-cloud
# MAGIC | SFTP                  | b7bf2577-4520-42c9-bae9-cad01560f7bc | sftp

# COMMAND ----------

connection_spec_id = "26f526f2-58f4-4712-961d-e41bf1ccc0e8"
source_res = flow_conn.createSourceConnection(
    {
        "name": "[CMLE][Week4] Data Landing Zone source connection for propensity scores",
        "data": {"format": "delimited"},
        "params": {
            "path": f"{dlz_container}/{import_path}",
            "type": "folder",
            "recursive": True,
        },
        "connectionSpec": {"id": connection_spec_id, "version": "1.0"},
    }
)

source_connection_id = source_res["id"]
source_connection_id

# COMMAND ----------

# MAGIC %md
# MAGIC The **target connection** is responsible for connecting to your Adobe Experience Platform dataset so that the resulting Data Flow will know where the data needs to be written. Because we already created our ingestion dataset in the previous section, we can simply tie it to that dataset ID and the corresponding schema.

# COMMAND ----------

target_res = flow_conn.createTargetConnectionDataLake(
    name="[CMLE][Week4] User Propensity Target Connection",
    datasetId=ingestion_dataset_id,
    schemaId=schema_id,
)

target_connection_id = target_res["id"]
target_connection_id

# COMMAND ----------

# MAGIC %md
# MAGIC We're still missing one step. If you look back to the previous cells, this is what we have as the schema of our scored dataframe:
# MAGIC - `userId`
# MAGIC - `prediction`
# MAGIC
# MAGIC And this is what we have as the schema of our ingestion dataset:
# MAGIC - `_$TENANTID.userid`
# MAGIC - `_$TENANTID.propensity`
# MAGIC
# MAGIC Although it may look obvious to us, we still need to let the platform know which fields maps to what. This can be achieved using the [Data Prep service](https://experienceleague.adobe.com/docs/experience-platform/data-prep/home.html) which allows you to specify a set of **transformations** to map one field to another. In our case the transformation is pretty simple, we just need to match the schemas without making any changes, but you can do a lot more extensive transformations using this service if needed.

# COMMAND ----------

from aepp import dataprep

dataprep_conn = dataprep.DataPrep()

# COMMAND ----------

mapping_res = dataprep_conn.createMappingSet(
    schemaId=schema_id,
    validate=True,
    mappingList=[
        {
            "sourceType": "ATTRIBUTE",
            "source": "prediction",
            "destination": f"_{tenant_id}.propensity",
        },
        {
            "sourceType": "ATTRIBUTE",
            "source": "userId",
            "destination": f"_{tenant_id}.userid",
        },
    ],
)

mapping_id = mapping_res["id"]
mapping_id

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we have everything we need to create a **Data Flow**. A data flow is the "recipe" that describes where the data comes from and where it should end up. We can also specify how often checks happen to find new data, but it cannot be lower than 15 minutes currently for platform stability reasons. A data flow is tied to a flow spec ID which contains the instructions for transfering data in an optimized way between a source and destination.
# MAGIC
# MAGIC For reference, here is a list of all the flow specs available for the most popular cloud storage accounts (these IDs are global across every single customer account and sandbox):
# MAGIC
# MAGIC | Cloud Storage Type    | Flow Spec ID                         | Flow Spec Name
# MAGIC |-----------------------|--------------------------------------|------------------
# MAGIC | Amazon S3             | 9753525b-82c7-4dce-8a9b-5ccfce2b9876 | CloudStorageToAEP
# MAGIC | Azure Blob Storage    | 14518937-270c-4525-bdec-c2ba7cce3860 | CRMToAEP
# MAGIC | Azure Data Lake       | 9753525b-82c7-4dce-8a9b-5ccfce2b9876 | CloudStorageToAEP
# MAGIC | Data Landing Zone     | 9753525b-82c7-4dce-8a9b-5ccfce2b9876 | CloudStorageToAEP
# MAGIC | Google Cloud Storage  | 9753525b-82c7-4dce-8a9b-5ccfce2b9876 | CloudStorageToAEP
# MAGIC | SFTP                  | 9753525b-82c7-4dce-8a9b-5ccfce2b9876 | CloudStorageToAEP

# COMMAND ----------

flow_spec = flow_conn.getFlowSpecs("name==CloudStorageToAEP")
flow_spec_id = flow_spec[0]["id"]
flow_spec_id

# COMMAND ----------

import time

# TODO: cleanup in aepp, first param should not be required
flow_res = flow_conn.createFlow(
    flow_spec_id,
    obj={
        "name": f"[CMLE][Week4] DLZ to AEP for user propensity (created by {username})",
        "flowSpec": {"id": flow_spec_id, "version": "1.0"},
        "sourceConnectionIds": [source_connection_id],
        "targetConnectionIds": [target_connection_id],
        "transformations": [
            {
                "name": "Mapping",
                "params": {"mappingId": mapping_id, "mappingVersion": 0},
            }
        ],
        "scheduleParams": {
            "startTime": str(int(time.time())),
            "frequency": "minute",
            "interval": "15",
        },
    },
)
dataflow_id = flow_res["id"]
dataflow_id

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the name of the transformation has to be set to `Mapping` or the job will fail.

# COMMAND ----------

# MAGIC %md
# MAGIC You should be able to see your Data Flow in the UI at the link below, and you may see some executions depending on when you check since it runs on a schedule and will still show the run even if there was no data to process, as shown in the screenshot below.

# COMMAND ----------

dataflow_link = get_ui_link(tenant_id, "source/dataflows", dataflow_id)
display_link(dataflow_link, f"Data Flow created as ID {dataflow_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ![Source Dataflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week4-Dataflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Note: If you would like to switch to a different cloud storage, you need to update the `flow_spec_id` variable above to the matching value in the table mentioned earlier in this section. You can refer to the name from the table above to find out the ID.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Ingest the scored users into the Unified Profile

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we have successfully setup a Data Flow that is listening on any files being written to the DLZ under our import path, and will automatically convert it to XDM and deliver it to our dataset where they will be picked up for ingestion into the Unified Profile. All that is left to do is to actually deliver the data. For that, we refer to our Spark dataframe computed earlier `df_to_ingest` and we need to write it to the DLZ under the afore-mentioned folder.
# MAGIC
# MAGIC Before we can do that we need to update the credentials, because we'll be writing to a different container in the DLZ (`dlz-user-container` instead of `dlz-destination`) so the credentials are different. This should not cause an issue with the lazy computation of our dataframe because we used `.cache()` to cache it so it should already be in memory right now.

# COMMAND ----------

spark.conf.set(f"fs.azure.account.auth.type.{dlz_storage_account}.dfs.core.windows.net", "SAS")
spark.conf.set(f"fs.azure.sas.token.provider.type.{dlz_storage_account}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set(f"fs.azure.sas.fixed.token.{dlz_storage_account}.dfs.core.windows.net", dlz_sas_token)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's determine the full path where we need to write. We use a similar convention as when the data as been egressed, which is `cmle/ingress/$DATASETID/exportTime=$EXPORTTIME`.

# COMMAND ----------

from datetime import datetime

protocol = "abfss"
scoring_export_time = datetime.utcnow().strftime('%Y%m%d%H%M%S')
output_path = f"{protocol}://{dlz_container}@{dlz_storage_account}.dfs.core.windows.net/{import_path}/{ingestion_dataset_id}/exportTime={scoring_export_time}/"
output_path

# COMMAND ----------

# MAGIC %md
# MAGIC Now we just need to write the datafrae to this output path. Note that because we chose `delimited` format in our Data Flow setup, we're just going to write the resulting files as CSV format, and include the header so the transformation knows which field is which column.

# COMMAND ----------

df_to_ingest \
  .write \
  .option("header", True) \
  .format("csv") \
  .save(output_path)

# COMMAND ----------

spark.read.format("csv").option("header", "true").load(output_path).count()

# COMMAND ----------

# MAGIC %md
# MAGIC Because the Data Flow is executed asynchronously every 15 minutes, it may take a few minutes before the data is ingested in the dataset. We can check the status of the runs below until we can see the run has successfully completed to check some summary statistics.

# COMMAND ----------

import time

# TODO: handle that more gracefully in aepp
finished = False
while not finished:
    try:
        runs = flow_conn.getRuns(prop=f"flowId=={dataflow_id}")
        for run in runs:
            run_id = run["id"]
            run_started_at = run["metrics"]["durationSummary"]["startedAtUTC"]
            run_ended_at = run["metrics"]["durationSummary"]["completedAtUTC"]
            run_duration_secs = (run_ended_at - run_started_at) / 1000
            run_size_mb = run["metrics"]["sizeSummary"]["outputBytes"] / 1024.0 / 1024.0
            run_num_rows = run["metrics"]["recordSummary"]["outputRecordCount"]
            run_num_files = run["metrics"]["fileSummary"]["outputFileCount"]
            print(f"Run ID {run_id} completed with: duration={run_duration_secs} secs; size={run_size_mb} MB; num_rows={run_num_rows}; num_files={run_num_files}")
        finished = True
    except Exception as e:
        print(f"No runs completed yet for flow {dataflow_id}")
        time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC Once this is done, you should be able to go back in your dataset at the same link as before and see a batch created successfully in it. You should also notice for that batch that the records ingested will also show up under **Existing Profile Fragments** which means they have been ingested in the Unified Profile successfully.

# COMMAND ----------

# MAGIC %md
# MAGIC ![Ingestion](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week4-Ingestion.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Storing the scoring dataset ID in the configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we got everything working, we just need to save the `ingestion_dataset_id` variable in the original configuration file, so we can refer to it in the following weekly assignment. To do that, execute the code below:

# COMMAND ----------

config.set("Platform", "scoring_dataset_id", ingestion_dataset_id)

with open(config_path, "w") as configfile:
    config.write(configfile)
