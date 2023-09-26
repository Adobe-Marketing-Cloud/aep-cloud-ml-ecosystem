# Databricks notebook source
# MAGIC %md
# MAGIC # Scope of Notebook
# MAGIC
# MAGIC The goal of this notebook is to showcase how you can generate a synthetic dataset through building a schema and fieldgroups and then use the Data Distiller to do exploratory data analysis.
# MAGIC
# MAGIC ![EndToEndDesign](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week1-Workflow.jpeg)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup
# MAGIC
# MAGIC There are a handful of libraries required to use this notebook and the ones that follow it. If you haven't already done so, please run the [RunMe Notebook]($./RunMe) the create a cluster that has the required libraries installed. They are all publicly available libraries and while the latest version should work fine, it is best practice to pin particular versions for your production workflows to ensure that upstream library releases don't causing breaking changes to your pipelines.
# MAGIC
# MAGIC This notebook also requires some configuration data to connect to your Adobe Experience Platform instance. You should be able to find all the values required above by following the Setup section of the **README**.
# MAGIC
# MAGIC The next cell will be looking for your configuration file under your **ADOBE_HOME** path to fetch the values used throughout this notebook. This environment variable is configured on your cluster to point to the DBFS FUSE mount point location of `/dbfs/home/{your_user_name}/.adobe`. See more details in the Setup section of the **README** to understand how to create your configuration file. Note that for deployment of production jobs, you'll want to take advantage [Databricks secret management](https://docs.databricks.com/en/security/secrets/index.html).
# MAGIC
# MAGIC Some imports and utility functions that will be used throughout this notebook are provided in the [Common Include]($./CommonInclude) notebook since they'll also be used in all the other notebooks.
# MAGIC
# MAGIC To ensure uniqueness of resources created as part of this notebook, we are using your local username to include in each of the resource titles to avoid conflicts.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %run ./CommonInclude

# COMMAND ----------

# MAGIC %md
# MAGIC In order to generate synthetic data we'll need to first create a schema and a dataset using the [aepp Python library](https://github.com/pitchmuc/aepp). This library is a REST API based on the workflow in the AEP UI.  Please see the following reference [guide](https://developer.adobe.com/experience-platform-apis/) for the underlying API's.  For the AEP UI workflow please click [here](https://experienceleague.adobe.com/docs/experience-platform/xdm/tutorials/create-schema-ui.html?lang=en).
# MAGIC
# MAGIC Your authentication credentials have already been loaded by the [Common Include]($./CommonInclude). Before continuing, please ensure you've configured the following pieces of information in the configuration file in your **ADOBE_HOME** directory, according to the `Setup` section of the **Readme**:
# MAGIC - Client ID
# MAGIC - Client secret
# MAGIC - Private key
# MAGIC - Technical account ID

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Prepare AEP Instance to Receive Data
# MAGIC
# MAGIC Before we can ingest the data into the platform, we need to set up the schemas and datasets for the experience events and profiles into which we'll be uploading the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Setting up schemas
# MAGIC
# MAGIC We will now create the schema to support our synthetic data. We need a few fields which will be included in the synthetic data:
# MAGIC
# MAGIC - Direct marketing information
# MAGIC - Web details
# MAGIC - Identity information
# MAGIC
# MAGIC These are already provided in your AEP instance as default field groups, so we'll be leveraging that for creation below.  The image below identifies the workflow in the AEP UI to create the schema. The above fields are already provided in the AEP instance as default field groups, we'll be using that information to create the schema details below.
# MAGIC
# MAGIC We first print out the tenantId, as this represents our ims org name.

# COMMAND ----------

from aepp import schema

schema_conn = schema.Schema()
tenant_id = schema_conn.getTenantId()

print(f"sandbox: {schema_conn.sandbox}")
print(f"tenant_id: {tenant_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.1 Creating the Experience Event field group
# MAGIC
# MAGIC Our goal is to create schemas for the Profile and Experience Events and fieldgroups in these schemas. A fieldgroup allows us to define and query segments around the profile and experience events. Conceptually a fieldgroup allows us to gather together a set of fields to represent data in our segments.

# COMMAND ----------

event_fieldgroup_spec = {
    "type": "object",
    "title": f"[CMLE] [Week1] Exp Event related to user propensity subscription (created by {username})",
    "description": "This mixin is used to define a propensity score that can be assigned to a given profile and associated experience events.",
    "allOf": [{"$ref": "#/definitions/customFields"}],
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
                        "userid": {
                            "title": "User ID",
                            "description": "This refers to the user having a propensity towards an outcome.",
                            "type": "string",
                        }
                    },
                }
            },
        }
    },
    "meta:intendedToExtend": [
        "https://ns.adobe.com/xdm/context/experienceevent"
    ],
}

event_fieldgroup_res = get_or_create_fieldgroup(schema_conn, event_fieldgroup_spec)
event_fieldgroup_id = event_fieldgroup_res["$id"]
event_fieldgroup_link = get_ui_link(tenant_id, "schema/mixin/browse", urllib.parse.quote(event_fieldgroup_id, safe="a"))
display_link(event_fieldgroup_link, event_fieldgroup_res['title'])

# COMMAND ----------

# MAGIC %md
# MAGIC ![AEP-FieldGroup](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week1-FieldGroup.jpeg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.2 Creating the Profile field group

# COMMAND ----------

profile_fieldgroup_spec = {
  	"type": "object",
	"title": f"[CMLE] [Week1] Profile Fieldgroup associated with user propensity subscription (created by {username})",
	"description": "This mixin is used to define a propensity score that can be assigned to a given profile and associated experience events.",
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

profile_fieldgroup_res = get_or_create_fieldgroup(schema_conn, profile_fieldgroup_spec)
profile_fieldgroup_id = profile_fieldgroup_res["$id"]
profile_fieldgroup_link = get_ui_link(tenant_id, "schema/mixin/browse", urllib.parse.quote(profile_fieldgroup_id, safe="a"))
display_link(profile_fieldgroup_link, profile_fieldgroup_res['title'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.3 Creating the Experience Event schema
# MAGIC
# MAGIC Now create the experience event schema and the descriptor. Schema descriptors are tenant-level metadata, unique to your IMS Organization and all descriptor operations take place in the tenant container.

# COMMAND ----------

event_schema_title = f"[CMLE] [Week1] Notebook with ecid added and Synthetic Event Schema (created by {username})"

event_schema_res = get_or_create_experience_event_schema(
    schema_conn,
    title=event_schema_title,
    mixinIds=[
        event_fieldgroup_id,
        "https://ns.adobe.com/xdm/context/experienceevent-directmarketing",
        "https://ns.adobe.com/xdm/context/experienceevent-web",
    ],
    description="Profile Schema generated by CMLE for synthetic events",
)

event_schema_id = event_schema_res["$id"]
event_schema_alt_id = event_schema_res["meta:altId"]
event_schema_link = get_ui_link(tenant_id, "schema/mixin/browse", urllib.parse.quote(event_schema_id, safe="a"))
display_link(event_schema_link, event_schema_title)

# COMMAND ----------

# MAGIC %md
# MAGIC ![AEP-Schema](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week1-Schema.jpeg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.4 Creating the Experience Event identity descriptor

# COMMAND ----------

identity_type = "ECID"
event_descriptor_obj = {
    "@type": "xdm:descriptorIdentity",
    "xdm:sourceSchema": event_schema_id,
    "xdm:sourceVersion": 1,
    "xdm:sourceProperty": f"/_{tenant_id}/userid",
    "xdm:namespace": identity_type,
    "xdm:property": "xdm:id",
    "xdm:isPrimary": True,
}
event_descriptor_res = get_or_create_descriptor(schema_conn, event_descriptor_obj)
event_descriptor_res

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.5 Creating the Profile Schema

# COMMAND ----------

profile_schema_title = f"[CMLE] [Week1] Notebook with ecid added and Profile Schema (created by {username})"

profile_schema_res = get_or_create_profile_schema(
    schema_conn,
    title=profile_schema_title,
    mixinIds=[profile_fieldgroup_id],
    description="Profile Schema generated by CMLEs",
)

profile_schema_id = profile_schema_res["$id"]
profile_schema_alt_id = profile_schema_res["meta:altId"]
profile_schema_link = get_ui_link(tenant_id, "schema/mixin/browse", urllib.parse.quote(profile_schema_id, safe="a"))
display_link(profile_schema_link, profile_schema_title)

# COMMAND ----------

identity_type = "ECID"

profile_descriptor_obj = {
    "@type": "xdm:descriptorIdentity",
    "xdm:sourceSchema": profile_schema_id,
    "xdm:sourceVersion": 1,
    "xdm:sourceProperty": f"/_{tenant_id}/userid",
    "xdm:namespace": identity_type,
    "xdm:property": "xdm:id",
    "xdm:isPrimary": True,
}

profile_descriptor_res = get_or_create_descriptor(schema_conn, profile_descriptor_obj)
profile_descriptor_res

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.6 Enabling Schemas for Unified Profile

# COMMAND ----------

event_enable_res = schema_conn.enableSchemaForRealTime(event_schema_alt_id)
event_enable_res

# COMMAND ----------

profile_enable_res = schema_conn.enableSchemaForRealTime(profile_schema_alt_id)
profile_enable_res

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Setting up datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.1 Creating the Experience Event dataset

# COMMAND ----------

from aepp import catalog
cat_conn = catalog.Catalog()

# COMMAND ----------

dataset_name = f"[CMLE] [Week1] Notebook with ecid used Dataset (created by {username})"
existing_dataset_ids = get_dataset_ids_by_name(cat_conn, dataset_name)
if len(existing_dataset_ids) == 0:
    dataset_res = cat_conn.createDataSets(
        name=dataset_name,
        schemaId=event_schema_id,
    )
    dataset_id = dataset_res[0].split("/")[-1]
    dataset_new_or_existing = "created new"
else:
    dataset_id = existing_dataset_ids[0]
    dataset_new_or_existing = f"reused first of {len(existing_dataset_ids)}"

dataset_link = get_ui_link(tenant_id, "dataset/browse", dataset_id)
display_link(dataset_link, f"{dataset_name} ({dataset_new_or_existing})")

# COMMAND ----------

# MAGIC %md
# MAGIC ![AEPDataset](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week1-DataSet.jpeg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.2 Enabling the dataset for Unified Profile
# MAGIC

# COMMAND ----------

cat_conn.enableDatasetProfile(dataset_id)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <!--<img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note" />-->
# MAGIC <b>Note:</b> After you do this step please go in the UI and click on the link above, if the profile toggle is not enabled please manually toggle the profile on.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset ID printed above is where we will be uploading all the synthetic data, and what we will use as the basis for querying the data and building our ML model down the line.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Statistical Simulation
# MAGIC
# MAGIC The first step is to create a detailed simulation that will allow a reasonable propensity model to be built. 
# MAGIC
# MAGIC Our goal in this task will be to create a propensity model for "subscription" events.
# MAGIC
# MAGIC A subscription event will be defined as an event where a `web.formFilledOut` event is recorded. 
# MAGIC These will be events where a customer subscribes to the desired plan.
# MAGIC
# MAGIC In order to incorporate our custom experience events we replace the identityMap with the tenantId

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 EventTypes and their contribution to propensity
# MAGIC
# MAGIC We will allow for several types of experience events to be received for each user.
# MAGIC We will create a "generative" model of subscriptions as follows:
# MAGIC
# MAGIC 1. We sample randomly from a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) for the number of advertising impressions, webPageViews, and emailsSent. **These events can happen at random times over a 10 week interval.**
# MAGIC 2. For each of these "base" exposure events, we then have a corresponding conversion:
# MAGIC     - If an advertising impression occurs, we then allow for an advertising click to happen, with a certain probability.  
# MAGIC     - If a web page view occurs, then linkClicks, productViews, purchases, propositionDisplays, Interacts and Dismisses can all occur.
# MAGIC     - For an Email Sent, opens and clicks can then also occur. 
# MAGIC     
# MAGIC 3. After all these base events have been generated, we then have a timeseries of events for each user. Each of the timeseries events affects the user's propensity to subscribe. After each event the user then has a certain probability of subscribing. The subscription is then evaluated with a [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) draw - if the user subscribes, no further subscription evaluations are made. If the subscription does not happen, the subscription possibility will continue to be evaluated. 
# MAGIC
# MAGIC 4. Extra - if more than 10 advertising impressions, or 5 emails are sent, the user churns, and no more events for that user are generated.
# MAGIC
# MAGIC You can find the definitions of the logic for the simulations in the [Simulation Helpers]($./SimulationHelpers) notebook. [Running the notebook](https://docs.databricks.com/en/notebooks/notebook-workflows.html) with the `%run` magic command runs it as if it were part of this notebook.

# COMMAND ----------

# MAGIC %run ./SimulationHelpers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Create Simulation Delta Table
# MAGIC
# MAGIC With the simulation logic defined, we can now use it to create our synthetic events dataset. We'll create the events in a [Delta table](https://docs.databricks.com/en/delta/index.html) first so we'll have a local copy to work from for the upload. 
# MAGIC
# MAGIC Also, since the simulation logic doesn't have any dependencies across users, we can treat it as an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem and create them in a distributed fashion across all of the worker nodes in our Databricks cluster rather than in a single thread just on the driver. This allows you to scale up to larger simulations, as will as showcasing techniques you can apply to other data sources you may want to ingest into the platform.
# MAGIC
# MAGIC For more information on mapInPandas and other vectorized Python functions available in Databricks, please refer to [the documentation](https://docs.databricks.com/en/pandas/pandas-function-apis.html#map).

# COMMAND ----------

from aepp import ingestion

ingest_conn = ingestion.DataIngestion()

# Wrap the main simulation function in another function we'll pass to mapInPandas.
# This function will receive batches of mock id's we'll use to drive the simulation
# process. 
# 
# Technical detail (feel free to skip):
#   An alternative would have been to pass in specifications to call for batches
#   from individual rows, but that can run into challenges with AQE automatically 
#   coalescing partitions on us. So this approach, though still a bit technical, can
#   end up being a little easier to understand since we avoid that.
def create_data_for_n_users_dataframe(dfs):
    # mapInPandas takes in batches of records as Pandas dataframes
    for df in dfs:
        # determine the size and start id for this batch
        batch_size = len(df)
        first_user_id_for_batch = df.id.min()

        # use those parameters for that batch to call our core simulation logic
        events = create_data_for_n_users(batch_size, first_user_id_for_batch)

        # and then yield the batch resulting from that call
        yield pd.DataFrame(events)


num_batches = 10
batch_size = 10000

batch_schema = T.StructType([
    T.StructField("userId", T.LongType()),
    T.StructField("eventType", T.StringType()),
    T.StructField("timestamp", T.StringType()),
    T.StructField("subscriptionPropensity", T.DoubleType()),
    T.StructField("subscribed", T.BooleanType())])

events_table_name = "synthetic_events"
target_event_count = num_batches * batch_size

# Run the simulation if the table isn't there and hasn't already been populated
if spark.catalog.tableExists(events_table_name) and not spark.table(events_table_name).isEmpty():
    print("table already exists and has data")
else:
    print("table doesn't already exist, so creating it")
    all_events = (
        spark.range(target_event_count)
        .mapInPandas(create_data_for_n_users_dataframe, batch_schema))
    all_events.write.mode("overwrite").saveAsTable("synthetic_events")

# build and show a link to the resulting Delta table
all_events = spark.table("synthetic_events")
synthetic_table_link = f"/explore/data/{spark.catalog.currentCatalog()}/{spark.catalog.currentDatabase()}/{events_table_name}"
display_link(synthetic_table_link, "Synthetic Delta Table in Unity Catalog")
display(all_events)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Batch ingestion 

# COMMAND ----------

# MAGIC %md
# MAGIC Now we're going to use the ingestion APIs using the same `aepp` library. Ingesting a batch of data is a multi-step process:
# MAGIC 1. Create an empty batch to initialize the connection
# MAGIC 2. Upload some data using the same batch ID we just created
# MAGIC 3. Mark the batch as completed successfully
# MAGIC
# MAGIC For more details about batch ingestion you can find API details [here](https://experienceleague.adobe.com/docs/experience-platform/ingestion/batch/overview.html?lang=en).
# MAGIC
# MAGIC Here again, we can take advantage of parallelization and distributed compute to upload multiple batches simultaneously across all the worker nodes in our cluster, using a similar technique as we used above.

# COMMAND ----------

event_ingestion_schema = T.StructType([
    T.StructField("batch_id", T.StringType()),
    T.StructField("batch_size", T.LongType()),
    T.StructField("start_time", T.TimestampType()),
    T.StructField("end_time", T.TimestampType()),
    T.StructField("duration", T.DoubleType()),
    T.StructField("message", T.StringType()),
    T.StructField("success", T.BooleanType())
])

def ingest_events(dfs: List[pd.DataFrame]):
    # ingest events runs on each worker, so we need to make sure its configured there
    # as well. the values defined above should be marshalled across, though we could
    # have re-read them from the config file if necessary
    aepp.configure(
        org_id=ims_org_id,
        tech_id=tech_account_id,
        secret=client_secret,
        path_to_key=private_key_path,
        client_id=client_id,
        environment=environment,
        sandbox=sandbox_name,
    )
    ingest_conn = ingestion.DataIngestion()
    for df in dfs:
        start_time = time.time()

        batch_id = None
        batch_data = []

        for i, x in df.iterrows():
            event = create_xdm_event(
                f"synthetic-user-{x['userId']}@adobe.com", 
                x["eventType"], x["timestamp"])
            batch_data.append(event)
        
        try:
            batch_res = ingest_conn.createBatch(datasetId=dataset_id)
            batch_id = batch_res["id"]

            file_path = f"batch-synthetic-{batch_id}"
            ingest_conn.uploadSmallFile(
                batchId=batch_id, datasetId=dataset_id, filePath=batch_id, data=batch_data)

            # Complete the batch
            ingest_conn.uploadSmallFileFinish(batchId=batch_id)

            success = True
            message = "OK"
        except Exception as e:
            message = str(e)
            success = False

        end_time = time.time()
        duration = end_time - start_time

        # Return a row with summary statistics for each batch uploaded
        yield pd.DataFrame([{
            "batch_id": batch_id,
            "batch_size": len(batch_data),
            "start_time": pd.to_datetime(start_time, unit="s"),
            "end_time": pd.to_datetime(end_time, unit="s"),
            "duration": end_time - start_time,
            "message": message,
            "success": success
        }])


ingestion_results_table_name = "batch_ingestion_results"
num_ingestion_partitions = 128

# Kick off the upload if it hasn't already been processed
# .. i.e., if the ingestion results table isn't already there
if not spark.catalog.tableExists(ingestion_results_table_name):
    batch_ingestion_results = (
        all_events
        .repartition(num_ingestion_partitions)
        .mapInPandas(ingest_events, event_ingestion_schema))

    batch_ingestion_results.write.mode("overwrite").saveAsTable(ingestion_results_table_name)

batch_ingestion_results = spark.table(ingestion_results_table_name)
display(batch_ingestion_results)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: Batches are ingested asynchronously in AEP. It may take some time for all the data generated here to be available in your dataset depending on how your AEP organization has been provisioned. You can check ingestion status for all your batches in [the dataset page of your AEP UI](https://experience.adobe.com/#/@TENANT/sname:SANDBOX/platform/dataset/browse/DATASETID)

# COMMAND ----------

from aepp import catalog

import time

cat_conn = catalog.Catalog()

num_batches = num_ingestion_partitions
all_ingested = False
while not all_ingested:
    incomplete_batches = cat_conn.getBatches(
        limit=min(100, num_batches),
        n_results=num_batches,
        output="dataframe",
        dataSet=dataset_id,
        status="staging",
    )

    num_incomplete_batches = len(incomplete_batches)
    if num_incomplete_batches == 0:
        print("All batches have been ingested")
        all_ingested = True
    else:
        print(f"Remaining batches being ingested: {num_incomplete_batches}")
        time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Data Exploration for Propensity Models
# MAGIC
# MAGIC In this notebook, we connect to the query service using the aepp library, and examine the data that we have uploaded. 
# MAGIC
# MAGIC We do the following steps:
# MAGIC
# MAGIC - Connect to query service using the aepp configuration parameters
# MAGIC - Discover the schema of the data, and explore a few rows
# MAGIC - Compute basic statistics
# MAGIC - Examine correlations among features, to inform feature creation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Interactive Session with Data Distiller

# COMMAND ----------

# MAGIC %md
# MAGIC Every dataset ID in the Adobe Experience Platform is tied to a table name in the Query Service world. We can easily get the table name by doing a lookup on the dataset ID and extracting the table name from the tags:

# COMMAND ----------

from aepp import catalog

cat_conn = catalog.Catalog()

dataset_info = cat_conn.getDataSet(dataset_id)
table_name = dataset_info[dataset_id]["tags"]["adobe/pqs/table"][0]
print(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC When you set the connection to the query service object, you'll setup a connection to the actual table you need to connect to. This will be faster and less resource-intensive for the query service API.

# COMMAND ----------

from aepp import queryservice

qs_conn = queryservice.QueryService().connection()
qs_conn["dbname"] = f"{sandbox_name}:{table_name}"
qs_cursor = queryservice.InteractiveQuery(qs_conn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Querying the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC We can use the interactive session created just before to issue any kinds of queries to the Query Service. As an example, here we simply select all the fields in our synthetic data table.

# COMMAND ----------

sample_experience_event_query = f'''SELECT * FROM {table_name} LIMIT 50'''
display(qs_cursor.query(sample_experience_event_query))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Querying Complex Fields
# MAGIC
# MAGIC Let's sample some of the fields in our dataset. We have different types. While some are timestamps and others are just primitives like strings, some are complex nested XDM structures. Let's see what we get when we query these as-is: 

# COMMAND ----------

schema_query= f'''
SELECT directMarketing, _id, eventType, timestamp 
FROM {table_name} 
WHERE directMarketing IS NOT NULL 
LIMIT 5'''
df = qs_cursor.query(schema_query, output="dataframe")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see when looking at the complex nested field, it's pretty hard to make sense of what this data is and the underlying structure:

# COMMAND ----------

df["directMarketing"].iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's run the same query again but with a twist: we can set the `auto_to_json` flag to be true. This configuration ensures that complex structures are automatically converted into a JSON object so that the field names can be easily queried.

# COMMAND ----------

schema_query= f"""
SET auto_to_json=true; 

SELECT directMarketing, _id, eventType, timestamp 
FROM {table_name} 
WHERE directMarketing IS NOT NULL 
LIMIT 5"""
df = qs_cursor.query(schema_query, output="dataframe")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, the data is now much easier to digest, and we can see clearly the underlying structure along with the different field names.

# COMMAND ----------

df["directMarketing"].iloc[0]

# COMMAND ----------

import json
import pprint as pp

pp.pprint(json.loads(df["directMarketing"].iloc[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Manually get some basic statistics
# MAGIC
# MAGIC Let's look at the number of rows and number of profiles in our synthetic dataset as an example of basic computations that can be done with Query Service:

# COMMAND ----------

basic_statistics_query = f"""
SELECT
    COUNT(_id) as "totalRows",  
    COUNT(DISTINCT _id) as "distinctUsers" 
FROM {table_name}"""
df = qs_cursor.query(basic_statistics_query, output="dataframe")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Create a sampled version of the table
# MAGIC
# MAGIC If our dataset is too big, or we simply don't need to get exact numbers for our queries, we can use the [sampling functionality](https://experienceleague.adobe.com/docs/experience-platform/query/sql/dataset-samples.html?lang=en) available in Query Service. This happens in multiple steps:
# MAGIC - First we have to **analyze** the table to create an actual sample with a specific sampling ratio.
# MAGIC - Then we can query the actual sample created which will automatically extrapolate the numbers to the full dataset.
# MAGIC
# MAGIC As an example below, we start by analyzing the table and creating a 5% sample:

# COMMAND ----------

def get_sample_table_name(table_name, sampling_rate):
    try:
        sql_text = f"""SELECT sample_meta('{table_name}')"""
        df = qs_cursor.query(sql_text, output="dataframe")
        df_at_rate = df[df["sampling_rate"] == sampling_rate]
        if len(df_at_rate) > 0:
            return df["sample_table_name"].iloc[0]
        else:
            return None
    except Exception as e:
        return None
    

# A sampling rate of 10 is 100% in Query Service, so for 5% we have to use 0.5
sampling_rate = 0.5

analyze_table_query=f"""
SET aqp=true; 

ANALYZE TABLE {table_name} TABLESAMPLE SAMPLERATE {sampling_rate}"""

sample_table_name = get_sample_table_name(table_name, sampling_rate)
if sample_table_name is None:
    print(f"sample table for sampling rate {sampling_rate} of table {table_name} doesn't exist - analyzing now")
    qs_cursor.query(analyze_table_query, output="raw")
    sample_table_name = get_sample_table_name(table_name, sampling_rate)
else:
    print(f"sample table already exists")

print(sample_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC If we want to see all the different samples available for our table, we can use the `sample_meta` function which will have an entry for each sample pointing to that sample's dataset ID and table name as well as the recorded sampling ratio.

# COMMAND ----------

sampled_version_of_table_query = f'''SELECT sample_meta('{table_name}')'''

df_samples = qs_cursor.query(sampled_version_of_table_query, output="dataframe")
display(df_samples)

# COMMAND ----------

# MAGIC %md
# MAGIC So now let's compare what happens when we run the same query on both the original table, and our 5% sample. We're using a very simple query to just do a `COUNT` to compare both the accuracy of the result, and also how much faster it is.

# COMMAND ----------

# MAGIC %%time
# MAGIC count_query=f'''SELECT count(*) AS count FROM {table_name}'''
# MAGIC df = qs_cursor.query(count_query, output="dataframe")
# MAGIC display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC To query the sample we get the sampling table name from the metadata above, and then query it directly and multiply the results by the sampling ratio to get an estimate.

# COMMAND ----------

# MAGIC %%time
# MAGIC count_query=f'''SELECT count(*) as cnt from {sample_table_name}'''
# MAGIC df = qs_cursor.query(count_query, output="dataframe")
# MAGIC approx_count = df["cnt"].iloc[0] / (sampling_rate / 100)
# MAGIC print(f"Approximate count: {approx_count} using {sampling_rate *10}% sample")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b>
# MAGIC     
# MAGIC You can also query the latest sample from that dataset by using `SELECT * from {table_name} WITHAPPROXIMATE`. However, it is not advised to do aggregation queries or joins with that since this is only a uniform random sample.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the results are pretty accurate &lt;1% error, and also the runtime is reduced by at least 20%, so using samples are a good choice for featurization data if we have a ML model that is not necessarily data-hungry.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Analyzing the data
# MAGIC
# MAGIC Let's now analyze the data by creating a few visualizations using some commonly requested questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Email Funnel Analysis
# MAGIC
# MAGIC Let's look at the funnel of how many users actually fill out the webForm. A funnel analysis is a method of understanding the steps required to reach an outcome on a website and how many users get through each of those steps.

# COMMAND ----------

simple_funnel_analysis_query = f'''
SELECT eventType, COUNT(DISTINCT _id) as "distinctUsers", COUNT(_id) as "distinctEvents" 
FROM {table_name} 
GROUP BY eventType 
ORDER BY distinctUsers DESC'''
funnel_df = qs_cursor.query(simple_funnel_analysis_query, output="dataframe")
display(funnel_df)

# COMMAND ----------

from plotly import graph_objects as go

email_funnel_events = [
    "directMarketing.emailSent", 
    "directMarketing.emailOpened", 
    "directMarketing.emailClicked", 
    "web.formFilledOut"
]

email_funnel_df = funnel_df[funnel_df["eventType"].isin(email_funnel_events)]

fig = go.Figure(
    go.Funnel(
        y=email_funnel_df["eventType"], 
        x=email_funnel_df["distinctUsers"],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.65, 
        marker={
            "color": ["deepskyblue", "lightsalmon", "tan", "teal"],
            "line": {"width": [4, 2, 2, 3, 1, 1], "color": ["wheat", "wheat", "blue", "wheat"]},
        },
        connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
    )
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Event correlation
# MAGIC Now, we analyze the correlation between various types of events. 
# MAGIC
# MAGIC We'll look for which events predict the `web.formFilledOut` outcome. 
# MAGIC
# MAGIC To do this, we must execute a more complex join query.

# COMMAND ----------

event_correlation_query=f'''
SELECT  eventType_First, eventType_Later, COUNT(DISTINCT userId) as "distinctUsers"
FROM 
    (
        SELECT a.eventType as eventType_First, 
                b.eventType as eventType_Later, 
                a._{tenant_id}.userid as userID 
        FROM {table_name} a
        JOIN {table_name} b
        ON a._{tenant_id}.userid = b._{tenant_id}.userid
        WHERE a.timestamp <= b.timestamp
    )
GROUP BY eventType_First, eventType_Later
ORDER BY distinctUsers DESC'''
event_correlation_df = qs_cursor.query(event_correlation_query, output="dataframe")
display(event_correlation_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we join the results of this correlation to obtain a cooccurrence matrix which we can then display to get a visual feel of which events are likely to be occurring together.

# COMMAND ----------

coocc_matrix = event_correlation_df
individual_counts = funnel_df
cocc_with_individual = coocc_matrix.merge(individual_counts, left_on="eventType_First", right_on="eventType")
cocc_with_individual["probability"] = cocc_with_individual["distinctUsers_x"]/ cocc_with_individual["distinctUsers_y"]

# COMMAND ----------

import seaborn as sns

pivoted = cocc_with_individual.pivot("eventType_First", "eventType_Later", "probability")
sns.heatmap(pivoted);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 A more robust correlation calculation
# MAGIC
# MAGIC Let's use an built-in feature of the query service (Spark functions) to get a better handle of correlations between various eventTypes. We'll use the [`corr` function](https://spark.apache.org/docs/3.5.0/api/sql/#corr) which computes [Pearson correlation coefficients](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between a given eventType and the target eventType.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run on the full dataset

# COMMAND ----------

large_correlation_query=f'''
SELECT SUM(webFormsFilled) as webFormsFilled_totalUsers,
       SUM(advertisingClicks) as advertisingClicks_totalUsers,
       SUM(productViews) as productViews_totalUsers,
       SUM(productPurchases) as productPurchases_totalUsers,
       SUM(propositionDismisses) as propositionDismisses_totaUsers,
       SUM(propositionDisplays) as propositionDisplays_totaUsers,
       SUM(propositionInteracts) as propositionInteracts_totalUsers,
       SUM(emailClicks) as emailClicks_totalUsers,
       SUM(emailOpens) as emailOpens_totalUsers,
       SUM(webLinkClicks) as webLinksClicks_totalUsers,
       SUM(webPageViews) as webPageViews_totalusers,
       corr(webFormsFilled, emailOpens) as webForms_EmalOpens,
       corr(webFormsFilled, advertisingClicks) as webForms_advertisingClicks,
       corr(webFormsFilled, productViews) as webForms_productViews,
       corr(webFormsFilled, productPurchases) as webForms_productPurchases,
       corr(webFormsFilled, propositionDismisses) as webForms_propositionDismisses,
       corr(webFormsFilled, propositionInteracts) as webForms_propositionInteracts,
       corr(webFormsFilled, emailClicks) as webForms_emailClicks,
       corr(webFormsFilled, emailOpens) as webForms_emailOpens,
       corr(webFormsFilled, emailSends) as webForms_emailSends,
       corr(webFormsFilled, webLinkClicks) as webForms_webLinkClicks,
       corr(webFormsFilled, webPageViews) as webForms_webPageViews
FROM(
    SELECT _{tenant_id}.userid as userID,
            SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) as webFormsFilled,
            SUM(CASE WHEN eventType='advertising.clicks' THEN 1 ELSE 0 END) as advertisingClicks,
            SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) as productViews,
            SUM(CASE WHEN eventType='commerce.productPurchases' THEN 1 ELSE 0 END) as productPurchases,
            SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) as propositionDismisses,
            SUM(CASE WHEN eventType='decisioning.propositionDisplay' THEN 1 ELSE 0 END) as propositionDisplays,
            SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) as propositionInteracts,
            SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) as emailClicks,
            SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) as emailOpens,
            SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) as emailSends,
            SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) as webLinkClicks,
            SUM(CASE WHEN eventType='web.webinteraction.pageViews' THEN 1 ELSE 0 END) as webPageViews
    FROM {table_name}
    GROUP BY userId
)
'''
large_correlation_df = qs_cursor.query(large_correlation_query, output="dataframe")
display(large_correlation_df)

# COMMAND ----------

cols = large_correlation_df.columns
corrdf = large_correlation_df[[col for col in cols if ("webForms_"  in col)]].melt()
corrdf["feature"] = corrdf["variable"].apply(lambda x: x.replace("webForms_", ""))
corrdf["pearsonCorrelation"] = corrdf["value"]

display(corrdf.fillna(0))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visualize the results:

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(5,10))
sns.barplot(data=corrdf.fillna(0), y="feature", x="pearsonCorrelation")
ax.set_title("Pearson Correlation of Events with the outcome event");

# COMMAND ----------

config.set("Platform", "dataset_id", dataset_id)

with open(config_path, "w") as configfile:
    config.write(configfile)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-success">
# MAGIC     <b>Conclusion</b>
# MAGIC With this information, we now have a hypothesis on necessary features to use in our model.
# MAGIC We will use the number of these various event types, as well as the recency of each event type as features 
# MAGIC for the model. The next step is to create these "featurized" datasets
# MAGIC </div>
