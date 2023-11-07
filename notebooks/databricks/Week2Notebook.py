# Databricks notebook source
# MAGIC %md
# MAGIC # Scope of Notebook
# MAGIC
# MAGIC The goal of this notebook is to showcase how you can prepare data for the future goal of consumption by an ML model, and leveraging functionality in the Adobe Experience Platform to generate features at scale and make it available in your choice of cloud storage.
# MAGIC
# MAGIC ![Workflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week2-Workflow.png)
# MAGIC
# MAGIC We'll go through several steps:
# MAGIC - **Creating a query** to encapsulate what a good featurized dataset will be.
# MAGIC - **Executing that query** and storing the results.
# MAGIC - **Setting up a flow** to export the results into cloud storage.
# MAGIC - **Executing that flow** to deliver the results.

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
# MAGIC We'll be using the [aepp Python library](https://github.com/pitchmuc/aepp) here to interact with AEP APIs and create a schema and dataset suitable for adding our synthetic data further down the line. This library simply provides a programmatic interface around the REST APIs, but all these steps could be completed similarly using the raw APIs directly or even in the UI. For more information on the underlying APIs please see [the API reference guide](https://developer.adobe.com/experience-platform-apis/).
# MAGIC
# MAGIC Before any calls can take place, we need to configure the library and setup authentication credentials. For this you'll need the following piece of information. For information about how you can get these, please refer to the `Setup` section of the **README**:
# MAGIC - Client ID
# MAGIC - Client secret
# MAGIC - Private key
# MAGIC - Technical account ID

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Creating Featurization Query

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous week we created some synthetic data under a dataset in your Adobe Experience Platform instance, and now we're ready to use it to generate features that can then be fed to our ML model. For this purpose we'll be using the [Query Service](https://experienceleague.adobe.com/docs/experience-platform/query/home.html?lang=en#) which lets us access data from any dataset and run queries at scale. The end goal here is to compress this dataset into a small subset of meaningful features that will be relevant to our model.

# COMMAND ----------

# MAGIC %md
# MAGIC Before we can issue queries, we need to find the table name corresponding to our dataset. Please make sure your dataset ID was entered in your configuration as part of the setup, it should be available under the `dataset_id` variable in this notebook.
# MAGIC
# MAGIC Every dataset in AEP should have a corresponding table in PQS based on its name. We can use AEP APIs to get that information with the code below:

# COMMAND ----------

from aepp import catalog

cat_conn = catalog.Catalog()

dataset_info = cat_conn.getDataSet(dataset_id)
table_name = dataset_info[dataset_id]["tags"]["adobe/pqs/table"][0]
table_name

# COMMAND ----------

# MAGIC %md
# MAGIC And because some of the data we created in the previous week is under a custom field group, we need to fetch your tenant ID since the data will be nested under it. This can be accomplished simply with the code below

# COMMAND ----------

from aepp import schema

schema_conn = schema.Schema()
tenant_id = schema_conn.getTenantId()

print(f"sandbox: {schema_conn.sandbox}")
print(f"tenant_id: {tenant_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can use that table to query it via Query Service. Every query is able to run at scale leveraging Spark-based distributed computing power in the backend, so the goal is to take this large dataset, extract meaningful features and only keep a smaller subset to feed into a ML model.
# MAGIC
# MAGIC We'll be leveraging `aepp` again to interact with Query Service.

# COMMAND ----------

from aepp import queryservice

qs = queryservice.QueryService()
qs_conn = qs.connection()
qs_conn['dbname']=f'{sandbox_name}:{table_name}'
qs_cursor = queryservice.InteractiveQuery(qs_conn)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> If at any point in this notebook the connection to Query Service is closed, you can refresh it by re-enabling that cell.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC Let's first define our ML problem scientifically:
# MAGIC - **What kind of problem** are we solving? We'd like to predict whether someone is likely to subscribe or not. We treat that as a binary classification problem, you are either subscribed or you are not.
# MAGIC - What is our **target variable**? Whether a subscription occurred or not for a given user.
# MAGIC - What are our **positive labels**? People who have at least 1 event corresponding to a subscription (marked as `web.formFilledOut`). We will keep a single feature row from the event where they actually subscribed.
# MAGIC - What are our **negative labels**? People who don't have a single event corresponding to a subscription. We will keep a random row to avoid having bias in the data.
# MAGIC
# MAGIC Let's put that into practice and start looking at our positive labels:

# COMMAND ----------

query_positive_labels = f"""
SELECT *
FROM (
    SELECT
        eventType,
        _{tenant_id}.userid as userId,
        SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY _{tenant_id}.userid) 
            AS "subscriptionOccurred"
    FROM {table_name}
)
WHERE subscriptionOccurred = 1 AND eventType = 'web.formFilledOut'
"""

df_positive_labels = qs_cursor.query(query_positive_labels, output="dataframe")
print(f"Number of positive classes: {len(df_positive_labels)}")
display(df_positive_labels.head())

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> 
# MAGIC     
# MAGIC We are using a sub-query because we want to filter on the `subscriptionOccurred` which is defined as part of the query, so it can't be used in a filter condition unless it is in a sub-query.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's look at our negative labels. Because we just want to retain a random row to avoid bias, we need to introduce randomness into our query:

# COMMAND ----------

query_negative_labels = f"""
SELECT *
FROM (
    SELECT
        _{tenant_id}.userid as userId,
        SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY _{tenant_id}.userid) 
            AS "subscriptionOccurred",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
)
WHERE subscriptionOccurred = 0 AND random_row_number_for_user = 1
"""

df_negative_labels = qs_cursor.query(query_negative_labels, output="dataframe")
print(f"Number of negative classes: {len(df_negative_labels)}")
display(df_negative_labels.head())

# COMMAND ----------

# MAGIC %md
# MAGIC Putting it all together, we can query both our positive and negative classes with the following query:

# COMMAND ----------

query_labels = f"""
SELECT *
FROM (
    SELECT
        eventType,
        _{tenant_id}.userid as userId,
        SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY _{tenant_id}.userid) 
            AS "subscriptionOccurred",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
)
WHERE (subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1)
"""

df_labels = qs_cursor.query(query_labels, output="dataframe")
print(f"Number of classes: {len(df_labels)}")
display(df_labels.head())

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's think what kind of features make sense for this kind of problem that we would like to eventually feed to an ML model. There's 2 main kinds of features we're interested in:
# MAGIC - **Actionable features**: different actions the user actually took in response to a marketing event.
# MAGIC - **Temporal features**: distribution over time of the actions the user took. This is useful for modeling to capture behavior over time and not just statically at any particular point in time.
# MAGIC
# MAGIC We can think of a few simple things that can be derived from this data for the actionable features:
# MAGIC - **Number of emails** that were sent for marketing purposes and received by the user.
# MAGIC - Portion of these emails that were actually **opened**.
# MAGIC - Portion of these emails where the user actually **clicked** on the link.
# MAGIC - **Number of products** that were viewed.
# MAGIC - Number of **propositions that were interacted with**.
# MAGIC - Number of **propositions that were dismissed**.
# MAGIC - Number of **links that were clicked on**.
# MAGIC
# MAGIC Regarding the temporal features, we can look at consecutive occurrences between various events. This can be accomplished using the `TIME_BETWEEN_PREVIOUS_MATCH` function. So we take the previous actionable features and look at their temporal distribution:
# MAGIC - Number of minutes between 2 consecutive emails received.
# MAGIC - Number of minutes between 2 consecutive emails opened.
# MAGIC - Number of minutes between 2 consecutive emails where the user actually clicked on the link.
# MAGIC - Number of minutes between 2 consecutive product views.
# MAGIC - Number of minutes between 2 propositions that were interacted with.
# MAGIC - Number of minutes between 2 propositions that were dismissed.
# MAGIC - Number of minutes between 2 links that were clicked on.
# MAGIC
# MAGIC Let's put all that in practice and look how we can create these features inside a query:

# COMMAND ----------

query_features = f"""
SELECT
    _{tenant_id}.userid as userId,
    SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "emailsReceived",
    SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "emailsOpened",       
    SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "emailsClicked",       
    SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "productsViewed",       
    SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "propositionInteracts",       
    SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "propositionDismissed",
    SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "webLinkClicks" ,
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_emailSent",
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_emailOpened",
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_emailClick",
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_productView",
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_propositionInteract",
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_propositionDismiss",
    TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
       OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
       AS "minutes_since_linkClick"
FROM {table_name}
"""

df_features = qs_cursor.query(query_features, output="dataframe")
display(df_features.head())

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we have defined all our features, and we also have our classes cleanly defined, so we can tie everything together in a final query that will represent our training set to be used later on on our ML model.

# COMMAND ----------

query_training_set = f"""
SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name} LIMIT 1000
)
WHERE (subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1)
ORDER BY timestamp
"""

df_training_set = qs_cursor.query(query_training_set, output="dataframe")
display(df_training_set.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Generating Features Incrementally
# MAGIC
# MAGIC Now in a typical ML workload you'll want to use incremental data to feed to your model, or data between some specific dates. For that purpose we can use snapshot information that is tracked inside Query Service every time a new batch of data is ingested, using the `history_meta` metadata table. For example, you can access the metadata for each batch of your dataset using the query below:

# COMMAND ----------

query_meta = f"""
SELECT * FROM (SELECT history_meta('{table_name}'))
"""

df_meta = qs_cursor.query(query_meta, output="dataframe")
print(f"Total number of snapshots/batches: {len(df_meta)}")
display(df_meta.head(n=128))

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's use that information to transform our featurization query into an incremental version of it. We can use [anonymous blocks](https://experienceleague.adobe.com/docs/experience-platform/query/sql/syntax.html?lang=en#anonymous-block) to create variables used to filter on the snapshots. Anonymous blocks are useful to embed multiple queries at once and do things like defining variables and such. We can then use Query Service's `SNAPSHOT BETWEEN x AND y` functionality to query data incrementally.
# MAGIC
# MAGIC In our case because this is the first time generating the features we will look at data between the most recent snapshot and its preceding one (which should correspond to the last batch of data ingested), but this can be extended to query data between any snapshots:

# COMMAND ----------

print(f"""
$$ BEGIN

SET @from_snapshot_id = SELECT parent_id FROM (SELECT history_meta('{table_name}')) WHERE is_current = true;
SET @to_snapshot_id = SELECT snapshot_id FROM (SELECT history_meta('{table_name}')) WHERE is_current = true;

SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
    SNAPSHOT BETWEEN @from_snapshot_id AND @to_snapshot_id
)
WHERE (subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1)
ORDER BY timestamp;

EXCEPTION
  WHEN OTHER THEN
    SELECT 'ERROR';

END $$;
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Note that we're not executing it interactively because this anonymous block is actually multiple queries chained together, and there is no simple way to return multiple result sets using PostgreSQL libraries.
# MAGIC
# MAGIC To solve that we can executed the query asynchronously and add a `CREATE TABLE x AS` statement at the beginning of our featurization query, to then look in that table. Note that because this is executed asynchronously, it goes into the Query Service scheduler and will take a few minutes to start executing, unlike the code we've been running until now which was synchronous and instant.

# COMMAND ----------

import time
import sys

ctas_table_name = f"cmle_example_training_set_incremental_{unique_id}"

query_training_set_incremental = f"""
$$ BEGIN

SET @from_snapshot_id = SELECT parent_id FROM (SELECT history_meta('{table_name}')) WHERE is_current = true;
SET @to_snapshot_id = SELECT snapshot_id FROM (SELECT history_meta('{table_name}')) WHERE is_current = true;

CREATE TABLE {ctas_table_name} AS
SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
    SNAPSHOT BETWEEN @from_snapshot_id AND @to_snapshot_id LIMIT 1000
)
WHERE (subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1)
ORDER BY timestamp;

EXCEPTION
  WHEN OTHER THEN
    SELECT 'ERROR';

END $$;
"""

# COMMAND ----------

query_incremental_res = qs.postQueries(
    name="[CMLE][Week2] Query to generate incremental training data",
    sql=query_training_set_incremental,
    dbname=f"{sandbox_name}:{table_name}"
)
query_incremental_id = query_incremental_res["id"]
print(f"Query started successfully and got assigned ID {query_incremental_id} - it will take some time to execute")

# COMMAND ----------

def wait_for_query_completion(query_id):
    while True:
        query_info = qs.getQuery(query_id)
        query_state = query_info["state"]
        if query_state in ["SUCCESS", "FAILED"]:
            break
        print("Query is still in progress, sleeping...")
        time.sleep(60)

    duration_secs = query_info["elapsedTime"] / 1000
    if query_state == "SUCCESS":
        print(f"Query completed successfully in {duration_secs} seconds")
    else:
        print(f"Query failed with the following errors:", file=sys.stderr)
        for error in query_info["errors"]:
            print(f"Error code {error['code']}: {error['message']}", file=sys.stderr)

# COMMAND ----------

wait_for_query_completion(query_incremental_id)

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> This should run in less than 10 minutes. If for whatever reason this does not finish in that time frame, it may get stuck creating the batch if the ingestion service is busy ingesting other data in your organization. We advise waiting longer or reaching out to your Adobe contact if this does not complete.
# MAGIC     
# MAGIC The same comment applies to subsequent cells where we use `wait_for_query_completion`.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC This `CREATE TABLE x AS` statement actually does several steps:
# MAGIC - It will create a brand **new dataset** in your Adobe Experience Platform organization and sandbox.
# MAGIC - The **schema** for this dataset will be created ad-hoc to **match the fields** of our featurization query, so there is no need to manually create schemas and fieldgroups for this.
# MAGIC
# MAGIC You can verify that by going in the UI at the link below, and making sure it you see the dataset as shown in the screenshot further down. The number of records should correspond to the batch size you used in the previous week since we are querying a single snapshot/batch.
# MAGIC
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> Re-executing the same query will fail, because it will try to create a table and not insert into it. We're solving this problem in the next section.
# MAGIC </div>

# COMMAND ----------

datasets_res = cat_conn.getDataSets(name=ctas_table_name)
if len(datasets_res) != 1:
    raise Exception(f"Expected a single dataset but got {len(datasets_res)} ones")
ctas_dataset_id = list(datasets_res.keys())[0]
ctas_dataset_link = get_ui_link(tenant_id, "dataset/browse", ctas_dataset_id)
display_link(ctas_dataset_link, f"{ctas_table_name} ({ctas_dataset_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ![CTAS](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week2-CTAS.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can just query it to see the structure of the data and verify it matches our query:

# COMMAND ----------

query_ctas = f"""
SELECT * FROM {ctas_table_name} LIMIT 10;
"""
qs = queryservice.QueryService()
qs_conn = qs.connection()
qs_conn['dbname'] = f'{sandbox_name}:{ctas_table_name}'
qs_cursor = queryservice.InteractiveQuery(qs_conn)
df_ctas = qs_cursor.query(query_ctas, output="dataframe")
display(df_ctas.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Templatizing the Featurization Query

# COMMAND ----------

# MAGIC %md
# MAGIC Now we've got a complete featurization query that can also be used to generate features incrementally, but we still want to go further:
# MAGIC - The **snapshot window should be configurable** and easy to change without having to constantly create new queries.
# MAGIC - The query itself should be **stored in a templatized way** so it can be referred to easily.
# MAGIC - The query should be able to **create the table** automatically as well as **inserting into** a pre-existing table.
# MAGIC
# MAGIC Query Service has this concept of [templates](https://experienceleague.adobe.com/docs/experience-platform/query/ui/query-templates.html?lang=en) that we will be leveraging in this section to satisfy the requirements mentioned above.

# COMMAND ----------

# MAGIC %md
# MAGIC The first step is to make make sure we either create the table if it does not exist, otherwise insert into it. This can be done by checking if the table exists using the `table_exists` function and adding a condition in our anonymous block based on that:

# COMMAND ----------

query_training_set_ctas_or_insert = f"""
$$ BEGIN

SET @from_snapshot_id = SELECT parent_id FROM (SELECT history_meta('{table_name}')) WHERE is_current = true;
SET @to_snapshot_id = SELECT snapshot_id FROM (SELECT history_meta('{table_name}')) WHERE is_current = true;
SET @my_table_exists = SELECT table_exists('{ctas_table_name}');

CREATE TABLE IF NOT EXISTS {ctas_table_name} AS
SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
    SNAPSHOT BETWEEN @from_snapshot_id AND @to_snapshot_id LIMIT 1000
)
WHERE (subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1)
ORDER BY timestamp;

INSERT INTO {ctas_table_name}
SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
    SNAPSHOT BETWEEN @from_snapshot_id AND @to_snapshot_id LIMIT 1000
)
WHERE 
    @my_table_exists = 't' AND
    ((subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1))
ORDER BY timestamp;

EXCEPTION
  WHEN OTHER THEN
    SELECT 'ERROR';

END $$;
"""

query_ctas_or_insert_res = qs.postQueries(
    name="[CMLE][Week2] Query to generate training data as CTAS or Insert",
    sql=query_training_set_ctas_or_insert,
    dbname=f"{sandbox_name}:all"
)
query_ctas_or_insert_id = query_ctas_or_insert_res["id"]
print(f"Query started successfully and got assigned ID {query_ctas_or_insert_id} - it will take some time to execute")

wait_for_query_completion(query_ctas_or_insert_id)

# COMMAND ----------

# MAGIC %md
# MAGIC The next step is to make the snapshot time window configurable. To do that we can replace the part containing the snapshot boundaries with variables as `$variable` so they can be passed at runtime using Query Service:

# COMMAND ----------

ctas_table_name = f"cmle_training_set_{unique_id}"

query_training_set_template = f"""
$$ BEGIN

SET @my_table_exists = SELECT table_exists('{ctas_table_name}');

CREATE TABLE IF NOT EXISTS {ctas_table_name} AS
SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
    SNAPSHOT BETWEEN $from_snapshot_id AND $to_snapshot_id
)
WHERE (subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1)
ORDER BY timestamp;

INSERT INTO {ctas_table_name}
SELECT *
FROM (
    SELECT _{tenant_id}.userid as userId, 
       eventType,
       timestamp,
       SUM(CASE WHEN eventType='web.formFilledOut' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid) 
           AS "subscriptionOccurred",
       SUM(CASE WHEN eventType='directMarketing.emailSent' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsReceived",
       SUM(CASE WHEN eventType='directMarketing.emailOpened' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsOpened",       
       SUM(CASE WHEN eventType='directMarketing.emailClicked' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "emailsClicked",       
       SUM(CASE WHEN eventType='commerce.productViews' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "productsViewed",       
       SUM(CASE WHEN eventType='decisioning.propositionInteract' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionInteracts",       
       SUM(CASE WHEN eventType='decisioning.propositionDismiss' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "propositionDismissed",
       SUM(CASE WHEN eventType='web.webinteraction.linkClicks' THEN 1 ELSE 0 END) 
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "webLinkClicks" ,
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailSent', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailSent",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailOpened', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailOpened",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'directMarketing.emailClicked', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_emailClick",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'commerce.productViews', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_productView",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'decisioning.propositionInteract', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionInteract",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'propositionDismiss', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_propositionDismiss",
       TIME_BETWEEN_PREVIOUS_MATCH(timestamp, eventType = 'web.webinteraction.linkClicks', 'minutes')
           OVER (PARTITION BY _{tenant_id}.userid ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
           AS "minutes_since_linkClick",
        row_number() OVER (PARTITION BY _{tenant_id}.userid ORDER BY randn()) AS random_row_number_for_user
    FROM {table_name}
    SNAPSHOT BETWEEN $from_snapshot_id AND $to_snapshot_id
)
WHERE 
    @my_table_exists = 't' AND
    ((subscriptionOccurred = 1 AND eventType = 'web.formFilledOut') OR (subscriptionOccurred = 0 AND random_row_number_for_user = 1))
ORDER BY timestamp;

EXCEPTION
  WHEN OTHER THEN
    SELECT 'ERROR';

END $$;
"""

# COMMAND ----------

# MAGIC %md
# MAGIC We're not executing it because it has actual variables in it that will need to be resolved at runtime, so executing it right now would fail. We're ready to turn this into a proper template, which requires the following:
# MAGIC - A **name** for your templatized query.
# MAGIC - Some set of **query parameters** that you might want to already save - in our case we're not setting any so both snapshot boundaries can be set at runtie.
# MAGIC - Your SQL **query**.
# MAGIC
# MAGIC Once you do this, the template should be available in the UI at the link below, as you can see in the screenshot.

# COMMAND ----------

all_templates = qs.getTemplates()
print(len(all_templates))

# COMMAND ----------

all_templates[0]["name"]

# COMMAND ----------

def get_template_by_name(name):
    all_templates = qs.getTemplates()
    for template in all_templates:
        if template["name"] == name:
            return template
    return None

def get_or_create_query_template(template_spec):
    template_name = template_spec["name"]
    template_res = get_template_by_name(template_name)
    if template_res is None:
        template_res = qs.createQueryTemplate(template_spec)
    return template_res

# COMMAND ----------

template_name = f"[CMLE][Week2] Template for training data created by {username}"
template_spec = {
    "sql": query_training_set_template,
    "queryParameters": {},
    "name": template_name
}
template_res = get_or_create_query_template(template_spec)
template_id = template_res["id"]
template_link = get_ui_link(tenant_id, "query/edit", template_id)

display_link(template_link, template_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Template](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week2-Template.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the template is saved, we can refer to it at any time, and passing any kind of values for the snapshots that we want. So for example if you have streaming data coming through your system, you just need to find out the beginning snapshot ID and end snapshot ID, and you can execute this featurization query that will take care of querying between these 2 snapshots.
# MAGIC
# MAGIC In this example, we'll just query the entire dataset, so the very first and very last snapshots:

# COMMAND ----------

query_snapshots = f"""
SELECT snapshot_id 
FROM (
    SELECT history_meta('{table_name}')
) 
WHERE is_current = true OR snapshot_generation = 0 
ORDER BY snapshot_generation ASC
"""

df_snapshots = qs_cursor.query(query_snapshots, output="dataframe")

snapshot_start_id = str(df_snapshots["snapshot_id"].iloc[0])
snapshot_end_id = str(df_snapshots["snapshot_id"].iloc[1])
print(f"Query will go from start snapshot ID {snapshot_start_id} to end snapshot ID {snapshot_end_id}")

display(df_snapshots.head())

# COMMAND ----------

query_final_res = qs.postQueries(
    name=f"[CMLE][Week2] Query to generate training data created by {username}",
    templateId=template_id,
    queryParameters={
        "from_snapshot_id": snapshot_start_id,
        "to_snapshot_id": snapshot_end_id,
    },
    dbname=f"{sandbox_name}:all"
)
query_final_id = query_final_res["id"]
print(f"Query started successfully and got assigned ID {query_final_id} - it will take some time to execute")

wait_for_query_completion(query_final_id)

# COMMAND ----------

# MAGIC %md
# MAGIC At that point we got our full featurized dataset that is ready to plug into a ML model. But it's still in an Adobe Experience Platform dataset so far, and we need to bring it back to our cloud storage account outside of the Experience Platform to use our tool of choice, which will be covered in the next section.
# MAGIC
# MAGIC Before we go through that, we just need to find te dataset ID corresponding to the output of our templatized query:

# COMMAND ----------

ctas_table_info = cat_conn.getDataSets(name=ctas_table_name)
created_dataset_id = list(ctas_table_info.keys())[0]
created_dataset_link = get_ui_link(tenant_id, "dataset/browse", created_dataset_id)

display_link(created_dataset_link, ctas_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Exporting the Featurized Dataset to Cloud Storage

# COMMAND ----------

# MAGIC %md
# MAGIC Now that our featurized data is in a dataset, we need to bring it out to an external cloud storage filesystem from which the ML model training and scoring will be performed. 
# MAGIC
# MAGIC For the purposes of this notebook we will be using the [Data Landing Zone (DLZ)](https://experienceleague.adobe.com/docs/experience-platform/sources/api-tutorials/create/cloud-storage/data-landing-zone.html?lang=en) as the filesystem. Every Adobe Experience Platform has a DLZ already setup as an [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs) container. We'll be using that as a delivery mechanism for the featurized data, but this step can be customized to delivery this data to any cloud storage filesystem.
# MAGIC
# MAGIC To setup the delivery pipeline, we'll be using the [Flow Service for Destinations](https://developer.adobe.com/experience-platform-apis/references/destinations/) which will be responsible for picking up the featurized data and dump it into the DLZ. There's a few steps involved:
# MAGIC - Creating a **source connection**.
# MAGIC - Creating a **target connection**.
# MAGIC - Creating a **data flow**.
# MAGIC
# MAGIC For that, again we use `aepp` to abstract all the APIs:

# COMMAND ----------

from aepp import flowservice

flow_conn = flowservice.FlowService()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Creating the Source Connection

# COMMAND ----------

# MAGIC %md
# MAGIC The source connection is responsible for connecting to your Adobe Experience Platform dataset so that the resulting flow will know exactly where to look for the data and in what format.

# COMMAND ----------

source_res = flow_conn.createSourceConnectionDataLake(
    name=f"[CMLE][Week2] Featurized Dataset source connection created by {username}",
    dataset_ids=[created_dataset_id],
    format="parquet"
)
source_connection_id = source_res["id"]
source_connection_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Creating the Target Connection

# COMMAND ----------

# MAGIC %md
# MAGIC The target connection is responsible for connecting to the destination filesystem. In our case, we want to connect to the DLZ and specify in what format the data will be stored, as well as the type of compression.
# MAGIC
# MAGIC Before we can create it however, we need to create a base connection to the DLZ. A base connection is just an instance of a connection spec that details how one authenticates to a particular destination. In our case, because we're using the DLZ which is a known entity internal to Adobe, we can just reference the standard DLZ connection spec ID and create an empty base connection.
# MAGIC
# MAGIC For reference, here is a list of all the connection specs available for the most popular cloud storage accounts (these IDs are global across every single customer account and sandbox):
# MAGIC
# MAGIC | Cloud Storage Type    | Connection Spec ID                   |
# MAGIC |-----------------------|--------------------------------------|
# MAGIC | Amazon S3             | 4fce964d-3f37-408f-9778-e597338a21ee |
# MAGIC | Azure Blob Storage    | 6d6b59bf-fb58-4107-9064-4d246c0e5bb2 |
# MAGIC | Azure Data Lake       | be2c3209-53bc-47e7-ab25-145db8b873e1 |
# MAGIC | Data Landing Zone     | 10440537-2a7b-4583-ac39-ed38d4b848e8 |
# MAGIC | Google Cloud Storage  | c5d93acb-ea8b-4b14-8f53-02138444ae99 |
# MAGIC | SFTP                  | 36965a81-b1c6-401b-99f8-22508f1e6a26 |

# COMMAND ----------

# TODO: implement in aepp a way to abstract that
connection_spec_id = "10440537-2a7b-4583-ac39-ed38d4b848e8"
base_connection_res = flow_conn.createConnection(data={
    "name": f"[CMLE][Week2] Base Connection to DLZ created by {username}",
    "auth": None,
    "connectionSpec": {
        "id": connection_spec_id,
        "version": "1.0"
    }
})
base_connection_id = base_connection_res["id"]
base_connection_id

# COMMAND ----------

# MAGIC %md
# MAGIC With that base connection, we're ready to create the target connection that will tie to our DLZ directory:

# COMMAND ----------

# TODO: implement in aepp a way to abstract that
target_res = flow_conn.createTargetConnection(
    data={
        "name": f"[CMLE][Week2] Data Landing Zone target connection created by {username}",
        "baseConnectionId": base_connection_id,
        "params": {
            "mode": "Server-to-server",
            "compression": compression_type,
            "datasetFileType": data_format,
            "path": export_path
        },
        "connectionSpec": {
            "id": connection_spec_id,
            "version": "1.0"
        }
    }
)

target_connection_id = target_res["id"]
target_connection_id

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> 
# MAGIC     
# MAGIC If you would like to switch to a different cloud storage, you need to update the `connection_spec_id` variable above to the matching value in the table mentioned earlier in this section.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Creating the Data Flow

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the source and target connections setup, we can construct the data flow. A data flow is the "recipe" that describes where the data comes from and where it should end up. We can also specify how often checks happen to find new data, but it cannot be lower than 3 hours currently for platform stability reasons. A data flow is tied to a flow spec ID which contains the instructions for transfering data in an optimized way between a source and destination.
# MAGIC
# MAGIC For reference, here is a list of all the flow specs available for the most popular cloud storage accounts (these IDs are global across every single customer account and sandbox):
# MAGIC
# MAGIC | Cloud Storage Type    | Flow Spec ID                         |
# MAGIC |-----------------------|--------------------------------------|
# MAGIC | Amazon S3             | 269ba276-16fc-47db-92b0-c1049a3c131f |
# MAGIC | Azure Blob Storage    | 95bd8965-fc8a-4119-b9c3-944c2c2df6d2 |
# MAGIC | Azure Data Lake       | 17be2013-2549-41ce-96e7-a70363bec293 |
# MAGIC | Data Landing Zone     | cd2fc47e-e838-4f38-a581-8fff2f99b63a |
# MAGIC | Google Cloud Storage  | 585c15c4-6cbf-4126-8f87-e26bff78b657 |
# MAGIC | SFTP                  | 354d6aad-4754-46e4-a576-1b384561c440 |
# MAGIC
# MAGIC In order to execute the data flow, There are two options available to you:
# MAGIC - If you do not want to wait you can do a **adhoc run** to execute it instantly in Section 4.4.
# MAGIC - Either **wait until it gets scheduled**. We selected to have it run every 3 hours, so you may need to wait up to 3 hours.
# MAGIC
# MAGIC We have selected the first option by default, if you would select the second option, please change the boolean on_schedule in below cell to True, skip the step for triggering adhoc run located the first cell in Section 4.4, wait up to 3 hours and execute the cells after.

# COMMAND ----------

import time
on_schedule = False
if on_schedule:
    schedule_params = {
        "interval": 3,
        "timeUnit": "hour",
        "startTime": int(time.time())
    }
else:
    schedule_params = {
        "interval": 1,
        "timeUnit": "day",
        "startTime": int(time.time() + 60*60*24*365)
    }


flow_spec_id = "cd2fc47e-e838-4f38-a581-8fff2f99b63a"
flow_obj = {
    "name": f"[CMLE][Week2] Flow for Featurized Dataset to DLZ created by {username}",
    "flowSpec": {
        "id": flow_spec_id,
        "version": "1.0"
    },
    "sourceConnectionIds": [
        source_connection_id
    ],
    "targetConnectionIds": [
        target_connection_id
    ],
    "transformations": [],
    "scheduleParams": {
        "interval": 3,
        "timeUnit": "hour",
        "startTime": int(time.time())
    }
}
flow_res = flow_conn.createFlow(
    obj = flow_obj,
    flow_spec_id = flow_spec_id
)
dataflow_id = flow_res["id"]
dataflow_id

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> 
# MAGIC
# MAGIC If you would like to switch to a different cloud storage, you need to update the `flow_spec_id` variable above to the matching value in the table mentioned earlier in this section.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC After you create the data flow, you should be able to see it in the UI to monitor executions, runtimes and its overall lifecycle. You can get the link below and should be able to see it in the UI as shown in the screenshot as well.

# COMMAND ----------

dataflow_link = get_ui_link(tenant_id, "destination/browse", dataflow_id)
display_link(dataflow_link, f"Data Flow ID {dataflow_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ![Dataflow](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week2-Dataflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Executing the Data Flow

# COMMAND ----------

# MAGIC %md
# MAGIC At this point we've just created our Data Flow, but it has not executed yet. Please follow the instructions for the option you selected in Section 4.3 :
# MAGIC - If you do not want to wait you can do a **adhoc run** to execute it instantly.
# MAGIC - Either **wait until it gets scheduled**. We selected to have it run every 3 hours, so you may need to wait up to 3 hours.
# MAGIC
# MAGIC In the cell below we're showing how to do the first option to trigger a adhoc run, if you selected the second option, you can skip the cell below and will need to wait up to 3 hours to execute the cells after.

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> Please wait at least 10 minutes after creating the dataflow before triggering the next cell, otherwise the job might not execute at all.
# MAGIC </div>

# COMMAND ----------

# TODO: use new functionality in aepp when it is released
from aepp import connector

connector = connector.AdobeRequest(
    config_object=aepp.config.config_object,
    header=aepp.config.header,
    loggingEnabled=False,
    logger=None,
)

endpoint = aepp.config.endpoints["global"] + "/data/core/activation/disflowprovider/adhocrun"

payload = {
    "activationInfo": {
        "destinations": [
            {
                "flowId": dataflow_id, 
                "datasets": [
                    {"id": created_dataset_id}
                ]
            }
        ]
    }
}

connector.header.update({"Accept":"application/vnd.adobe.adhoc.dataset.activation+json; version=1"})
activation_res = connector.postData(endpoint=endpoint, data=payload)
activation_res

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> 
# MAGIC
# MAGIC If you see an error such as `Invalid parameter: Flow for id 93790efa-645b-4400-8afe-b6f135734656 is incorrect. Error is [Adhoc run can not be executed for Flow spec=cd2fc47e-e838-4f38-a581-8fff2f99b63a.`. it means your cloud storage is not yet whitelisted for exporting datasets. Please reach out to your Adobe contact to have it enabled.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> 
# MAGIC
# MAGIC If you see an error such as `Invalid parameter: Following order ID(s) are not ready for dataset export, please wait for 10 minutes and retry.`. it means you need to wait a few minutes and retry again.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC <div class="alert alert-block alert-warning">
# MAGIC <b>Note:</b> 
# MAGIC
# MAGIC If you get a message saying a run already exists, it means that either this dataset has been exported already based on the schedule, or that you've already done an adhoc export before.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can check the execution of our Data Flow to make sure it actually executes. You can run the following cell until you can see the run appear.

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
            run_size_mb = run["metrics"]["sizeSummary"]["outputBytes"] / 1024. / 1024.
            run_num_rows = run["metrics"]["recordSummary"]["outputRecordCount"]
            run_num_files = run["metrics"]["fileSummary"]["outputFileCount"]
            print(f"Run ID {run_id} completed with: duration={run_duration_secs} secs; size={run_size_mb} MB; num_rows={run_num_rows}; num_files={run_num_files}")
        finished = True
    except Exception as e:
        print(f"No runs completed yet for flow {dataflow_id}")
        time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that a run of our Data Flow has executed successfully, we're all set! We can do a sanity check to verify that the data indeed made its way into the DLZ. For that, we recommend setting up [Azure Storage Explorer](https://azure.microsoft.com/en-us/products/storage/storage-explorer) to connect to your DLZ container using [this guide](https://experienceleague.adobe.com/docs/experience-platform/destinations/catalog/cloud-storage/data-landing-zone.html?lang=en). To get the credentials, you can execute the code below to get the SAS URL needed:

# COMMAND ----------

# TODO: use functionality in aepp once released
from aepp import connector

connector = connector.AdobeRequest(
    config_object=aepp.config.config_object,
    header=aepp.config.header,
    loggingEnabled=False,
    logger=None,
)

endpoint = aepp.config.endpoints["global"] + "/data/foundation/connectors/landingzone/credentials"

dlz_credentials = connector.getData(endpoint=endpoint, params={
    "type": "dlz_destination"
})
dlz_container = dlz_credentials["containerName"]
dlz_sas_token = dlz_credentials["SASToken"]
dlz_storage_account = dlz_credentials["storageAccountName"]
dlz_sas_uri = dlz_credentials["SASUri"]
print(f"DLZ container: {dlz_container}")
print(f"DLZ storage account: {dlz_storage_account}")
print(f"DLZ SAS URL: {dlz_sas_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Once setup you should be able to see your featurized data as a set of Parquet files under the following directory structure: `cmle/egress/$DATASETID/exportTime=$TIMESTAMP` - see screenshot below.

# COMMAND ----------

print(f"Featurized data in DLZ should be available under {export_path}/{created_dataset_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ![DLZ](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week2-ExportedDataset.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Saving the featurized dataset to the configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we got everything working, we just need to save the `created_dataset_id` variable in the original configuration file, so we can refer to it in the following weekly assignments. To do that, execute the code below:

# COMMAND ----------

config.set("Platform", "featurized_dataset_id", created_dataset_id)

with open(config_path, "w") as configfile:
    config.write(configfile)
