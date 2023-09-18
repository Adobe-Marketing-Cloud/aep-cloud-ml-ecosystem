# Databricks notebook source
# MAGIC %md
# MAGIC # Scope of Notebook
# MAGIC
# MAGIC This notebook allows you to plug in your featuried dataset from the previous week into an ml model, in this case we use random forest.  You will then be able to store the trained model in mlflow and calculate performance characteristics around the model like AUC and accuracy.  The advanced section of this notebook outlines how to retrieve the best set of hyperparameters to certify a model for production use.

# COMMAND ----------

# MAGIC %md
# MAGIC ![ml-model-train](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week3-Workflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Before we run anything, make sure to install the following required libraries for this notebook. They are all publicly available libraries and the latest version should work fine.

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook requires some configuration data to properly authenticate to your Adobe Experience Platform instance. You should be able to find all the values required above by following the Setup section of the **README**.
# MAGIC
# MAGIC The next cell will be looking for your configuration file under your **ADOBE_HOME** path to fetch the values used throughout this notebook. See more details in the Setup section of the **README** to understand how to create your configuration file.

# COMMAND ----------

# MAGIC %run ./CommonInclude

# COMMAND ----------

# MAGIC %md
# MAGIC Before any calls can take place, we need to configure the library and setup authentication credentials. For this you'll need the following piece of information. For information about how you can get these, please refer to the `Setup` section of the **Readme**:
# MAGIC - Client ID
# MAGIC - Client secret
# MAGIC - Private key
# MAGIC - Technical account ID

# COMMAND ----------

# MAGIC %md
# MAGIC The private key needs to be accessible on disk from this notebook. We recommend uploading it to DBFS and refering to it with the `/dbfs` prefix. This can be achieved by clicking in the Databricks notebook interface on `File > Upload data to DBFS` and then selecting the **private.key** file you downloaded during the setup, click `Next` and then you should have the option to copy the path. Make sure it starts with `/dbfs/FileStore` - for example if you uploaded your private key into `/FileStore/shared_upload/your_username` then the final path should be `/dbfs/FileStore/shared_uploads/your_username/private.key`. Copy that value into the cell `Private Key Path` at the very top of this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Running a model on AEP data

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous week we generated our featurized data in the Data Landing Zone under the `dlz-destination` container. We can now read it so we can use it to train our ML model. Because this data can be pretty big, we want to first read it via a Spark dataframe, so we can then use a sample of it for training.
# MAGIC
# MAGIC The featurized data exported into the Data Landing Zone is under the format **cmle/egress/$DATASETID/exportTime=$EXPORTTIME**. We know the dataset ID which is in your config under `featurized_dataset_id` so we're just missing the export time so we know what to read. To get that we can simply list files in the DLZ and find what the value is. The first step is to retrieve the credentials for the DLZ related to the destination container:

# COMMAND ----------

from aepp import connector

connector = connector.AdobeRequest(
    config_object=aepp.config.config_object,
    header=aepp.config.header,
    loggingEnabled=False,
    logger=None)

dlz_endpoint = (
    aepp.config.endpoints["global"]
    + "/data/foundation/connectors/landingzone/credentials")

dlz_credentials = connector.getData(endpoint=dlz_endpoint, params={"type": "dlz_destination"})
dlz_container = dlz_credentials["containerName"]
dlz_sas_token = dlz_credentials["SASToken"]
dlz_storage_account = dlz_credentials["storageAccountName"]
dlz_sas_uri = dlz_credentials["SASUri"]

# COMMAND ----------

# MAGIC %md
# MAGIC Now we use some Python libraries to authenticate and issue listing commands so we can get the paths and extract the time from it.

# COMMAND ----------

from adlfs import AzureBlobFileSystem
from fsspec import AbstractFileSystem

abfs = AzureBlobFileSystem(account_name=dlz_storage_account, sas_token=dlz_sas_token)
export_time = get_export_time(abfs, dlz_container, export_path, featurized_dataset_id)
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
# MAGIC Now that we have a dataframe representing our features from AEP, we can use them to create or update a feature table in [Unity Catalog](https://www.databricks.com/product/unity-catalog) via [Databricks Feature Store](https://docs.databricks.com/en/machine-learning/feature-store/index.html).

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

# We need an instance of the feature store client to use the API.
fs = feature_store.FeatureStoreClient()
feature_table_name = "user_propensity_features"

# Create the feature table if its not already there, otherwise just update it.
if not spark.catalog.tableExists(feature_table_name):
    fs.create_table(
        name=feature_table_name,
        primary_keys=["userId"],
        description="Features about email and marketing response of users.",
        df=dlz_input_df)
else:
    if spark.table(feature_table_name).isEmpty():
        fs.write_table(
            name=feature_table_name, 
            df=dlz_input_df,
            mode="merge")

# Compose and display a link to the feature table.
catalog_name = spark.catalog.currentCatalog()
database_name = spark.catalog.currentDatabase()
feature_table_link = f"/explore/data/{catalog_name}/{database_name}/{feature_table_name}"
display_link(feature_table_link, "User Propensity Features in Unity Catalog")

# Finally, let's look at a dataframe read in from the feature table.
feature_df = fs.read_table(feature_table_name)
display(feature_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can then sample it to keep only a portion of the data for training before we bring the data in memory for use in the `scikit-learn` library. Here we're just going to use a sampling ratio of 50%, but you are welcome to use a bigger or smaller ratio. We use sampling **without** replacement to ensure the same profiles don't get picked up multiple times.

# COMMAND ----------

sampling_ratio = 0.5
sample_df = feature_df.sample(withReplacement=False, fraction=sampling_ratio)
display(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Creating baseline models as experiments in MLFlow

# COMMAND ----------

# MAGIC %md
# MAGIC Before doing any ML we can look at summary statistics to understand the structure of the data, and what kind of algorithm(s) might be suited to solve the problem. [Databricks profiling](https://www.databricks.com/blog/2021/12/07/introducing-data-profiles-in-the-databricks-notebook.html) tools are well suited to this task. One way to access the profiling tools is via the visualization dropdown any time you display a dataframe, as shown in the screenshot below.
# MAGIC
# MAGIC ![databricks-profiling](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week3-DatabricksProfiler.png)
# MAGIC
# MAGIC Another way is to call it directly in a notebook cell as shown below via `dbutils.data.summarize`. 
# MAGIC That's the approach we'll take in the following cell. Further information about this command
# MAGIC can be found in [the documentation](https://docs.databricks.com/en/dev-tools/databricks-utils.html#summarize-command-dbutilsdatasummarize).

# COMMAND ----------

dbutils.data.summarize(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC By default, the MLflow Python client creates models in the Databricks workspace model registry. However, to take better advantage of the Lakehouse AI platform's ability to provide end-to-end lineage and governance of our both our data and AI assets, we'll be storing our [model in Unity Catalog](https://docs.databricks.com/en/mlflow/models-in-uc.html). This means we'll be using the three level namespace of _catalog\_name.schema\_name.model\_name_. The model name for this example was already created for us based on the catalog and schema we're already using, but let's print it out here so you'll know what to go look for.
# MAGIC
# MAGIC Note that the common include setup has also already taken care of specifying that we 
# MAGIC want to use model registry by setting our registry uri with `mlflow.set_registry_uri("databricks-uc")`.

# COMMAND ----------

displayHTML(f"""The model will be registered as <b>{model_name}</b>.""")

# COMMAND ----------

# MAGIC %md
# MAGIC To use data from the feature store in our model, we create a training set using the feature store client, passing a set of the feature lookups we want to use for that training set. This metadata will be hermetically sealed with the rest of the model metadata when we log the model. This enables the feature store to track lineage of the feature and models, and also helps streamline downstream scoring of the model. 
# MAGIC
# MAGIC At scoring time, all we need to pass is the keys for the batch we want to score, and feature store will automatically join in all the features needed to run inference. This is helpful because it enables data scientists to publish new models that require new features without requiring changes to downstream inference code, as well as helping to avoid skew between training and serving, since we know for sure they'll be using the same feature generation logic.
# MAGIC
# MAGIC Note that our Feature Store table will also be stored in Unity Catalog.

# COMMAND ----------

feature_names = [
    'emailsReceived', 'emailsOpened', 'emailsClicked', 
    'productsViewed', 
    'propositionInteracts', 'propositionDismissed', 'webLinkClicks', 
    'minutes_since_emailSent', 'minutes_since_emailOpened', 'minutes_since_emailClick', 
    'minutes_since_productView', 'minutes_since_propositionInteract', 
    'minutes_since_propositionDismiss', 'minutes_since_linkClick'
]

label_name = "subscriptionOccurred"

feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="userId",
        feature_names=feature_names,
    )
]

training_set = fs.create_training_set(
    df=sample_df.select("userId", label_name),
    feature_lookups=feature_lookups,
    label=label_name)

# COMMAND ----------

# MAGIC %md
# MAGIC In order to feed data to our model, we need to do a few preparation steps:
# MAGIC - Separate the target variable (which in our case is whether a subscription occured or not) from the other variables.
# MAGIC - Split the data into a training and test set so we can evaluate our model performance down the line.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# From the training set instance, we can load a spark dataframe and then
# convert that to the pandas dataframe we'll use for training our scikit-learn model.
df_train = training_set.load_df().toPandas()

# Feature Selection
userIds, X, y = (
    df_train["userId"],
    df_train[feature_names].fillna(0), 
    df_train[label_name])

# Train test split
ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
    userIds, X, y, train_size=0.8, random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). Whereas before we needed to create a PythonModel wrapper in order to call `predict_proba`, we can now simply pass it as an argument called `pyfunc_predict_fn` when [logging](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model) the model.
# MAGIC
# MAGIC For reference, here's what a wrapper would have looked like if we weren't using the new `pyfunc_predict_fn` parameter:
# MAGIC
# MAGIC     import mlflow
# MAGIC     import mlflow.pyfunc
# MAGIC
# MAGIC     class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
# MAGIC         def __init__(self, model):
# MAGIC             self.model = model
# MAGIC
# MAGIC         def predict(self, context, model_input):
# MAGIC             return self.model.predict_proba(model_input)[:, 1]
# MAGIC
# MAGIC  Using [custom pyfunc models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom) is still useful if there is additional pre-processing or post-processing that needs to be applied beyond that, or for any other custom processing, but this case was common enough that the MLflow added `pyfunc_predict_fn` as a special keyword argument for _scikit-learn_ models to help remove some of the boilerplate required.

# COMMAND ----------

# MAGIC %md
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

import cloudpickle
import sklearn

import mlflow
import mlflow.pyfunc
import mlflow.sklearn

run_name_untuned = f"{model_name}_untuned"
rf_model_artifact_path = "random_forest_model"

# mlflow.start_run creates a new MLflow run to track the performance of this model.
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name=run_name_untuned) as untuned_run:
    n_estimators = 10
    rf_clf = RandomForestClassifier(n_estimators=n_estimators)
    rf_clf.fit(X_train, y_train)

    mlflow.log_param("n_estimators", n_estimators)

    accuracy = rf_clf.score(X_test, y_test)
    
    # Use the accuracy as a metric in MLFlow
    mlflow.log_metric("accuracy", accuracy)
    print("Random Forest (RF) Accuracy:", accuracy)

    # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = rf_clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions_test)

    # Use the area under the ROC curve as a metric in MLFlow
    mlflow.log_metric("auc", auc_score)
    print("Random Forest (RF) AUC score:", auc_score)

    # Log the model with a signature that defines the schema of the model's inputs and outputs.
    # When the model is deployed, this signature will be used to validate inputs.
    signature = infer_signature(X_train, rf_clf.predict(X_train))

    # MLflow contains utilities to create a conda environment used to serve models.
    # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__),
            "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None)
    
    fs.log_model(
        model=rf_clf,
        artifact_path=rf_model_artifact_path,
        flavor=mlflow.sklearn,
        training_set=training_set,
        conda_env=conda_env,
        signature=signature,
        pyfunc_predict_fn="predict_proba")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Registering models in Unity Catalog with MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a baseline model, we can register it in Unity Catalog so we can easily 
# MAGIC fetch them again later to compare with future iterations of the models once we do more tuning. 
# MAGIC By registering this model in Unity Catalog, you can share the model securely from anywhere 
# MAGIC within Databricks, across all your workspaces attached to this Unity Catalog metastore 
# MAGIC (i.e., in this region).
# MAGIC
# MAGIC Let's start with the Random Forest model. We need the `run_id`, which we can get from the 
# MAGIC `untuned_run` object we captured when we started the run above. Additionally, we also need 
# MAGIC the artifact path of the model within the run. We use the `run_id` and the `artifact_path` 
# MAGIC to compose the `model_uri` which we'll use to reference the run from MLflow experiment tracking.
# MAGIC
# MAGIC Now we're ready to register it. To register we need 2 pieces of information:
# MAGIC - The path under which the model artifacts were stored.
# MAGIC - The name of the model we'd like to register it under.

# COMMAND ----------

import time

rf_model_name = model_name
rf_run_id = untuned_run.info.run_id
rf_model_run_uri = f"runs:/{rf_run_id}/{rf_model_artifact_path}"
rf_model_version = mlflow.register_model(rf_model_run_uri, rf_model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Tuning models at scale and selecting production model

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have our baseline models computed and saved, we need to tune them and find the best performing model out of all of them. Each of these 2 algorithms has a lot of hyper-parameters, and different combinations of those can yield vastly different models with dramatically different performance characteristics. Because it wouldn't be possible to try all these combinations manually, we are leveraging Databricks' large-scale computing capabilities to distribute the hyper-parameter tuning process.
# MAGIC
# MAGIC The library `hyperopt` provides a good way to create efficient and scalable hyper-parameter tuning process, and integrates with Apache Spark via the use of `SparkTrials` to distribute the workload.
# MAGIC
# MAGIC The first thing we need to figure out is the degree of parallelism to use. There is a trade-off here:
# MAGIC - Greater parallelism will lead to speedups, but a less optimal hyper-parameter sweep.
# MAGIC - Lower parallelism will be slower but will do a better job trying the various combinations of hyper-parameters.
# MAGIC
# MAGIC A rule of thumb is to determine the number of trials you want to run, and then set the degree of parallelism to be the square root of that.
# MAGIC
# MAGIC Remember to factor in the size of the cluster you are on when determining the parallelism setting. You can only simultaneously run as many trials as there are cores in your cluster. Also, its generally a good idea to turn off autoscaling on your cluster and just explicitly set the number of workers you want for your run.

# COMMAND ----------

import math

from hyperopt import SparkTrials

# Feel free to change max_evals if you want fewer/more trial runs
max_evals = 256
parallelism = int(math.sqrt(max_evals))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we put everything together, and we'll be tracking each hyper-parameter tuning trial into a separate MLFlow experiment so we can easily refer to them later to find out the best-performing ones. Each parameter configuration will be saved in MLFlow, so we do not need to save anything manually.
# MAGIC
# MAGIC One thing we need to determine is the search space for the various hyper-parameters, which can greatly impact the quality of the distributed tuning. You can refer to the official API pages for [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to find out the range of acceptable values. We implement distributions following these definitions to define the search space below, and create a single search space that includes the choice of algorithm so we can run trials for both Random Forest in a distributed tuning job.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

rf_space = {
    "model": "random_forest",
    "kwargs": {
        "n_estimators": scope.int(hp.quniform("rf_n_estimators", 10, 200, 10)),
        "max_depth": scope.int(hp.quniform("rf_max_depth", 2, 12, 1)),
        "criterion": hp.choice("rf_criterion", ["gini", "entropy"]),
        "min_samples_leaf": scope.int(hp.uniform("rf_min_samples_leaf", 1, 5)),
        "min_samples_split": scope.float(hp.uniform("rf_min_samples_split", 0.01, 0.05)),
    },
}

search_space = {"model_choice": hp.choice("model_choice", [rf_space])}

models = {"random_forest": RandomForestClassifier}

# COMMAND ----------

# MAGIC %md
# MAGIC All that's left is to define the objective function that will be distributed in the cluster. `hyperopt` will be passing a sample which contains all the hyper-parameters chosen for a given trial (including the choice of algorithm), and we translate from that sample to an initialized model. Because in the search space we used keys matching the parameters for these algorithms, we can directly unpack the dictionary to get a model initialized with the requested hyper-parameters.
# MAGIC
# MAGIC Depending on your `max_evals` value earlier and the size of your Databricks cluster this could take some time.
# MAGIC
# MAGIC Now that we've identified the best set of hyperparameters using our validation split, let's do one final training run using our full training set and get a final set of offline evaluation metrics using our hold-out test set. We'll then log this one as our best model training run. 

# COMMAND ----------

from sklearn.model_selection import cross_val_score


def sample_to_model(sample):
    kwargs = sample["model_choice"]["kwargs"]
    return models[sample["model_choice"]["model"]](**kwargs)


def objective_fn(sample):
    with mlflow.start_run(nested=True):
        model_name = sample["model_choice"]["model"]
        mlflow.set_tag("model_choice", model_name)

        model = sample_to_model(sample)
        auc_score = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=5).mean()
        mlflow.log_metric("auc", auc_score)

        # Set the loss to -1*auc_score so fmin maximizes the auc_score
        return {"status": STATUS_OK, "loss": -1 * auc_score}


spark_trials = SparkTrials(parallelism=parallelism)

with mlflow.start_run(run_name=f"{model_name}_hyperopt_tuning") as hero_run:
    best_params = fmin(
        fn=objective_fn,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=spark_trials)

    print(best_params)
    eval_params = space_eval(search_space, best_params)
    print(eval_params)
    params = eval_params["model_choice"]["kwargs"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    mlflow.log_params(params)
    predictions_test = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric("auc", auc_score)

    # Log the model using the feature store API
    fs.log_model(
        model=model,
        artifact_path="best_model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        conda_env=conda_env,
        signature=signature,
        pyfunc_predict_fn="predict_proba")

# COMMAND ----------

# MAGIC %md
# MAGIC After the computations have completed, you'll have access to a lot of information:
# MAGIC - What is your best-performing model, what are its hyper-parameters, and what is its AUC.
# MAGIC - How do the various hyper-parameters interact with each other.
# MAGIC - Is there any of the models that performs better on aggregate.
# MAGIC
# MAGIC A lot of this information can be found in MLFlow by going into the Databricks **Experiments**, selecting the experiment group (typically the name of this notebook), and then expanding the experiment container that we started with `hyperopt` and selecting the individual trials here. After clicking on `Compare` you will see a few visualizations to understand the results of the tuning better.
# MAGIC
# MAGIC ![mlflow-model-comparison](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week3-MLflowModelComparison.png)
# MAGIC
# MAGIC To complete our understanding of the tuning, we'll do a scatter plot to represent the AUC (or loss) for each of the trials. This can help identify whether there are some outliers, and if there is a general trend emerging that could be caught by visual inspection. We use `plotly` to draw the graph, and also add additional hover data to each scatter point representing the hyper-parameters used for that trial.

# COMMAND ----------

import pandas as pd
import numpy as np

import plotly.express as px

def unpack(x):
    if x:
        return x[0]
    return np.nan
  
def add_hover_data(fig, df, model_choice, ignore_cols=["loss", "trial_number", "model_choice"]):
  fig.update_traces(
      customdata = df.loc[df["model_choice"] == model_choice],
      hovertemplate = "<br>".join(
          [
              f"{col}: %{{customdata[{i}]}}"
              for i, col in enumerate(df.columns) if not trials_df.loc[trials_df["model_choice"] == model_choice][col].isnull().any() 
            and col not in ignore_cols
          ]
      ),
      selector = {"name": model_choice},
  )
  return fig


trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in spark_trials])
trials_df["loss"] = [t["result"]["loss"] for t in spark_trials]
trials_df["trial_number"] = trials_df.index
trials_df["model_choice"] = trials_df["model_choice"].apply(
    lambda x: "random_forest" if x == 0 else "gradient_boosting"
)
fig = px.scatter(trials_df, x="trial_number", y="loss", color="model_choice")
fig = add_hover_data(fig, trials_df, "random_forest")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Though we have the `best_run` from earlier from when we logged it explicitly, here we show an alternative way to retrieve it. There may be a variety of reasons why you may want to search across runs in MLflow, this being one of them.
# MAGIC
# MAGIC Finally, we find and record the best overall performing model. Because this one is the best overall, we create a brand new model version in the Model Registry and this is the model we'll be using going forward for scoring.

# COMMAND ----------

best_run_filter = f'tags.mlflow.runName = "{model_name}_hyperopt_tuning"'
best_run_global = mlflow.search_runs(
    filter_string=best_run_filter, 
    order_by=['metrics.auc DESC']).iloc[0]
best_run_id_global = best_run_global.run_id
best_auc_global = best_run_global["metrics.auc"]
print(f"Best global run ID: {best_run_id_global}")
print(f"Best global AUC: {best_auc_global}")

top_model_name = model_name
top_model_version = mlflow.register_model(f"runs:/{best_run_id_global}/best_model", top_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Previously, in the Workspace MLflow Model Registry, we had a concept of stages of a registered model 
# MAGIC as a means of marking a particular version as either _Production_ or _Staging_. Recent versions of 
# MAGIC MLflow, and in particular the Unity Catalog implementation of the Model Registry, we instead use a
# MAGIC concept of [model aliases](https://mlflow.org/docs/latest/model-registry.html#using-registered-model-aliases)
# MAGIC to mark a model version. You can create your own alias names and you can use
# MAGIC more than the two names that were used for stages previously. Common choices include _Champion_ and
# MAGIC _Challenger_ to mark one model as _the current best one we know of so far used for most traffic_ and
# MAGIC the other as _one that might be better than the champion but we need to test it to find out_. 
# MAGIC
# MAGIC Because this model is our top model after tuning, we are ready to promote it to become the _Champion_. 
# MAGIC Promoting it will help refer to it when calling MLflow, and also will show it as having the _Champion_ 
# MAGIC alias in Unity Catalog.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name=top_model_name, 
    alias="Champion", 
    version=top_model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC The Models page now shows the model version in stage **Production**
# MAGIC
# MAGIC ![mlflow-model-comparison](/files/static/7cf4bf44-5482-4426-a3b3-842be2f737b1/media/CMLE-Notebooks-Week3-ModelAliasApplied.png)
# MAGIC
# MAGIC You can now refer to the model using the path **models:/${catalog}.{schema}.{model}@Champion**

# COMMAND ----------

prod_model_uri = f"models:/{top_model_name}@Champion"

test_df = spark.createDataFrame(
    pd.DataFrame({
        "userId": ids_test,
        label_name: y_test}))

pred_df = (
    fs.score_batch(
        model_uri=prod_model_uri, 
        df=test_df,
        result_type="array<double>")
    .withColumn("prediction", F.col("prediction")[1])
    .cache())

display(pred_df.select("userId", label_name, "prediction"))

# COMMAND ----------

pred_pandas_df = pred_df.select("userId", label_name, "prediction").toPandas()
y_test = pred_pandas_df[label_name]
y_pred = pred_pandas_df["prediction"]

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, y_pred)}')

# COMMAND ----------

# MAGIC %md
# MAGIC We can do one last validation on our best model to make sure it doesn't overfit. For that we are computing the confusion matrix to see the share of predictions who are false positives or false negatives which we'd like to minimize.

# COMMAND ----------

threshold = 0.5
full_pred_df = (
    fs.score_batch(
        model_uri=prod_model_uri,
        df=sample_df.select("userId", label_name),
        result_type="array<double>")
    .withColumn("prediction", F.when(F.col("prediction")[1] > threshold, 1).otherwise(0)))

full_pred_pandas_df = full_pred_df.select(label_name, "prediction").toPandas()
full_y_pred = full_pred_pandas_df["prediction"]
full_y_true = full_pred_pandas_df[label_name]

# COMMAND ----------

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

matrix = confusion_matrix(full_y_pred, full_y_true)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
cm_display = ConfusionMatrixDisplay(matrix, display_labels=["notSubscribed", "subscribed"])
cm_display.plot(ax=ax);

# COMMAND ----------

# MAGIC %md
# MAGIC This confusion matrix makes for useful documentation. Let's store it as an
# MAGIC artifact alongside our model in this run in our experiment. Note that we could
# MAGIC also store additional plots both at the parent run level or against any of the
# MAGIC child runs we were performing during our hyperparameter sweep above.

# COMMAND ----------

with mlflow.start_run(run_id=best_run_id_global) as hero_run:
    mlflow.log_figure(fig, "confusion_matrix.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Saving the final model name to configuration
# MAGIC
# MAGIC Now that we got everything working, we just need to save the updated `model_name` variable in the original configuration file, so we can refer to it in the following weekly assignments. To do that, execute the code below:

# COMMAND ----------

config.set("Cloud", "model_name", model_name)

with open(config_path, "w") as configfile:
    config.write(configfile)
