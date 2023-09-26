# Databricks notebook source
import aepp.connector

if not hasattr(aepp.connector.AdobeRequest, "getData_orig"):
    aepp.connector.AdobeRequest.getData_orig = aepp.connector.AdobeRequest.getData

    def getDataPatched(
        self,
        endpoint: str,
        params: dict = None,
        data: dict = None,
        headers: dict = None,
        *args,
        **kwargs,
    ):
        # "fieldgroups/" was fixed in 0.3.1.post5
        # .. but leaving patch could still help in case someone has older version installed
        targets = ["fieldgroups/", "schemas/", "descriptors/"]
        if params and "start" in params and any(t in endpoint for t in targets):
            if self.logger:
                self.logger.debug("patching params to rename start to page")
            params["page"] = params["start"]
            del params["start"]
        return self.getData_orig(endpoint, params, data, headers, *args, **kwargs)

    aepp.connector.AdobeRequest.getData = getDataPatched
    print("aepp.connector.AdobeRequest.getData patched")
else:
    print("aepp.connector.AdobeRequest.getData already patched")

# COMMAND ----------

import os
from configparser import ConfigParser

if "ADOBE_HOME" not in os.environ:
    raise Exception("ADOBE_HOME environment variable needs to be set.")

config = ConfigParser()
config_path = os.path.join(os.environ["ADOBE_HOME"], "conf", "config.ini")

if not os.path.exists(config_path):
    raise Exception(f"Looking for configuration under {config_path} but config not found, please verify path")

config.read(config_path)
  
ims_org_id = config.get("Platform", "ims_org_id")
sandbox_name = config.get("Platform", "sandbox_name")
environment = config.get("Platform", "environment")
client_id = config.get("Authentication", "client_id")
client_secret = config.get("Authentication", "client_secret")
private_key_path = config.get("Authentication", "private_key_path")
tech_account_id = config.get("Authentication", "tech_acct_id")
dataset_id = config.get("Platform", "dataset_id")
featurized_dataset_id = config.get("Platform", "featurized_dataset_id")
scoring_dataset_id = config.get("Platform", "scoring_dataset_id")
export_path = config.get("Cloud", "export_path")
import_path = config.get("Cloud", "import_path")
data_format = config.get("Cloud", "data_format")
compression_type = config.get("Cloud", "compression_type")
model_name = config.get("Cloud", "model_name")

if not os.path.exists(private_key_path):
    raise Exception(f"Looking for private key file under {private_key_path} but key not found, please verify path")


# COMMAND ----------

import re

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
username = current_user[:current_user.rfind('@')]
unique_id = s = re.sub("[^0-9a-zA-Z]+", "_", username)

print(f"current_user: {current_user}")
print(f"username: {username}")
print(f"unique_id: {unique_id}")

# COMMAND ----------

import aepp

aepp.configure(
    org_id=ims_org_id,
    tech_id=tech_account_id,
    secret=client_secret,
    path_to_key=private_key_path,
    client_id=client_id,
    environment=environment,
    sandbox=sandbox_name,
)

# COMMAND ----------

import urllib.parse
from pyspark.sql import functions as F
from pyspark.sql import types as T
import pandas as pd
import time
from typing import List, Dict, Set, Tuple


def get_ui_link(tenant_id, resource_type, resource_id):
    if environment == "prod":
        prefix = f"https://experience.adobe.com"
    else:
        prefix = f"https://experience-{environment}.adobe.com"
    return f"{prefix}/#/@{tenant_id}/sname:{sandbox_name}/platform/{resource_type}/{resource_id}"


def display_link(link, text=None):
    if text is None:
        text = link
    html = f"""<a href="{link}">{text}"""
    displayHTML(html)


def get_fieldgroup_by_title(schema_conn, title):
    """Return the field group with the given title, otherwise None if not found."""
    fieldgroups = schema_conn.getFieldGroups()
    for fieldgroup in fieldgroups:
        if fieldgroup['title'] == title:
            return fieldgroup
    return None


def get_or_create_fieldgroup(schema_conn, fieldgroup_spec):
    title = fieldgroup_spec['title']
    existing_fieldgroup = get_fieldgroup_by_title(schema_conn, title)
    if existing_fieldgroup is None:
        fieldgroup_res = schema_conn.createFieldGroup(fieldgroup_spec)
        return fieldgroup_res
    else:
        return existing_fieldgroup


def get_descriptor_by_schema(schema_conn, schema_id):
    descriptors = schema_conn.getDescriptors()
    for descriptor in descriptors:
        if descriptor["xdm:sourceSchema"] == schema_id:
            return descriptor
    return None


def get_or_create_descriptor(schema_conn, descriptor_obj):
    schema_id = descriptor_obj["xdm:sourceSchema"]
    descriptor = get_descriptor_by_schema(schema_conn, schema_id)
    if descriptor is None:
        descriptor = schema_conn.createDescriptor(
            descriptorObj={
                "@type": "xdm:descriptorIdentity",
                "xdm:sourceSchema": schema_id,
                "xdm:sourceVersion": 1,
                "xdm:sourceProperty": f"/_{tenant_id}/userid",
                "xdm:namespace": identity_type,
                "xdm:property": "xdm:id",
                "xdm:isPrimary": True,
            }
        )
        return descriptor
    else:
        return descriptor


def get_schema_by_title(schema_conn, title):
    schemas = schema_conn.getSchemas()
    for schema in schemas:
        if schema["title"] == title:
            return schema
    return None


def get_or_create_experience_event_schema(schema_conn, title, mixinIds, description):
    schema = get_schema_by_title(schema_conn, title)
    if schema is None:
        schema_res = schema_conn.createExperienceEventSchema(
            name=title,
            mixinIds=mixinIds,
            description=description,
        )
        return schema_res
    else:
        return schema


def get_or_create_profile_schema(schema_conn, title, mixinIds, description):
    schema = get_schema_by_title(schema_conn, title)
    if schema is None:
        schema_res = schema_conn.createProfileSchema(
            name=title,
            mixinIds=mixinIds,
            description=description,
        )
        return schema_res
    else:
        return schema


def get_dataset_ids_by_name(cat_conn, name):
    return [k for k, v in cat_conn.getDataSets().items() if v["name"] == name]

# COMMAND ----------

from adlfs import AzureBlobFileSystem
from fsspec import AbstractFileSystem

def get_export_time(fs: AbstractFileSystem, container_name: str, base_path: str, dataset_id: str):
    featurized_data_base_path = f"{container_name}/{base_path}/{dataset_id}"
    featurized_data_export_paths = fs.ls(featurized_data_base_path)

    if len(featurized_data_export_paths) == 0:
        raise Exception(f"Found no exports for featurized data from dataset ID {dataset_id} under path {featurized_data_base_path}")
    elif len(featurized_data_export_paths) > 1:
        print(f"Found {len(featurized_data_export_paths)} exports from dataset dataset ID {dataset_id} under path {featurized_data_base_path}, using most recent one")

    featurized_data_export_path = featurized_data_export_paths[-1]
    featurized_data_export_time = featurized_data_export_path.strip().split("/")[-1].split("=")[-1]
    return featurized_data_export_time


# COMMAND ----------

catalog_name = unique_id
database_name = "aep_cloud_ml_ecosystem"

spark.sql(f"create catalog if not exists {catalog_name}")
spark.sql(f"use catalog {catalog_name}")
spark.sql(f"create database if not exists {database_name}")
spark.sql(f"use schema {database_name}")

print(f"catalog_name: {catalog_name}")
print(f"database_name: {database_name}")

# COMMAND ----------

import mlflow

# We'll be using Unity Catalog as our Model Registry
mlflow.set_registry_uri("databricks-uc")

# Let's set up our experiment and registered model name.
experiment_name = f"/Users/{current_user}/aep-byoml/main-experiment"
base_model_name = "cmle_propensity"
model_name = f"{catalog_name}.{database_name}.{base_model_name}"
mlflow.set_experiment(experiment_name)

print(f"experiment_name: {experiment_name}")
print(f"model_name: {model_name}")

# COMMAND ----------

# MAGIC %config InlineBackend.figure_format='retina'

# COMMAND ----------

print("Configuration loaded")
