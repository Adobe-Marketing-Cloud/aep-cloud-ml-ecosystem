# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Cluster and Jobs

# COMMAND ----------

# MAGIC %md
# MAGIC Here we're using a few helper libraries for setting up our infrastructure.
# MAGIC For your workflow, you will likely find it more convenient to explore one of the following:
# MAGIC
# MAGIC 1. Databricks CLI
# MAGIC 2. Databricks REST API
# MAGIC 3. Databricks Terraform Provider
# MAGIC
# MAGIC The helpers we're using here are simple wrappers around the Databricks REST API.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

import re
import os

from solacc.companion import NotebookSolutionCompanion

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
username = current_user[:current_user.rfind('@')]
unique_id = s = re.sub("[^0-9a-zA-Z]+", "_", username)

# COMMAND ----------

task_notebook_names = [name for name in os.listdir() if re.match(r"Week\d+Notebook", name)]
task_notebook_names.sort()

job_name = f"{unique_id}-aep-ml-accelerator"
cluster_key = f"{unique_id}-aep-ml"

job_cluster_config = {
    "job_cluster_key": cluster_key,
    "new_cluster": {
        "spark_version": "13.3.x-cpu-ml-scala2.12",
        "spark_conf": {
            "spark.databricks.delta.formatCheck.enabled": "false"
        },
        "num_workers": 4,
        "spark_env_vars": {
            "ADOBE_HOME": f"/dbfs/home/{current_user}/.adobe"
        },
        "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
        "custom_tags": {
            "usage": "solacc_testing",
            "group": "CME",
            "accelerator": "media-mix-modeling"
        },
    },
}

pypi_packages = [
    "PyGreSQL==5.2.5",
    "adlfs==2023.9.0",
    "fsspec==2023.9.0",
    "s3fs==2023.9.0",
    "aepp==0.3.1.post5",
    "mmh3==4.0.1",
    "rstr==3.2.1",
    "faker==19.6.2",
    "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.3.0-py3-none-any.whl"
]

job_json = {
    "name": job_name,
    "timeout_seconds": 28800,
    "max_concurrent_runs": 1,
    "tags": {
        "usage": "aep-ml",
        "group": "CME"
    },
    "tasks": [
        {
            "job_cluster_key": cluster_key,
            "notebook_task": {
                "notebook_path": notebook_name
            },
            "task_key": f"{notebook_name}",
            "libraries": [
                {
                    "pypi": {
                        "package": pypi_package_string
                    }
                }
                for pypi_package_string in pypi_packages
            ]
        }
        for i, notebook_name in enumerate(task_notebook_names)
    ],
    "job_clusters": [
        job_cluster_config
    ]
}

# Wire up sequential dependencies among the notebooks
for i in range(1, len(job_json["tasks"])):
    job_json["tasks"][i]["depends_on"] = [{"task_key": job_json["tasks"][i - 1]["task_key"]}]

import pprint as pp
pp.pprint(job_json)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the job and cluster has been specified, we create them along with
# MAGIC a small interactive cluster we can use for the rest of our notebooks. If you
# MAGIC don't want to use these, or need to go through your local platform administrator
# MAGIC you can refer to the job specs to know what resources to request.

# COMMAND ----------

companion = NotebookSolutionCompanion()
job_params = companion.customize_job_json(
    job_json, job_name, companion.solacc_path, companion.cloud)
job_id = companion.create_or_update_job_by_name(job_params)
job_cluster_params = job_params["job_clusters"][0]
interactive_cluster_params = companion.convert_job_cluster_to_cluster(job_cluster_params)
interactive_cluster_params["autotermination_minutes"] = 120
cluster_id = companion.create_or_update_cluster_by_name(interactive_cluster_params)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Copy Media Files
# MAGIC
# MAGIC Databricks Notebooks can display images and refer to other media elements, but only 
# MAGIC for images either available publicly online or within the FileStore directory in 
# MAGIC DBFS (requires workspace access to view from there). Until these repositories are 
# MAGIC available more broadly, we copy all the media files we'll see throughout these notebooks
# MAGIC into the DBFS FileStore directory under a path based on a UUID so we don't run into
# MAGIC any naming conflicts with existing files and directories.

# COMMAND ----------

import os

# Copy image files (needed until we have public image URL's)

def ensure_media_copied():
    # This can likely go away after the repository is release publicly, as we can
    # adjust the image URL's to reference the public raw URL's in GitHub.
    
    # compose the target directory
    filestore_fuse = "FileStore"
    accelerator_uuid = "7cf4bf44-5482-4426-a3b3-842be2f737b1"
    static_files_dir = os.path.join(filestore_fuse, f"static/{accelerator_uuid}")

    # check if target directory exists
    if not os.path.exists(os.path.join("/dbfs", static_files_dir)):
        # compose the source directory
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        media_dir = os.path.join(parent_dir, "media")

        print("static files directory not found in FUSE mount's FileStore")

        # create target directory
        print(f"creating {static_files_dir}")
        dbutils.fs.mkdirs(static_files_dir)

        # copy media directory
        print(f"copying media files")
        source_path = os.path.join("file:/", media_dir.strip("/"))
        dest_path = os.path.join(static_files_dir, "media")
        dbutils.fs.cp(source_path, dest_path, recurse=True)
    else:
        print("static files directory already found")

ensure_media_copied()
