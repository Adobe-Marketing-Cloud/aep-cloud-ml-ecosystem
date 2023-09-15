# Databricks notebook source
import os

# TODO: Set up cluster if doesn't already exist
#       Ensure cluster has required libs installed and pinned


# TODO: Set up workflow(s) covering end to end run, etc...


# TODO: Set up initial config files templates


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
