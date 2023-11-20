# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
#
# This script is used to clean up the AWS resources used to test Adobe CMLE
# notebooks. It retrieves the SageMaker domain user profile from CDK/CloudFormation
# outputs and deletes all user applications in preparation to destroy the CDK
# stack. It will also remove the EFS filesystem that hosts Studio user directories.
#
# NOTE: This script expects the config.ini file to be in the conf/ folder at the 
# root of the Adobe CMLE repository.
#
# Args:
# none
#

import boto3
import os
import logging
from rich.logging import RichHandler
from rich.traceback import install
from botocore.exceptions import ValidationError, ClientError
from botocore.config import Config
from configparser import ConfigParser
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[RichHandler()])
log = logging.getLogger()
install(show_locals=True)

# AWS clients
region = 'us-west-2' if os.getenv('REGION_NAME') == None else os.getenv('REGION_NAME')
client_config = Config(region_name=region)
cfn = boto3.client('cloudformation', config=client_config)
sagemaker = boto3.client('sagemaker', config=client_config)
efs = boto3.client('efs', config=client_config)
ec2 = boto3.client('ec2', config=client_config)

# Get stack id from config.ini
try:
    log.info(f'Parsing config file')
    config = ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "conf", "config.ini")
    config.read(config_path)
    log.info(f'CONFIG FILE SECTIONS: {config.sections()}')
    stack_id=config.get('AWS', 'cfn_stack_id')
    log.info(f'STACK ID: {stack_id}')
except Exception as e:
    log.error(e)
    raise ValueError('Could not find config.ini file or required values in the file.')

# Get SageMaker domain user
try:
    # Validate CloudFormation ID and get outputs
    log.info('Validating CloudFormation ID')
    response = cfn.describe_stacks(StackName=stack_id)
    log.info('Found stack')
    outputs = response['Stacks'][0]['Outputs']
    for output in outputs:
        if output['OutputKey'] == 'StudioDomainUserProfile':
            user_profile = output['OutputValue']
            log.info(f'SageMaker Studio domain user profile found: {user_profile}')
        elif output['OutputKey'] == 'EFSID':
            efs_id = output['OutputValue']
            log.info(f'SageMaker Studio EFS filesystem ID found: {efs_id}')
        elif output['OutputKey'] == 'VPCID':
            vpc_id = output['OutputValue']
            log.info(f'VPC ID found: {vpc_id}')
except ValidationError as e:
    log.error(f'Could not find stack from provided stack ID: {stack_id}')
    raise(e)
except ClientError as e:
    log.error(e)

# List SageMaker Studio domain user profile apps and delete them
try:
    log.info(f'Listing apps for user profile {user_profile}')
    apps = sagemaker.list_apps(
        UserProfileNameEquals=user_profile
    )
    log.info(f'Found apps: {apps}')
    log.info(f'Deleting apps...')
    domain_id = apps['Apps'][0]['DomainId']
    
    for app in apps['Apps']:
        log.info(app)
        log.info(f'Deleting {app["AppName"]}')
        if app['Status'] == 'Deleted':
            log.info(f'{app["AppName"]} already deleted, moving on...')
            continue
        response = sagemaker.delete_app(
            DomainId=app['DomainId'],
            UserProfileName=user_profile,
            AppType=app['AppType'],
            AppName=app['AppName']
        )

    log.info('Checking status of Studio apps clean up')
    for app in apps['Apps']:
        log.info(f'Checking {app["AppName"]}')
        response = sagemaker.describe_app(
            DomainId=app['DomainId'],
            UserProfileName=user_profile,
            AppType=app['AppType'],
            AppName=app['AppName']
        )
        while (response['Status'] != 'Deleted'):
            log.info('Waiting for deletion...')
            time.sleep(60)
            response = sagemaker.describe_app(
                DomainId=app['DomainId'],
                UserProfileName=user_profile,
                AppType=app['AppType'],
                AppName=app['AppName']
            )
        log.info(f'{app["AppName"]} deleted')
except ClientError as e:
    log.error(f'Problem listing or deleting apps for sm-aep-domain-user.')

# Delete EFS mount targets and filesystem
try: 
    log.info('Finding EFS mount targets')
    mnt_targets = efs.describe_mount_targets(
        FileSystemId=efs_id
    )
    for target in mnt_targets['MountTargets']:
        log.info(f'Found mount target: {target["MountTargetId"]}')
        log.info(f'Deleting mount target: {target["MountTargetId"]}')
        efs.delete_mount_target(
            MountTargetId=target["MountTargetId"]
        )
    log.info('Waiting 60s for mount targets to clean up')
    time.sleep(60)
    log.info('Mount targets deleted')
except ClientError as e:
    log.error(e)

try:
    log.info(f'Deleting EFS filesystem: {efs_id}')
    response = efs.delete_file_system(
        FileSystemId=efs_id
    )
    log.info('Filesystem deleted')    
except ClientError as e:
    log.error(e)