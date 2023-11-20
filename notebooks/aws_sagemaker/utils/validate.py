# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
#
# This script reads AWS values from a config.ini used by Adobe CMLE notebooks and 
# validates the S3 bucket, S3 prefix, and credentials. It retrieves resource names
# from config.ini and CDK/CloudForamtion outputs; and tests interaction with the 
# S3 location.
#
# NOTE: This script expects the config.ini file to be in the conf/ folder at the 
# root of the Adobe CMLE repository.
#
# Args:
# CloudFormation stack ID
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
secrets_manager = boto3.client('secretsmanager', config=client_config)

# Get bucket name and prefix from config.ini
try:
    log.info(f'Parsing config file')
    config = ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "conf", "config.ini")
    config.read(config_path)
    log.info(f'CONFIG FILE SECTIONS: {config.sections()}')
    s3_bucket_name=config.get('AWS','s3_bucket_name')
    #s3_bucket_name='TESTINGANONEXISTENTBUCKET'
    #s3_bucket_name='mui-temp' # testing a bucket that is not authorized by creds
    s3_prefix=config.get('AWS','s3_prefix')
    stack_id=config.get('AWS', 'cfn_stack_id')
    log.info(f'BUCKET NAME: {s3_bucket_name}')
    log.info(f'PREFIX: {s3_prefix}')
    log.info(f'STACK ID: {stack_id}')
except Exception as e:
    log.error(e)
    raise ValueError('Could not find config.ini file or required values in the file.')

# Get credentials
try:
    # Validate CloudFormation ID and get outputs
    log.info('Validating CloudFormation ID')
    response = cfn.describe_stacks(StackName=stack_id)
    log.info('Found stack')
    outputs = response['Stacks'][0]['Outputs']
    for output in outputs:
        if output['OutputKey'] == 'DataFlowUserAccessKey':
            access_key = output['OutputValue']
            log.info(f'ACCESS KEY: {access_key}')        
        elif output['OutputKey'] == 'DataFlowUserSecretKey':
            secret_name = output['OutputValue']
            log.info(f'Found secret stored in Secrets Manager: {secret_name}')
except ValidationError as e:
    log.error(f'Could not find stack from provided stack ID: {stack_id}')
    raise(e)  
except ClientError as e:
    raise(e)

# Get secret access key from Secrets Manager
try:
    log.info('Retrieving secret access key from Secrets Manager') 
    response = secrets_manager.get_secret_value(SecretId=secret_name)
    secret_key = response['SecretString']
except ClientError as e:
    log.error('Could not retrieve secret access key from Secrets Manager')
    raise(e)

log.info(f'SECRET KEY: {secret_key}')

# Test S3 read/write with AWS creds for AEP data flow service
try:
    s3_client_config = Config(
        region_name=region
    )

    s3 = boto3.client('s3', 
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=s3_client_config
    )

    log.info('Beginning S3 test')
    test_data = {'test': 'test'}
    object_path = f'{s3_prefix}/DELETEME'
    log.info('Testing S3 write')

    response = s3.put_object(
        Bucket=s3_bucket_name, 
        Key=object_path, 
        Body=json.dumps(test_data)
    )
    log.info(f'S3 RESPONSE: {response}')
    log.info('Test object written successfully')

    log.info('Testing S3 read')
    response = s3.head_object(
        Bucket=s3_bucket_name,
        Key=object_path
    )
    log.info(f'S3 RESPONSE: {response}')
    log.info('Test object read successfully')
    
    log.info('Cleaning up test S3 object')
    response = s3.delete_object(
        Bucket=s3_bucket_name,
        Key=object_path
    )
    log.info(f'S3 RESPONSE: {response}')
    log.info('Test object deleted successfully')
    
    log.info('SUCCESS: S3 bucket, prefix, and credentials are valid. Ok to proceed with the notebooks.')
except ClientError as e:
    log.error('FAIL: S3 bucket, prefix, or credentials are not valid')
    raise(e)
