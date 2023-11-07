# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
#
# This CDK project launches infrastructure for Adobe CMLE notebook testing.  
# Details about what resources are provisioned can be found in the AWS-specific  
# README within the Adobe CMLE repository.
#
from aws_cdk import (
    CfnOutput,
    CfnTag,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_secretsmanager as secretsmanager
)
from constructs import Construct

class InfraStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VPC and S3 endpoint
        vpc = ec2.Vpc(self, 'sm-aep-vpc',
            ip_addresses=ec2.IpAddresses.cidr('10.10.0.0/16'),
            gateway_endpoints={
                "S3": ec2.GatewayVpcEndpointOptions(
                    service=ec2.GatewayVpcEndpointAwsService.S3
                )
            },
            max_azs=2,
            nat_gateways=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    cidr_mask=24,
                    name='sm-aep-public',
                    subnet_type=ec2.SubnetType.PUBLIC
                ), 
                ec2.SubnetConfiguration(
                    cidr_mask=24,
                    name='sm-aep-private',
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                )
            ],
            vpc_name='sm-aep-vpc'
        )

        # Security group for Studio communication
        studio_security_group = ec2.SecurityGroup(self, 'sm-aep-security-group',
            vpc=vpc,
            description='Studio notebook security group',
            security_group_name='sm-aep-security-group',
            allow_all_outbound=True
        )
        studio_security_group.add_ingress_rule(ec2.Peer.ipv4(vpc.vpc_cidr_block), ec2.Port.tcp(2049), description='NFS between domain and EFS mounts')
        studio_security_group.add_ingress_rule(studio_security_group, ec2.Port.tcp_range(8192, 65535), description='Connectivity between Jupyter Server application and Kernel Gateway applications')

        # IAM role for SageMaker Studio
        studio_execution_role = iam.Role(self, 'sm-aep-studio-exec-role',
            assumed_by=iam.ServicePrincipal('sagemaker.amazonaws.com'),
            description='SageMaker Studio execution role'                            
        )
        studio_execution_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess'))
        studio_execution_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AWSCloudFormationReadOnlyAccess'))

        # AEP Data Flow interaction requires an AWS key/secret to authorize data movement to/from S3
        # Create IAM user specifically for this purpose 
        data_flow_user = iam.User(self, "data-flow-user")
        data_flow_user_group = iam.Group(self, "data-flow-user-group")
        data_flow_user_group.add_user(data_flow_user)
        access_key = iam.AccessKey(self, "data-flow-user-access-key", user=data_flow_user)
        secret_key = secretsmanager.Secret(self, "data-flow-user-secret-key",
            secret_name="data-flow-user-secret-key",
            secret_string_value=access_key.secret_access_key
        )
        secret_key.grant_read(studio_execution_role)

        # S3 storage to host AEP datasets
        bucket = s3.Bucket(self, 'sm-aep-data',
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY
        )
        bucket.add_cors_rule(
            allowed_methods=[
                s3.HttpMethods.GET, 
                s3.HttpMethods.PUT, 
                s3.HttpMethods.POST, 
                s3.HttpMethods.DELETE
            ],
            allowed_origins=['*'],
            allowed_headers=['*']
        )
        bucket.grant_read_write(studio_execution_role)
        bucket.grant_read_write(data_flow_user_group)

        # SageMaker Studio domain
        studio_domain = sagemaker.CfnDomain(self, 'sm-aep-studio-domain',
            auth_mode='IAM',
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=studio_execution_role.role_arn,
                security_groups=[studio_security_group.security_group_id]
            ),
            domain_name='sm-aep-domain',
            subnet_ids=[
                vpc.private_subnets[0].subnet_id,
                vpc.private_subnets[1].subnet_id
            ],
            vpc_id=vpc.vpc_id,
            app_network_access_type='VpcOnly',
            tags=[
                CfnTag(
                    key='Name', 
                    value='sm-aep-domain'
                )
            ]
        )

        # SageMaker Studio domain user profile
        studio_domain_user_profile = sagemaker.CfnUserProfile(self, 'sm-aep-studio-domain-user-profile',
            domain_id=studio_domain.attr_domain_id,
            user_profile_name='sm-aep-domain-user'
        )

        jupyter_app = sagemaker.CfnApp(self, 'jupyter-app',
            app_name='default',
            app_type='JupyterServer',
            domain_id=studio_domain.attr_domain_id,
            user_profile_name=studio_domain_user_profile.user_profile_name
        )

        kernel_app = sagemaker.CfnApp(self, "kernel-app",
            app_name='kernel',
            app_type='KernelGateway',
            domain_id=studio_domain.attr_domain_id,
            user_profile_name=studio_domain_user_profile.user_profile_name,
            resource_spec=sagemaker.CfnApp.ResourceSpecProperty(
                instance_type='ml.m5.large',
                sage_maker_image_arn='arn:aws:sagemaker:us-west-2:236514542706:image/sagemaker-data-science-310-v1'
            )
        )

        jupyter_app.add_dependency(studio_domain_user_profile)
        kernel_app.add_dependency(studio_domain_user_profile)

        CfnOutput(self, "Stack ID", value=self.stack_id)
        CfnOutput(self, "VPC ID", value=vpc.vpc_id)
        CfnOutput(self, "Studio Domain ID", value=studio_domain.attr_domain_id)
        CfnOutput(self, "Studio Domain URL", value=studio_domain.attr_url)
        CfnOutput(self, "Studio Domain User Profile", value=studio_domain_user_profile.user_profile_name)        
        CfnOutput(self, "S3 Data Bucket", value=bucket.bucket_name)
        CfnOutput(self, "Data Flow IAM User", value=data_flow_user.user_name)
        CfnOutput(self, "Data Flow User Access Key", value=access_key.access_key_id)
        CfnOutput(self, "Data Flow User Secret Key", value=secret_key.secret_name)
        CfnOutput(self, "EFS ID", value=studio_domain.attr_home_efs_file_system_id)
            

