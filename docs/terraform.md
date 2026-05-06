# Terraform Infrastructure Guide

This document provides a high-level overview and technical reference for managing the AWS infrastructure of the RecSys project using Terraform and EKS Auto Mode.

## Architecture Overview

The infrastructure for the RecSys project is built on **Amazon EKS with Auto Mode** enabled. This significantly simplifies cluster management by automating the provisioning of compute, networking, and core add-ons.

### Core Components
- **Identity (OIDC & IAM)**: Sets up an OIDC provider for the EKS cluster to enable IAM Roles for Service Accounts (IRSA). This is used by ExternalDNS and the AWS Load Balancer Controller.
- **Compute (EKS Auto Mode & Karpenter)**: 
    - **EKS Auto Mode**: Automatically manages the underlying compute nodes, removing the need for manual Node Group management.
    - **Spot Orchestration**: Managed through Karpenter (integrated into Auto Mode). A custom `spot-compute` NodePool is deployed to prefer EC2 Spot instances for cost efficiency.
- **Networking**: Discovers the existing VPC and subnets dynamically. Note that the current environment uses **public subnets**; therefore, AWS Fargate is not supported (as it requires private subnets).
- **Managed Add-ons**:
    - **AWS Load Balancer Controller**: Manages ALBs for Ingress resources (managed via Helm/Terraform).
    - **Metrics Server**: Enables Horizontal Pod Autoscaling (HPA), managed as an **EKS Add-on**.

---

## Prerequisites

1. **AWS CLI**: Authenticated with the `recsys-iam-user`.
2. **Terraform**: Version `1.6.0` or higher.
3. **S3 Backend**: The state is stored in `recsys-data-storage-067518243363-ap-southeast-1-an`.

---

## Configuration Management

### `terraform.tfvars` (Primary)
This file is the single source of truth for infrastructure variables. It overrides defaults and environment variables.
- `eks_cluster_name`: The name of the target cluster (`recsys-cluster`).
- `aws_region`: The deployment region (`ap-southeast-1`).
- `private_subnet_ids`: Used by various modules for resource placement.

---

## Essential Commands

### 1. Initialization & State Management
```bash
cd deployment/terraform
terraform init
terraform plan
terraform apply
```

### 2. Connecting to the Cluster
```bash
aws eks update-kubeconfig --name recsys-cluster --region ap-southeast-1
```

### 3. Managing Spot Instances (NodePools)
Since Karpenter is managed by EKS Auto Mode, we define compute preferences via NodePool CRDs:
```bash
# Apply the Spot NodePool
kubectl apply -f deployment/kubernetes/karpenter-spot-nodepool.yaml

# Check NodePool status
kubectl get nodepools.karpenter.sh
```

---

## Cost Optimization Strategy

- **Spot Instances**: The `spot-compute` NodePool is configured to use Spot instances for the majority of workloads, saving up to 90% on compute costs.
- **ALB Consolidation**: Multiple Ingress resources (API, Grafana, etc.) are consolidated into a single Application Load Balancer using the annotation:
  `alb.ingress.kubernetes.io/group.name: "recsys"`
- **HPA (Horizontal Pod Autoscaling)**: Enabled via the EKS Metrics Server add-on, ensuring pods scale down during low-traffic periods to minimize resource consumption.

---

## Troubleshooting

### EKS Access Entry (Permission Denied)
If your IAM identity cannot access the cluster via `kubectl`, grant admin access:
```bash
aws eks create-access-entry --cluster-name recsys-cluster --principal-arn <YOUR_USER_ARN>
aws eks associate-access-policy --cluster-name recsys-cluster --principal-arn <YOUR_USER_ARN> \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy --access-scope type=cluster
```

### Subnet Constraints
If a resource (like Fargate) fails with a "not a private subnet" error, verify the subnet's routing. Fargate and some VPC Lattice features require subnets with no direct route to an Internet Gateway.
