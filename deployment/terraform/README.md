# Terraform EKS bootstrap

This directory provisions:
- VPC (public/private subnets + NAT)
- EKS cluster + managed node group
- IRSA roles for ALB controller, ExternalDNS, and cert-manager
- Helm-based installation of:
  - `aws-load-balancer-controller`
  - `external-dns`
  - `cert-manager`
  - `metrics-server`

## Prerequisites

1. Terraform `>= 1.6`
2. AWS credentials with permissions for VPC/EKS/IAM/Route53/ELB
3. Existing Route53 hosted zone ID for your domain

## Usage

```bash
cd deployment/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars with your values
terraform init
terraform plan
terraform apply
```

## Notes

- This stack intentionally keeps MLflow out of EKS; serving should use DagsHub MLflow.
- The Kubernetes app manifests under `deployment/kubernetes/` still need EKS-specific ingress/service-account/secret overlays.

