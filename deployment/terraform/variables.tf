variable "project_name" {
  description = "Project/application name used for tags and resource naming."
  type        = string
  default     = "recsys"
}

variable "environment" {
  description = "Environment name (for example: dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region where the existing EKS cluster is located."
  type        = string
}

variable "eks_cluster_name" {
  description = "The name of the existing EKS cluster."
  type        = string
}

variable "terraform_state_bucket" {
  description = "The name of the S3 bucket used for Terraform state."
  type        = string
}

variable "github_repo" {
  description = "GitHub repository (repo:org/repo) for OIDC trust."
  type        = string
}

variable "tags" {
  description = "Extra tags to apply to created resources."
  type        = map(string)
  default     = {}
}

variable "private_subnet_ids" {
  description = "A list of private subnet IDs for EKS resources (Fargate/Nodes)."
  type        = list(string)
  default     = []
}

variable "enable_efs_model_cache" {
  description = "Enable EFS-backed RWX storage for model cache sharing across pods."
  type        = bool
  default     = true
}

variable "efs_storage_class_name" {
  description = "Kubernetes storage class name for shared model cache PVCs."
  type        = string
  default     = "recsys-efs-sc"
}

variable "efs_transition_to_ia" {
  description = "Lifecycle transition for EFS files not accessed recently."
  type        = string
  default     = "AFTER_30_DAYS"
}
