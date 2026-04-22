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
  default     = "unique-pop-otter"
}

variable "github_repo" {
  description = "GitHub repository (repo:org/repo) for OIDC trust."
  type        = string
  default     = "repo:tlong-ds/recsys-group-project"
}

variable "tags" {
  description = "Extra tags to apply to created resources."
  type        = map(string)
  default     = {}
}
