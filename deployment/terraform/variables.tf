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
  description = "AWS region to deploy EKS."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the EKS VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones used by private/public subnets."
  type        = list(string)
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDRs for worker nodes."
  type        = list(string)
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDRs for ALB ingress."
  type        = list(string)
}

variable "cluster_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.30"
}

variable "node_instance_types" {
  description = "Instance types for the default managed node group."
  type        = list(string)
  default     = ["m6i.large"]
}

variable "node_min_size" {
  description = "Managed node group minimum size."
  type        = number
  default     = 2
}

variable "node_desired_size" {
  description = "Managed node group desired size."
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Managed node group maximum size."
  type        = number
  default     = 6
}

variable "route53_zone_id" {
  description = "Hosted zone ID used by ExternalDNS and cert-manager."
  type        = string
}

variable "domain_filters" {
  description = "Allowed DNS zones for ExternalDNS updates."
  type        = list(string)
}

variable "tags" {
  description = "Extra tags to apply to created resources."
  type        = map(string)
  default     = {}
}

