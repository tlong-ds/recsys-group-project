output "cluster_name" {
  description = "EKS cluster name."
  value       = data.aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = data.aws_eks_cluster.this.endpoint
}

output "external_dns_role_arn" {
  description = "IRSA role ARN for ExternalDNS."
  value       = module.external_dns_irsa.iam_role_arn
}

output "github_actions_role_arn" {
  description = "ARN of the IAM Role for GitHub Actions deployment."
  value       = module.github_actions_role.arn
}

output "model_cache_efs_file_system_id" {
  description = "EFS file system ID used for shared model cache."
  value       = var.enable_efs_model_cache ? aws_efs_file_system.model_cache[0].id : null
}

output "model_cache_storage_class_name" {
  description = "Kubernetes StorageClass name for shared model cache."
  value       = var.enable_efs_model_cache ? kubernetes_storage_class_v1.recsys_efs[0].metadata[0].name : null
}
