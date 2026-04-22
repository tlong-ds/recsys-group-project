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
