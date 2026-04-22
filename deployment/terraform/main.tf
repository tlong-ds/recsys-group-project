locals {
  name = "${var.project_name}-${var.environment}"
  tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )

  # Construct OIDC ARN manually to avoid iam:ListOpenIDConnectProviders permission issues
  oidc_issuer       = data.aws_eks_cluster.this.identity[0].oidc[0].issuer
  oidc_provider_url = replace(local.oidc_issuer, "https://", "")
  oidc_provider_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.oidc_provider_url}"
}

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

################################################################################
# IAM Roles for Service Accounts (IRSA)
################################################################################

module "alb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name = "${local.name}-alb-controller"

  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = local.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = local.tags
}

module "external_dns_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name                  = "${local.name}-external-dns"
  attach_external_dns_policy = true

  oidc_providers = {
    main = {
      provider_arn               = local.oidc_provider_arn
      namespace_service_accounts = ["kube-system:external-dns"]
    }
  }

  tags = local.tags
}

################################################################################
# Helm Releases
################################################################################

resource "helm_release" "aws_load_balancer_controller" {
  name             = "aws-load-balancer-controller"
  repository       = "https://aws.github.io/eks-charts"
  chart            = "aws-load-balancer-controller"
  namespace        = "kube-system"
  create_namespace = false
  version          = "1.11.0"

  set {
    name  = "clusterName"
    value = var.eks_cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "true"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.alb_controller_irsa.iam_role_arn
  }

  set {
    name  = "vpcId"
    value = "vpc-0611a00fc16788168"
  }

  set {
    name  = "region"
    value = var.aws_region
  }
}

################################################################################
# GitHub Actions OIDC & Deployment Role
################################################################################

module "github_oidc" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-github-oidc-provider"
  version = "~> 5.39"
}

module "github_actions_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-github-oidc-role"
  version = "~> 5.39"

  name = "${local.name}-github-actions-deployer"

  subjects = ["${var.github_repo}:*"]

  policies = {
    EKS_Admin  = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
    S3_Backend = aws_iam_policy.github_actions_s3_backend.arn
  }

  tags = local.tags
}

resource "aws_iam_policy" "github_actions_s3_backend" {
  name        = "${local.name}-github-actions-s3-backend"
  description = "Allow GitHub Actions to access Terraform state bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Effect   = "Allow"
        Resource = [
          "arn:aws:s3:::recsys-data-storage-067518243363-ap-southeast-1-an",
          "arn:aws:s3:::recsys-data-storage-067518243363-ap-southeast-1-an/*"
        ]
      }
    ]
  })
}
