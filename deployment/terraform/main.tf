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
  oidc_provider_arn = aws_iam_openid_connect_provider.eks.arn
  efs_mount_subnet_ids = (
    length(var.private_subnet_ids) > 0
    ? toset(var.private_subnet_ids)
    : toset(data.aws_eks_cluster.this.vpc_config[0].subnet_ids)
  )
}

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

data "tls_certificate" "eks" {
  url = data.aws_eks_cluster.this.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = data.aws_eks_cluster.this.identity[0].oidc[0].issuer

  tags = local.tags
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

module "efs_csi_irsa" {
  count   = var.enable_efs_model_cache ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name             = "${local.name}-efs-csi-controller"
  attach_efs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = local.oidc_provider_arn
      namespace_service_accounts = ["kube-system:efs-csi-controller-sa"]
    }
  }

  tags = local.tags
}

################################################################################
# EFS Shared Model Cache (RWX)
################################################################################

resource "aws_eks_addon" "efs_csi" {
  count = var.enable_efs_model_cache ? 1 : 0

  cluster_name                = var.eks_cluster_name
  addon_name                  = "aws-efs-csi-driver"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  service_account_role_arn    = module.efs_csi_irsa[0].iam_role_arn

  depends_on = [module.efs_csi_irsa]
}

resource "aws_security_group" "efs_nfs" {
  count = var.enable_efs_model_cache ? 1 : 0

  name_prefix = "${local.name}-efs-nfs-"
  description = "Allow EKS nodes to access EFS over NFS"
  vpc_id      = data.aws_eks_cluster.this.vpc_config[0].vpc_id

  ingress {
    description     = "NFS from EKS cluster security group"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [data.aws_eks_cluster.this.vpc_config[0].cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, { Name = "${local.name}-efs-nfs" })
}

resource "aws_efs_file_system" "model_cache" {
  count = var.enable_efs_model_cache ? 1 : 0

  creation_token   = "${local.name}-model-cache"
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy {
    transition_to_ia = var.efs_transition_to_ia
  }

  tags = merge(local.tags, { Name = "${local.name}-model-cache" })
}

resource "aws_efs_mount_target" "model_cache" {
  for_each = var.enable_efs_model_cache ? local.efs_mount_subnet_ids : toset([])

  file_system_id  = aws_efs_file_system.model_cache[0].id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs_nfs[0].id]
}

resource "kubernetes_storage_class_v1" "recsys_efs" {
  count = var.enable_efs_model_cache ? 1 : 0

  metadata {
    name = var.efs_storage_class_name
    labels = {
      app = "recsys-api"
    }
  }

  storage_provisioner    = "efs.csi.aws.com"
  reclaim_policy         = "Retain"
  volume_binding_mode    = "Immediate"
  allow_volume_expansion = true

  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.model_cache[0].id
    directoryPerms   = "770"
    basePath         = "/dynamic_provisioning"
  }

  mount_options = ["tls"]

  depends_on = [
    aws_eks_addon.efs_csi,
    aws_efs_mount_target.model_cache,
  ]
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
    value = data.aws_eks_cluster.this.vpc_config[0].vpc_id
  }

  set {
    name  = "region"
    value = var.aws_region
  }
}

################################################################################
# Metrics Server (Managed by EKS Add-on)
################################################################################
# Metrics Server is already installed as an EKS managed add-on.

################################################################################
# EKS Fargate Profile
################################################################################
# Fargate Profile skipped as the current VPC only contains public subnets.
# EKS Auto Mode can handle serverless workloads if private subnets are added.

################################################################################
# Karpenter (Managed by EKS Auto Mode)
################################################################################
# EKS Auto Mode is enabled on this cluster, which provides managed Karpenter 
# functionality. Manual Karpenter installation is omitted to avoid conflicts.

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
    EKS_Admin  = aws_iam_policy.github_actions_eks_admin.arn
    S3_Backend = aws_iam_policy.github_actions_s3_backend.arn
  }

  tags = local.tags
}

resource "aws_iam_policy" "github_actions_eks_admin" {
  name        = "${local.name}-github-actions-eks-admin"
  description = "Allow GitHub Actions to manage EKS cluster"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters",
          "eks:AccessKubernetesApi"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_eks_access_entry" "github_actions" {
  cluster_name  = var.eks_cluster_name
  principal_arn = module.github_actions_role.arn
  type          = "STANDARD"
}

resource "aws_eks_access_policy_association" "github_actions_admin" {
  cluster_name  = var.eks_cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  principal_arn = module.github_actions_role.arn

  access_scope {
    type = "cluster"
  }
}

resource "aws_iam_policy" "github_actions_s3_backend" {
  name        = "${local.name}-github-actions-s3-backend"
  description = "Allow GitHub Actions to access Terraform state bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Effect = "Allow"
        Resource = [
          "arn:aws:s3:::${var.terraform_state_bucket}",
          "arn:aws:s3:::${var.terraform_state_bucket}/*"
        ]
      }
    ]
  })
}
