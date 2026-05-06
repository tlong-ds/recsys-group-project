terraform {
  required_version = ">= 1.6.0"

  backend "s3" {
    bucket         = "recsys-data-storage-067518243363-ap-southeast-1-an"
    key            = "terraform/state.tfstate"
    region         = "ap-southeast-1"
    dynamodb_table = "terraform-state-lock"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.13"
    }
  }
}

