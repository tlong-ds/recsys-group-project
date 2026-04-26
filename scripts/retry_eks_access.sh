#!/bin/bash
set -e

CLUSTER_NAME="unique-pop-otter"
PRINCIPAL_ARN="arn:aws:iam::067518243363:user/recsys-iam-user"
POLICY_ARN="arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

echo "Checking current identity..."
aws sts get-caller-identity

echo "--------------------------------------------------------"
echo "Step 1: Checking if access entry exists..."
if aws eks describe-access-entry --cluster-name "$CLUSTER_NAME" --principal-arn "$PRINCIPAL_ARN" >/dev/null 2>&1; then
    echo "✅ Access entry already exists for $PRINCIPAL_ARN"
else
    echo "Creating access entry..."
    aws eks create-access-entry \
        --cluster-name "$CLUSTER_NAME" \
        --principal-arn "$PRINCIPAL_ARN" \
        --type STANDARD
    echo "✅ Access entry created."
fi

echo "--------------------------------------------------------"
echo "Step 2: Associating Cluster Admin Policy..."
aws eks associate-access-policy \
    --cluster-name "$CLUSTER_NAME" \
    --principal-arn "$PRINCIPAL_ARN" \
    --policy-arn "$POLICY_ARN" \
    --access-scope type=cluster

echo "✅ Successfully associated $POLICY_ARN with $PRINCIPAL_ARN"
echo "--------------------------------------------------------"
echo "Setup complete! You can now run 'kubectl get nodes' as the recsys-iam-user to verify access."
