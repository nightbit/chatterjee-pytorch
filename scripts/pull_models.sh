#!/usr/bin/env bash
# Fetch model checkpoints from S3 into your local models/ folder
aws --profile xin-s3 s3 sync s3://xin-thesis-assets-eu/models/ models/