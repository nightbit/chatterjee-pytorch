#!/usr/bin/env bash
# Push new model checkpoints from models/ up to S3
aws --profile xin-s3 s3 sync models/ s3://xin-thesis-assets-eu/models/