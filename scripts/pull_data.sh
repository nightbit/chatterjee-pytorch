#!/usr/bin/env bash
# Fetch raw data from S3 into your local data/ folder
aws --profile xin-s3 s3 sync s3://xin-thesis-assets-eu/data/ data/