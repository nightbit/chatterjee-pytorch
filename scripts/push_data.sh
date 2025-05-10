#!/usr/bin/env bash
# Push any new or updated files from data/ up to S3
aws --profile xin-s3 s3 sync data/ s3://xin-thesis-assets-eu/data/