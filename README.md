# 🚚 Supply Chain Delay Predictor

> Predicting e-commerce delivery delays using machine learning on 100,000+ real orders.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This project builds an end-to-end ML pipeline to predict whether an order 
will be delivered late — enabling supply chain teams to proactively intervene 
before SLA breaches occur.

**Business problem:** Late deliveries damage customer satisfaction and incur 
SLA penalties. Early prediction allows logistics teams to prioritise 
at-risk shipments.

**Dataset:** Olist Brazilian E-Commerce — 100k+ real orders across 2016–2018.

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 96,000+ delivered orders |
| Overall delay rate | 8.1% |
| Final model | XGBoost + SMOTE |
| ROC-AUC Score | 0.80 |
| Delay Recall | 69% |
| Top delay predictor | Seller historical delay rate (36.2%) |

## Key Findings

- **Seller reliability** is the #1 predictor of delays at 36.2% importance
- **Order month** drives 19.1% of delay risk — strong seasonality effect
- **Promised lead time** accounts for 12.8% — overpromising causes failures
- Identified and fixed class imbalance using SMOTE — improved delay 
  detection from 5% to 69% recall

## Project Structure


