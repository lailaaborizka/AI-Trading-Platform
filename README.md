<!-- Much thanks to https://github.com/othneildrew/Best-README-Template for the template -->
<!-- And to https://github.com/alexandresanlim/Badges4-README.md-Profile for the badges -->

# AI Trading Platform

###### This is an **AI Trading Platform** project and is a team work for the graduation project in the bachelor of engineering aimed at developing an advanced system that utilizes machine learning and sentiment analysis to improve trading strategies for the stock market.

## Overview

The **AI Trading Platform** project aims to create an advanced trading system that leverages artificial intelligence (AI) techniques to enhance trading strategies and decision-making in financial markets. This platform integrates **Machine Learning** (ML) and **Deep Learning** (DL) methodologies to predict stock prices, optimize portfolio management, and improve overall trading performance.

This README outlines the key features of the project, installation instructions, usage, and provides a comprehensive explanation of sentiment analysis, technical indicators, and machine learning models used.

<details>
  <summary><b>Table of Contents</b></summary>
	<ol>
		<li><a href="#foreword">Foreword</a></li>
		<li><a href="#features">Features</a></li>
		<li><a href="#installation">Installation</a></li>
	</ol>
</details>

## Foreword

The **AI Trading Platform** is designed to enhance stock market trading decisions using artificial intelligence techniques. By leveraging **Machine Learning** and **Sentiment Analysis**, it analyzes both quantitative data (such as stock prices) and qualitative data (such as social media sentiment) to optimize trading strategies and predict future stock trends.

In the current stock market, being able to predict stock movements and understand market sentiment can significantly improve trading decisions. This project uses historical stock data, real-time sentiment from social media platforms like Twitter, and technical analysis to build predictive models for stock trading.

###### Built With

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white) ![Scipy](https://img.shields.io/badge/scipy-FF6633?style=for-the-badge&logo=spicy&logoColor=white)  
![SCIKIT-IMAGE](https://img.shields.io/badge/scikit--image-5b80b1?style=for-the-badge&logo=python&logoColor=white) ![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

## Features

### 1. **Sentiment Analysis**

- **Purpose**: To gauge market sentiment by analyzing public opinion from social media platforms like Twitter and StockTwits.
- **Methodology**:
  - **Data Collection**: Tweets and financial data are gathered to assess the public sentiment towards specific stocks.
  - **Sentiment Scoring**: VADER (Valence Aware Dictionary and sEntiment Reasoner) is used to analyze the sentiment of each tweet, categorizing sentiment as positive, negative, or neutral.
  - **Integration with Trading Models**: The sentiment scores are integrated into trading models to enhance trading strategies, reacting to shifts in public sentiment.

### 2. **Technical Indicators**

- **Purpose**: To use historical stock data to inform trading decisions.
- **Types of Indicators**:
  - **Price Indicators**: Closing prices, moving averages to identify trends.
  - **Volume Indicators**: Reflects stock trading volume to indicate strength of price movements.
  - **Technical Indicators**: Such as Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and others.

### 3. **Machine Learning Models**

- **Models Used**: The system utilizes a variety of machine learning models for stock prediction and sentiment classification:
  - **Supervised Models**: Support Vector Machine (SVM), Random Forest, Logistic Regression, Naïve Bayes, and XGBoost.
  - **Deep Learning Models**: Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN).

## Installation

### Prerequisites

Before you can use the **AI Trading Platform**, make sure to install the following dependencies:

- **Python 3.x**
- **Required Libraries**

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/AI-Trading-Platform.git
cd AI-Trading-Platform
pip install -r requirements.txt
```
