# Stock-Screener
S&amp;P 500 stock screener utilizing Yahoo Finance and simple momentum/machine learning algorithms. Developed as an exercise in essential python modules and fundamental data science and machine learning methods.

DO NOT TRADE ON THESE RESULTS! 

## How to Run

```bash
pip install -r requirements.txt
python random_forest_screener.py
python momentum_screener.py
```
There should be an excel file in the relevant directory entitled f'Top {N} Stocks by {METHOD}'

## Future directions for improvement:

### As of 2/16/22...
  - REFACTOR!!!
  - Implement threading for momentum screener
  - Resolve backtesting issues to better evaluate profitibility
  - Experiment with classification/other machine learning methods 
    - Am I brave enough to mess with neural networks?
  - Replace yfinance with a faster and more reliable bs4 parser.
