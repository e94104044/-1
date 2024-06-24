import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import tkinter as tk
from tkinter import messagebox, ttk

class PortfolioOptimizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('股票投資組合最佳化')
        
        # 界面佈局
        tk.Label(self, text='初始資金:').grid(row=0, column=0, padx=10, pady=10)
        self.input_initial_fund = tk.Entry(self)
        self.input_initial_fund.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(self, text='股票代碼（用逗號分隔）:').grid(row=1, column=0, padx=10, pady=10)
        self.input_stocks = tk.Entry(self)
        self.input_stocks.grid(row=1, column=1, padx=10, pady=10)
        
        tk.Label(self, text='交易成本 (%):').grid(row=2, column=0, padx=10, pady=10)
        self.input_transaction_cost = tk.Entry(self)
        self.input_transaction_cost.grid(row=2, column=1, padx=10, pady=10)
        
        tk.Label(self, text='風險容忍度 (0-100):').grid(row=3, column=0, padx=10, pady=10)
        self.risk_tolerance_scale = ttk.Scale(self, from_=0, to=100, orient='horizontal')
        self.risk_tolerance_scale.grid(row=3, column=1, padx=10, pady=10)
        
        self.optimize_button = tk.Button(self, text='最佳化投資組合', command=self.optimize_portfolio)
        self.optimize_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        
        self.result_label = tk.Label(self, text='', justify='left')
        self.result_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)
        
    def optimize_portfolio(self):
        try:
            initial_fund = float(self.input_initial_fund.get())
            stocks = self.input_stocks.get().split(',')
            transaction_cost = float(self.input_transaction_cost.get()) / 100
            risk_tolerance = float(self.risk_tolerance_scale.get()) / 100  # 轉換為小數形式
            
            if not (0 <= risk_tolerance <= 1):
                raise ValueError("風險容忍度必須在0到100之間")
            
            # 獲取股票數據
            try:
                data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
                data = data[stocks]  # 只選擇用戶輸入的股票代碼
            except FileNotFoundError:
                messagebox.showerror('錯誤', '找不到 stock_data.csv 文件。請確認文件存在並且路徑正確。')
                return
            except KeyError as e:
                messagebox.showerror('錯誤', f'股票代碼 {e} 不存在於資料中。')
                return
            
            returns = data.pct_change().mean() * 252
            cov_matrix = data.pct_change().cov() * 252
            
            def portfolio_performance(weights, returns, cov_matrix, transaction_cost):
                portfolio_return = np.sum(returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                adjusted_return = portfolio_return - transaction_cost * np.sum(np.abs(weights))
                return adjusted_return, portfolio_volatility
            
            def objective_function(weights, returns, cov_matrix, transaction_cost, risk_tolerance):
                p_return, p_volatility = portfolio_performance(weights, returns, cov_matrix, transaction_cost)
                penalty = np.sum((weights - np.mean(weights))**2)  # 懲罰項，權重偏差越大，懲罰越重
                return - (risk_tolerance * p_return - (1 - risk_tolerance) * p_volatility) + penalty
            
            num_assets = len(stocks)
            args = (returns, cov_matrix, transaction_cost, risk_tolerance)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_weights = num_assets * [1. / num_assets,]
            
            opts = minimize(objective_function, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            
            optimized_weights = opts.x
            opt_return, opt_volatility = portfolio_performance(optimized_weights, returns, cov_matrix, transaction_cost)
            
            result_text = (f'最佳化權重: {np.round(optimized_weights, 2)}\n'
                           f'預期回報率: {opt_return:.2f}\n'
                           f'預期波動率: {opt_volatility:.2f}')
            self.result_label.config(text=result_text)
            
        except Exception as e:
            messagebox.showerror('錯誤', str(e))

if __name__ == '__main__':
    app = PortfolioOptimizer()
    app.mainloop()
