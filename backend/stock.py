#based on the data in the file "small_dataset.json", we are going to build an "Stock" object which will have the following attributes:
#['Name', 'Ticker', 'History', 'Sustainability', 'Institutional Holders', 'Balance Sheet', 'Country', 'Currency', 'Industry', 'Sector', 'Number of employees', 'Description']
#The "Stock" object will have the following methods:
# get_ESR_score(self)
# get_price_history(self)
import json
import pandas as pd


class Stock:
    def __init__(self, data):
        self.data = data
        self.Name = data['Name']
        self.Ticker = data['Ticker']
        self.History = data['History']
        self.history_df = None
        self.Sustainability = data['Sustainability']
        self.Institutional_Holders = data['Institutional Holders']
        self.Balance_Sheet = data['Balance Sheet']
        self.Country = data['Country']
        self.Currency = data['Currency']
        self.Industry = data['Industry']
        self.Sector = data['Sector']
        self.Number_of_employees = data['Number of employees']
        self.Description = data['Description']
    
    def get_ESG_score(self):
        return self.Sustainability['esgScores']['environmentScore'], self.Sustainability['esgScores']['totalEsg']
    
    def get_price_history(self):
        df = pd.DataFrame(self.History).T  # Transpose to get dates as rows
        df.index.name = 'Date'
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        self.history_df = df    
        return df
    
    def get_annualized_returns(self): 
        df = self.get_price_history()  
        df.index = pd.to_datetime(df.index)

        # Ensure the DataFrame is sorted by date (ascending)
        df = df.sort_index()

        # Get the initial and final close prices
        initial_price = df['Close'].iloc[0]
        final_price = df['Close'].iloc[-1]

        # Calculate the number of days between the first and last entry
        num_days = (df.index[-1] - df.index[0]).days
        num_years = num_days / 365.25  # Using 365.25 to account for leap years

        # Calculate the annualized return using the CAGR formula
        annualized_return = (final_price / initial_price) ** (1 / num_years) - 1

        return annualized_return

    
    def __str__(self):
        return f"Name: {self.Name}\nTicker: {self.Ticker}\nCountry: {self.Country}\nCurrency: {self.Currency}\nIndustry: {self.Industry}\nSector: {self.Sector}\nNumber of employees: {self.Number_of_employees}\nDescription: {self.Description}"
    
    def __repr__(self):
        return f"Stock({self.data})"

if __name__ == '__main__':
    pass