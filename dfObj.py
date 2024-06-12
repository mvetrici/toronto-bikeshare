import numpy as np
import pandas as pd
import os
import io

MAX_LENGTH = 120

class dfObj():
    """A dataframe object with ."""
    def __init__(self, name, df: pd.DataFrame, dtype: str = 'DataFrame'):
        """
        self.name: str (name of the file)
        self.df: pd.DataFrame
        self.dtype: str (Trip, Weather, Bike Station, TTC Station)
        self.length: int (length of the dataframe)
        
        """
        self.name = name
        self._df = df
        self._dtype = dtype
        self._length = len(df)
        return

    def get_df(self) -> pd.DataFrame:
        return self._df

    def get_type(self) -> str:
        return self._dtype
    
    def __str__(self) -> str:
        return '\n' + f"{self.name} (type {self._dtype}):\n" + str(self.get_df())
    
    def getinfo(self):
        return self._df.info()
    
    # TODO!!!
    def add_col(self):
        """Add columns after creating bins."""
        print("Creating columns...")
        columns = []
        for col in self._df.columns:
            if ('date' in col.lower() or 'time' in col.lower()) and self._df[col].dtype == 'object':
                columns.append(col)
        print(columns)
        for col in columns[:1]: # NOTE
            try:
                self.df[col] = pd.to_datetime(self.df[col]) # format='%m/%d/%Y %H:%M'
                ext = ''
                if 'start' in col.lower():
                    ext = '_start'
                new_col = 'Datetime' + ext  
                self.df.rename({col: new_col}, axis=1, inplace=True)
                print("STILL creating columns...")
                # Month
                self.df['Month'] = self.df[new_col].dt.strftime("%B")
                # Weekday
                self.df['Weekday'] = self.df[new_col].dt.strftime("%A")
                
                # Season
                bins = [pd.to_datetime('12/21/2022'), 
                        pd.to_datetime('03/20/2023'), 
                        pd.to_datetime('06/20/2023'), 
                        pd.to_datetime('09/20/2023'),
                        pd.to_datetime('12/20/2023'),
                        pd.to_datetime('12/31/2023')] # right inclusive, include last  
                self.df["Season"] = pd.cut(self.df[new_col].dt.date, bins=bins, include_lowest=True, ordered=False, labels=["Winter", "Spring", "Summer", "Fall", "Winter"])
                
                # Time period
                if self.dtype != 'Weather':
                    bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                            pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                            pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                            pd.to_datetime('23:59:59').time()]
                    labels = ["Overnight", "AM", "Midday", "PM", "Evening", "Overnight"]
                    self.df["Time Period" + ext] = pd.cut(self.df[new_col].dt.time, bins=bins, include_lowest=True, ordered=False, labels=labels)
                
                if 'date' not in self.df.columns:
                    self.df['date'] = self.df[new_col].dt.date
            except ValueError:
                pass
        # then, use i value to iterate and find ending time (to do time period)
        # theoreticlaly  only happen for 
        if self.dtype == "Trip":
            for col in columns[1:]: 
                try: 
                    ext = ''
                    self.df[col] = pd.to_datetime(self.df[col])
                    if 'end' in col.lower(): 
                        ext = '_end'
                    new_col = "Datetime" + ext
                    self.df.rename({col: new_col}, axis=1, inplace=True)
                    bins = [pd.to_datetime('00:00:00').time(), pd.to_datetime('06:00:00').time(), 
                    pd.to_datetime('10:00:00').time(), pd.to_datetime('15:00:00').time(),
                    pd.to_datetime('19:00:00').time(), pd.to_datetime('23:00:00').time(),
                    pd.to_datetime('23:59:59').time()]
                    labels = ["Overnight", "AM", "Midday", "PM", "Evening", "Overnight"]
                    self.df["Time Period" + ext] = pd.cut(self.df[new_col].dt.time, bins=bins, include_lowest=True, ordered=False, labels=labels)
                except ValueError:
                    pass

        if self.dtype == 'Weather': 
            bins = [-16, 0, 5, 15, 30]
            labels = ["Freezing", "Cold", "Cool", "Warm"]
            self.df["Temperature Range"] = pd.cut(self.df["Mean_Temp_(°C)"], 
                    bins=bins, include_lowest=True, labels=labels)
            self.df["Precip"] = (self.df["Total_Precip_(mm)"] > 0)
        return 