from abc import ABC, abstractmethod
import os
import pandas as pd

# abstract basic data inspection strategies
#---------------------------------------------------------------
#subclasses must implement the inspect method

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame):
        """
        Perform specific type of data inspection

        parameters:
        df : the dataframe on which the inspection is to be performed

        returns None
        This method prints inspection results directly
        """
        pass

#Concrete strategy for DATATYPE inspection
#------------------------------------------

class DatatypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame):
        """
        Inspect the datatypes of the columns in the dataframe

        parameters:
        data : the dataframe on which the inspection is to be performed

        returns None
        This method prints inspection results directly


        """
        print("Datatypes of the Nullcounts in the dataframe: ")
        print(data.info())
        print("Summary statistics of Numerical Columns:  ")
        print(data.descripbe())
        print("Summary statistics of Categorical Columns: ")
        print(data.describe(include =['O']))

#Concrete strategy for summary inspection
#------------------------------------------

class SummaryInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame):
        """
        Inspect the summary of the columns in the dataframe

        parameters:
        data : the dataframe on which the inspection is to be performed

        returns None
        This method prints inspection results directly


        """
        print("Summary statistics of Numerical Columns:  ")
        print(data.descripbe())
        print("Summary statistics of Categorical Columns: ")
        print(data.describe(include =['O']))


#context class that uses  DataInspectionStrategy
#------------------------------------------------
class DataInspection:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataInspectionStrategy):

        #sets a new strategy for the data inspector
        self._strategy = strategy

    def execute_inspection(self, data: pd.DataFrame):
        #executes the inspection strategy
        self._strategy.inspect(data)


