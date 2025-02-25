import os
import pandas as pd
import numpy as np
from abc import ABC,abstractmethod
import zipfile

#Define an abstract class for data ingestion
class DataIngestion(ABC):
    @abstractmethod
    def ingest(self,file_path : str) -> pd.DataFrame:
        '''Abstract method to ingest data from a given filepath'''
        pass


#Implement a concrete class for Zip ingestion
class ZipDataIngestor(DataIngestion):
    def ingest(self,file_path: str) -> pd.DataFrame:
        """Extract the file and return the datframe"""

        # check weather it is a zip file or not
        if not file_path.endswith('.zip'):
            raise ValueError('Not a zip file')
        
        #extract zipfile
        with zipfile.ZipFile(file_path,'r') as file_zip:
            file_zip.extractall('extracted_data')

        # find the extracted csv file

        extracted_files = os.listdir('extracted_data')
        extracted_csv = [file for file in extracted_files if file.endswith('.csv')]

        if len(extracted_csv) == 0:
            raise FileNotFoundError('No CSV file found in the zip') 
        if len(extracted_csv) > 1:
            raise ValueError('Multiple CSV files found in the zip')
        
        csv_file_path = os.path.join('extracted_data',extracted_csv[0])
        df = pd.read_csv(csv_file_path)
        return df
    
# implement a factory for Data ingestion
class DataIngestionFactory:
    @staticmethod
    def get_data_ingestor(file_extension : str) -> DataIngestion:
        if file_extension == 'zip':
            return ZipDataIngestor()
        else:
            raise ValueError(f'File extension not supported : {file_extension}')
    
#Example usage
if __name__ == '__main__':
    # #specify the file path
    # file_path = r'C:\Users\Akhil.Reddy\Desktop\Loantap\data\logistic_regression.zip'

    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[-1][1:]

    # #create an instance of the ZipDataIngestor class
    # data_ingestor = DataIngestionFactory.get_data_ingestor(file_extension)

    # #Here we have the data stored in the dataframe

    # df = data_ingestor.ingest(file_path)

    pass




