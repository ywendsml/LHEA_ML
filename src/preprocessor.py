import pandas as pd
from phase_rule import *

class DataReader:
    def __init__(self, file_path):
        self.path = file_path

    def load_data(self):
        self.data = pd.read_csv(self.path)
        return self.data

    def display_data(self, rows=10):
        print(self.data.head(rows)) 


class DataPreprocessor:
    def __init__(self, data, metal_elements, targeted_phases, other_phases):
        self.data = data
        # These are the columns to be kept
        self.metal_elements = metal_elements
        # These are the phases to be predicted
        self.targeted_phases = targeted_phases
        # These are the other phases to be grouped together
        self.other_phases = other_phases

    def group_phases(self):
        # Combine 'BCC' and '3_BCC' into a new 'BCC' category
        self.data.loc[:, 'BCC'] = self.data[['BCC', '3_BCC']].max(axis=1)
        # Keep 'FCC' as it is
        self.data.loc[:, 'FCC'] = self.data['FCC']
        # Combine all other categories into 'others'
        self.data.loc[:, 'others'] = self.data.loc[:, self.other_phases].max(axis=1)
        # Now drop the original columns
        self.data = self.data.drop(columns=self.other_phases + ['3_BCC'])

    def drop_columns(self):
        # Drop all columns that are not metal elements or targeted phases
        columns_to_drop = []
        for col in self.data.columns.tolist():
            if col not in self.metal_elements and col not in self.targeted_phases:
                columns_to_drop.append(col)
        self.data.drop(columns_to_drop, axis=1, inplace=True)

    def ouput_data(self):
        # Output the preprocessed data for the next steps
        return self.data
    
class PhaseRulePredictorCalculator:
    def __init__(self, data):
        self.data = data

if __name__ == '__main__':
    reader = DataReader('data/LHEA_v1_data.csv')
    data = reader.load_data()
    elements = ['Al', 'Cr', 'Fe', 'Mn', 'Ti', 'Nb', 'V', 'Mg', 'Zn', 'Li', 'Sn', 'Cu', 'Sc', 'Ta', 'Zr', 'Mo', 'Co', 'Ni', 'Hf', 'W']
    phases = ['BCC', 'FCC', 'others']
    others = ['L2_1', 'Laves', 'Zr_2_Al', 'IM']
    a_processor = DataPreprocessor(data, elements, phases, others)
    a_processor.group_phases()
    a_processor.drop_columns()
    Engineering_ready_data = a_processor.ouput_data()
    print(Engineering_ready_data.head(10))
    