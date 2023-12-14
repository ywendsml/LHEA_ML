from typing import List

from math import log, sqrt

import numpy as np
import pandas as pd


def get_combination(fixed_elements: List, other_possible_elements:List) -> List:
    """
    This function is to generate all possible combinations of elements in the alloy candidate
    """

    combinations_manual = []

    # Loop through the list to pick the first element
    for i in range(len(other_possible_elements)):
        # Loop through the remaining elements to pick the second one
        for j in range(i + 1, len(other_possible_elements)):
            # Loop through the remaining elements to pick the third one
            for k in range(j + 1, len(other_possible_elements)):
                # Loop through the remaining elements to pick the fourth one
                for l in range(k + 1, len(other_possible_elements)):
                    combinations_manual.append(fixed_elements + [other_possible_elements[i], other_possible_elements[j], 
                                                                other_possible_elements[k], other_possible_elements[l]])
                    
    print("Total Number of Combination: ")
    print(len(combinations_manual))
    return combinations_manual


# Read the database to get element details
def fetch_element_details(element_symbol):
    """
    This function is to fetch the element details from the database and return a dictionary of elements' detail.
    """

    database = pd.read_csv('./data/DATABASE.csv')
    element_row = database[database['Element'] == element_symbol]
    if len(element_row) == 0:
        print(f"Element {element_symbol} not found in the database!")
        exit(0)

    return element_row.iloc[0].to_dict()

# For example, user input is Al20Ca20V20Cu20Fe20, one has to break it down into Al, Ca, V, Cu, Fe and their corresponding percentages
# You could not input Al and expect the program to know that you want 100% Al. You have to specify the percentage. E.g. Al100

# Receive Alloy Composition and return number of components in the alloy candidate and element_info_list
def generate_element_info(elements):
    """
    This function is to generate element information list for the alloy candidate.
    """
    components = len(elements)
    if components < 2 or components > 6:
        print("Invalid number of components. It should be between 2 and 6.")
        exit(0)
    
    element_info_list = []
    for i in range(components):
        symbol = elements[i]
        #print(symbol)
        element_details = fetch_element_details(symbol)
        element_info_list.append(element_details)
        
    return components, element_info_list

def calc_pairwise_enthalpy_mixing(phi_a, phi_b, electron_density_a, electron_density_b, molar_volume_a, molar_volume_b, R_by_P_a, R_by_P_b, type_a, type_b, constant_a, constant_b):
    """
    This function is to calculate the pairwise enthalpy of mixing.
    """

    Q_by_P = 9.4
    reciprocal_electron_density_a = 1.0 / electron_density_a
    reciprocal_electron_density_b = 1.0 / electron_density_b

    c_A = (0.5 * molar_volume_a) / (0.5 * molar_volume_a + 0.5 * molar_volume_b)
    c_B = (0.5 * molar_volume_b) / (0.5 * molar_volume_a + 0.5 * molar_volume_b)

    molar_volume_a_alloy = molar_volume_a * (1.0 + constant_a * c_B * (phi_a - phi_b))
    molar_volume_b_alloy = molar_volume_b * (1.0 + constant_b * c_A * (phi_b - phi_a))

    # Determining the P value
    if type_a == "\t\tT" and type_b == "\t\tT":
        P = 14.1
    elif type_a == "\t\tNT" and type_b == "\t\tNT":
        P = 10.6
    elif type_a != type_b:
        P = 12.3

    # Calculating enthalpy of mixing for "a" mixed with "b"
    delta_H_mix_AB = -(phi_a - phi_b) * (phi_a - phi_b)
    delta_H_mix_AB += Q_by_P * (electron_density_a - electron_density_b) * (electron_density_a - electron_density_b)
    if type_a != type_b:
        delta_H_mix_AB -= 0.73 * R_by_P_a * R_by_P_b
    delta_H_mix_AB *= (2.0 * P * molar_volume_a_alloy) / (reciprocal_electron_density_a + reciprocal_electron_density_b)

    # Calculating enthalpy of mixing for "b" mixed with "a"
    delta_H_mix_BA = -(phi_b - phi_a) * (phi_b - phi_a)
    delta_H_mix_BA += Q_by_P * (electron_density_b - electron_density_a) * (electron_density_b - electron_density_a)
    if type_a != type_b:
        delta_H_mix_BA -= 0.73 * R_by_P_a * R_by_P_b
    delta_H_mix_BA *= (2.0 * P * molar_volume_b_alloy) / (reciprocal_electron_density_a + reciprocal_electron_density_b)

    # Averaging the enthalpies
    delta_H_mix = c_A * c_B * (0.5 * delta_H_mix_AB + 0.5 * delta_H_mix_BA)

    return delta_H_mix

# Function to calculate the density
def compute_density(percentage, element_info_subset):
    numerator = sum([0.01 * percentage[i] * element_info_subset[i]['\tAtomic Wt.(g/mol)'] for i in range(len(percentage))])
    denominator = sum([(0.01 * percentage[i] * element_info_subset[i]['\tAtomic Wt.(g/mol)']) / element_info_subset[i]['\tDensity(g/cm^3)'] for i in range(len(percentage))])
    return numerator / denominator

        
def compute_values(percentage, element_info_subset):
    num_components = len(percentage)
        
    avg_atomic_radius = sum([0.01 * percentage[i] * element_info_subset[i]['\tRadius(pm)'] for i in range(num_components)])
            
    # Calculate the atomic size difference (delta_radius)
    delta_radius = sum([0.01 * percentage[i] * (1.0 - element_info_subset[i]['\tRadius(pm)'] / avg_atomic_radius)**2 for i in range(num_components)])
            
    # Calculate enthalpy of mixing by summing pairwise enthalpy for all pairs
    enthalpy_mixing = 0.0
    for i in range(num_components):
        for j in range(i+1, num_components):
            enthalpy_mixing += 4.0 * 0.01 * percentage[i] * 0.01 * percentage[j] * calc_pairwise_enthalpy_mixing(
                element_info_subset[i]['\tPhi(V)'], element_info_subset[j]['\tPhi(V)'],
                element_info_subset[i]['\tnWS^1/3'], element_info_subset[j]['\tnWS^1/3'],
                element_info_subset[i]['\tVm^2/3(cm^2)'], element_info_subset[j]['\tVm^2/3(cm^2)'],
                element_info_subset[i]['\tR/P(v^2)'], element_info_subset[j]['\tR/P(v^2)'],
                element_info_subset[i]['\tType'], element_info_subset[j]['\tType'],
                element_info_subset[i]['\tConstant(a)'], element_info_subset[j]['\tConstant(a)']
            )
            
    # Calculate entropy of mixing
    entropy_mixing = -8.314 * sum([0.01 * percentage[i] * log(0.01 * percentage[i]) for i in range(num_components)])
            
    # Calculate melting temperature
    melting_temperature = sum([0.01 * percentage[i] * element_info_subset[i]['\tMelting Pt.(K)'] for i in range(num_components)])
        
    # Calculate omega
    omega = (melting_temperature * entropy_mixing) / abs(1000.0 * enthalpy_mixing)
            
    return sqrt(delta_radius) * 100, omega, enthalpy_mixing, entropy_mixing, melting_temperature


def calc_ML_features(components, element_info_list, density_max=5.0, min_percentage=5, max_percentage=35, step_percentage=1):
    """
    This function is awesome and to calculate the ML features for alloy candidates with 1 element combination.
    """    

    # Logic for five components
    feature_list = []
    if components == 5:
        for i in range(min_percentage, max_percentage+1, step_percentage):
            for j in range(min_percentage, max_percentage+1, step_percentage):
                for k in range(min_percentage, max_percentage+1, step_percentage):
                    for l in range(min_percentage, max_percentage+1, step_percentage):
                        m = 100 - i - j - k - l
                        temp_percentages = [i, j, k, l, m]
                        if m < max_percentage and m > min_percentage:
                            density = compute_density(temp_percentages, element_info_list[:5])
                            if density < density_max:
                                delta_radius, omega, enthalpy_mixing, entropy_mixing, melting_temperature = compute_values(temp_percentages, element_info_list[:5])
                                instance_feature = temp_percentages + [density] + [delta_radius] + [enthalpy_mixing] + [entropy_mixing] + [melting_temperature] + [omega]
                                feature_list.append(instance_feature)
        return feature_list
    else:
        print("Maximum number of components reached")
        exit(0)

def get_all_features(combinations_manual):
    """
    This function is to get all features for all alloy candidates with all element combination.
    Return: total_features_np, an ndarray of all alloy candidates with all features.
    """

    total_features = []

    for i in range(len(combinations_manual)):
        components, element_info = generate_element_info(combinations_manual[i])
        temp_array = calc_ML_features(components, element_info)
        element_string = ''.join(combinations_manual[i])
        temp_new_array = [[element_string] + row for row in temp_array]
        total_features.extend(temp_new_array)
        print(f"Done for {combinations_manual[i]}")
        print("Progress: ", i+1, "/", len(combinations_manual))

    total_feature_np = np.array(total_features)

    return total_feature_np

def phase_rule_calculator(fixed_elements = ['Al'], other_possible_elements = ['Ti','Mn','Mo','Nb','V']):
    """
    This function is to calculate the phase rule for alloy candidates, given the fixed elements and other possible elements.
    Return: total_features, an ndarray of all alloy candidates with all features.
    """

    combinations_manual = get_combination(fixed_elements, other_possible_elements)
    print("\n-----------------\n")

    total_features = get_all_features(combinations_manual)
    print("\n-----------------\n")
    
    return total_features

def phase_rule_calculatore_for_single_instance(elements=['Al','Ti','Mn','Mo','Nb'], percentage=[35, 35, 10, 10, 10]):
    """
    This function is to calculate the phase rule for a single instance, given the elements and their corresponding percentage.
    Return: total_features, an ndarray of all alloy candidates with all features.
    """

    _, element_info = generate_element_info(elements)

    density = compute_density(percentage, element_info[:5])

    delta_radius, omega, enthalpy_mixing, entropy_mixing, melting_temperature = compute_values(percentage, element_info[:5])
    instance_feature = percentage + [density] + [delta_radius] + [enthalpy_mixing] + [entropy_mixing] + [melting_temperature] + [omega]

    return instance_feature



if __name__ == "__main__":

    fixed_elements = ['Al']

    # other_possible_elements = ['Ti','Mn','Mo','Nb','V','Sn','Cu','Zr','Ni','Co','Cr','Fe','W','Ta','Sc','Y','Ag']
    
    other_possible_elements = ['Ti','Mn','Mo','Nb','V']

    total_features = phase_rule_calculator(fixed_elements=fixed_elements, other_possible_elements=other_possible_elements)

    print(total_features)

    test_single_instance_feature = phase_rule_calculatore_for_single_instance(elements=['Al','Ti','Mn','Mo','Nb'], percentage=[35, 35, 10, 10, 10])
    
    print(test_single_instance_feature)