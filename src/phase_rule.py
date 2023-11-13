import pandas as pd

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