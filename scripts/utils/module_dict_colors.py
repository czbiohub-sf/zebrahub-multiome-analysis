# A module to define the color palettes used in this paper
import matplotlib.pyplot as plt
import seaborn as sns

# a color palette for the "coarse" grained celltype annotation ("annotation_ML_coarse")
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3'
}

# Script to generate a color palette with 33 distinct colors
# # Start with the seaborn colorblind palette and extend it
# base_palette = sns.color_palette("Set3", 12)
# extended_palette = sns.color_palette("Set1", 9) + sns.color_palette("Pastel2", 8) + sns.color_palette("Dark2", 8)

# # Combine the base and extended palettes to get 33 unique colors
# combined_palette = base_palette + extended_palette

# # manually swap some colors from the front with the ones from the back
# teal_color = (0.0, 0.5019607843137255, 0.5019607843137255)  # RGB for teal
# combined_palette[-1] = teal_color  # Replace the light yellow with teal

# combined_palette[1] = combined_palette[-1]
# combined_palette[17] = combined_palette[-3]
# combined_palette[19] = combined_palette[-4]
# combined_palette[32] = (213/256, 108/256, 85/256)
# combined_palette[7] = (0.875, 0.29296875, 0.609375)
# combined_palette[25] = (0.75390625, 0.75390625, 0.0)
# combined_palette[21] = (0.22265625, 0.23046875, 0.49609375)

# combined_palette = combined_palette[:33]  # Ensure we only take the first 33 colors

# # Verify the palette length
# assert len(combined_palette) == 33, "The palette must have exactly 33 colors"

# # Plot the color palette
# plt.figure(figsize=(10, 2))
# sns.palplot(combined_palette)
# plt.title("Combined Palette with 32 Unique Colors")
# plt.show()