# define the color palettes for the plots/figures for the manuscript
# 1) color palette for the cell type (annotation_ML_coarse)
# 2) color palette for the timepoints (viridis)

# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from colorsys import rgb_to_hsv

# Cell types ("annotation_ML_coarse") - coarse annotation by Merlin Lange
cell_types = [
       'epidermis', 'pronephros', 'hindbrain', 'axial_mesoderm',
       'neural_optic', 'neural_floor_plate', 'neural_crest', 'PSM',
       'optic_cup', 'lateral_plate_mesoderm',
       'midbrain_hindbrain_boundary', 'neural_telencephalon',
       'differentiating_neurons', 'muscle', 'fast_muscle',
       'heart_myocardium', 'somites', 'NMPs', 'pharyngeal_arches',
       'floor_plate', 'hemangioblasts', 'neural_posterior', 'tail_bud',
       'endoderm', 'hematopoietic_vasculature', 'endocrine_pancreas',
       'hatching_gland', 'neurons', 'notochord', 'enteric_neurons',
       'neural_unknown', 'primordial_germ_cells'
]

def color_dict_celltype(cell_types, combined_palette):
    # Create a dictionary mapping cell types to colors
    cell_type_color_dict = {cell_type: combined_palette[i] for i, cell_type in enumerate(cell_types)}

    return cell_type_color_dict

# Generate a color palette with 33 distinct colors
def color_palette_celltype():
    # Start with the seaborn colorblind palette and extend it
    base_palette = sns.color_palette("Set3", 12)
    extended_palette = sns.color_palette("Set1", 9) + sns.color_palette("Pastel2", 8) + sns.color_palette("Dark2", 8)

    # Combine the base and extended palettes to get 33 unique colors
    combined_palette = base_palette + extended_palette

    # manually swap some colors from the front with the ones from the back
    teal_color = (0.0, 0.5019607843137255, 0.5019607843137255)  # RGB for teal
    combined_palette[-1] = teal_color  # Replace the light yellow with teal

    combined_palette[1] = combined_palette[-1]
    combined_palette[17] = combined_palette[-3]
    combined_palette[19] = combined_palette[-4]
    combined_palette[32] = (213/256, 108/256, 85/256)
    combined_palette[7] = (0.875, 0.29296875, 0.609375)
    combined_palette[25] = (0.75390625, 0.75390625, 0.0)
    combined_palette[21] = (0.22265625, 0.23046875, 0.49609375)

    # Subset for 33 colors
    combined_palette = combined_palette[:33]  

    return combined_palette


# Additional function/module to sort the colors using "HSV" scale
# visually inspect the colormap by sorting based on the HSV scale
def sort_color_palette_hsv(combined_palette):
    # Convert RGB to HSV and sort by HSV values
    sorted_palette = sorted(combined_palette, key=lambda color: rgb_to_hsv(color[0], color[1], color[2]))

    # Visualize the sorted palette to manually check for similarity
    # Plot the color palette
    # plt.figure(figsize=(10, 2))
    # sns.palplot(sorted_palette)
    # plt.title("Combined Palette with 32 Unique Colors")
    # plt.show()

    return sorted_palette