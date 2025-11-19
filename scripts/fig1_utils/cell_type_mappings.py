"""
Cell type annotation and tissue lineage mappings for zebrafish multiome analysis.

This module provides cell type to tissue lineage mappings used to organize
and categorize cell types in the zebrafish developmental atlas.
"""

from typing import Dict, List

# Cell type to tissue lineage mapping dictionary
# Maps broad tissue categories to specific cell type annotations
CELLTYPE_TO_LINEAGE: Dict[str, List[str]] = {
    "CNS": [
        "neural",
        "neural_optic",
        "neural_optic2",
        "neural_posterior",
        "neural_telencephalon",
        "neurons",
        "hindbrain",
        "midbrain_hindbrain_boundary",
        "midbrain_hindbrain_boundary2",
        "optic_cup",
        "spinal_cord",
        "differentiating_neurons",
        "floor_plate",
        "neural_floor_plate",
        "enteric_neurons",
    ],

    "Neural Crest": [
        "neural_crest",
        "neural_crest2"
    ],

    "Paraxial Mesoderm": [
        "somites",
        "fast_muscle",
        "muscle",
        "PSM",  # Presomitic mesoderm
        "floor_plate2",
        "NMPs",  # Neuromesodermal progenitors
        "tail_bud",
        "notochord",
    ],

    "Lateral Mesoderm": [
        "lateral_plate_mesoderm",
        "heart_myocardium",
        "hematopoietic_vasculature",
        "pharyngeal_arches",
        "pronephros",
        "pronephros2",
        "hemangioblasts",
        "hatching_gland",
    ],

    "Endoderm": [
        "endoderm",
        "endocrine_pancreas",
    ],

    "Epiderm": [
        "epidermis",
        "epidermis2",
        "epidermis3",
        "epidermis4"
    ],

    "Germline": [
        "primordial_germ_cells"
    ],
}


def map_celltype_to_tissue(cell_type: str, tissue_dict: Dict[str, List[str]] = None) -> str:
    """
    Map a cell type to its corresponding tissue lineage group.

    Searches through the tissue dictionary to find which broad tissue category
    a specific cell type belongs to. If the cell type is not found in any
    tissue category, returns "Unknown".

    Args:
        cell_type: The cell type annotation to map (e.g., "neural", "somites")
        tissue_dict: Dictionary mapping tissue groups to lists of cell types.
                    If None, uses the default CELLTYPE_TO_LINEAGE dictionary.

    Returns:
        The tissue lineage group the cell type belongs to, or "Unknown" if not found

    Example:
        >>> map_celltype_to_tissue("neural")
        'CNS'
        >>> map_celltype_to_tissue("somites")
        'Paraxial Mesoderm'
        >>> map_celltype_to_tissue("unknown_celltype")
        'Unknown'
    """
    if tissue_dict is None:
        tissue_dict = CELLTYPE_TO_LINEAGE

    for tissue, cell_types in tissue_dict.items():
        if cell_type in cell_types:
            return tissue

    return "Unknown"  # Return Unknown if cell type not found in dictionary


def get_celltype_by_lineage(lineage: str, tissue_dict: Dict[str, List[str]] = None) -> List[str]:
    """
    Get all cell types that belong to a specific tissue lineage.

    Args:
        lineage: The tissue lineage name (e.g., "CNS", "Paraxial Mesoderm")
        tissue_dict: Dictionary mapping tissue groups to lists of cell types.
                    If None, uses the default CELLTYPE_TO_LINEAGE dictionary.

    Returns:
        List of cell type annotations belonging to the specified lineage

    Example:
        >>> get_celltype_by_lineage("Neural Crest")
        ['neural_crest', 'neural_crest2']
    """
    if tissue_dict is None:
        tissue_dict = CELLTYPE_TO_LINEAGE

    return tissue_dict.get(lineage, [])


def get_all_lineages(tissue_dict: Dict[str, List[str]] = None) -> List[str]:
    """
    Get a list of all tissue lineage categories.

    Args:
        tissue_dict: Dictionary mapping tissue groups to lists of cell types.
                    If None, uses the default CELLTYPE_TO_LINEAGE dictionary.

    Returns:
        List of all tissue lineage names

    Example:
        >>> get_all_lineages()
        ['CNS', 'Neural Crest', 'Paraxial Mesoderm', ...]
    """
    if tissue_dict is None:
        tissue_dict = CELLTYPE_TO_LINEAGE

    return list(tissue_dict.keys())


def get_all_celltypes(tissue_dict: Dict[str, List[str]] = None) -> List[str]:
    """
    Get a flattened list of all cell type annotations across all lineages.

    Args:
        tissue_dict: Dictionary mapping tissue groups to lists of cell types.
                    If None, uses the default CELLTYPE_TO_LINEAGE dictionary.

    Returns:
        List of all cell type annotations

    Example:
        >>> celltypes = get_all_celltypes()
        >>> 'neural' in celltypes
        True
    """
    if tissue_dict is None:
        tissue_dict = CELLTYPE_TO_LINEAGE

    all_celltypes = []
    for cell_types in tissue_dict.values():
        all_celltypes.extend(cell_types)

    return all_celltypes
