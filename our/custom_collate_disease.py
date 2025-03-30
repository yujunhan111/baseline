import torch

def custom_collate_disease(batch):
    """
    Collate function that maintains batch structure for demographic, disease data, and death labels.

    Args:
        batch (list): List of dictionaries, each containing 'demographic', 'disease_data', and 'death_labels'

    Returns:
        dict: Properly collated batch with:
            - 'demographic': stacked tensor of demographic features
            - 'disease_data': list of disease data dictionaries maintaining batch structure
            - 'death_labels': list of death label dictionaries maintaining batch structure
    """
    return {
        'demographic': torch.stack([item['demographic'] for item in batch]),
        'disease_data': [item['disease_data'] for item in batch],
        'death_labels': [item['death_labels'] for item in batch]
    }