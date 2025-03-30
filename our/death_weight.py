import torch
death_weights = {
            'in_hosp_die': torch.tensor([45.0]).cuda(),
            'icu_die': torch.tensor([2.0]).cuda(),
            'icu_24hour_die': torch.tensor([2.0]).cuda()
        }