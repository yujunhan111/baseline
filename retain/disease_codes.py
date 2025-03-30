import torch
DISEASE_CODES = {

            # 'Atrial fibrillation': [9053,8807,3055,3142,3141,3136],
            'HEART FAILURE': [748,749,750,896,982,1552,1583],
            'chromosomalabnormalities': [943, 959, 1486],
            'pneumonia': [698, 699, 744, 745, 746, 1251, 1348, 1464, 1569, 1692, 1739, 1749, 1760],
            #'Respiratory Failure ': [742, 1374, 1549],
            'Tuberculosis  ':[1100, 1102, 1103, 1923, 1924, 1925],
            'Tuberculosis(删掉抗体)  ':[1100, 1102, 1103],
            #'Asthma':[736, 737, 738, 1303, 1407, 1491],
            'Nervous System Malformations':[1228, 1349, 1598, 1691],
            #'congenital malformations of intestine':[968, 980, 1195],
            'Heart Malformations':[899, 901, 916, 917, 1331, 1332, 1399, 1679, 1758],
            #'Respiratory Malformations Indices:':[909, 910, 915],
            'congenital malformations': [899, 901, 906, 909, 910, 915, 916, 917, 918, 928, 934, 960, 961, 962, 968, 977, 978, 979, 980, 1192, 1195, 1202, 1212, 1228, 1309, 1331, 1332, 1349, 1399, 1450, 1469, 1479, 1486, 1598, 1604, 1679, 1691, 1706, 1742, 1751, 1753, 1758],

        }

disease_weights = {
    # 'PTSD': torch.tensor([125.0]).cuda(),  # 0.008
    'Diabetes': torch.tensor([25.0]).cuda(),  # 0.04
    # 'Hyperlipidemia': torch.tensor([11.0]).cuda(),  # 0.091
     'Asthma': torch.tensor([30.0]).cuda(),  # 203（3.9%）
     'pneumonia': torch.tensor([12.0]).cuda(),  # 203（3.9%）
     'HEART FAILURE': torch.tensor([30.0]).cuda(),  # 203（3.9%）
     'Heart Malformations': torch.tensor([10.0]).cuda(),  # 203（3.9%）
     'Respiratory Malformations Indices': torch.tensor([30.0]).cuda(),  # 203（3.9%）
     'Nervous System Malformations': torch.tensor([30.0]).cuda(),  # 203（3.9%）
     'congenital malformations of intestine': torch.tensor([30.0]).cuda(),  # 203（3.9%）
     'Respiratory Failure': torch.tensor([100.0]).cuda(),  # 203（3.9%）
     'congenital malformations': torch.tensor([6.0]).cuda(),  # 203（3.9%）
     'Diseases of the Respiratory System': torch.tensor([6.0]).cuda(),  # 203（3.9%）
     'chromosomalabnormalities': torch.tensor([6.0]).cuda(),  # 203（3.9%）

}