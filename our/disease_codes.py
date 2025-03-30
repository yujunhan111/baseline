import torch
DISEASE_CODES = {
            'PTSD': [8745],
            'Type 2 diabetes': [9073, 8801, 8668, 8554, 8297],
            "Hyperlipidemia": [9015, 7767, 3134],
            "Depression": [866,869,2578,3150,8806,9030,9042,10999],
            "Obstructive airway disease": [942,1065,1067,1082,8974,9010],
            "Gastroesophageal reflux":[2921,3095,8914,7618,8498],
            "Arteriosclerosis": [418,419,7878,7689,9044],
            "acute kidney injury":[9045,847],
            "sepsis": [8906, 7471,8998],
            "liver dysfunction": [981],
            "coagulation disorder": [8911,554],
            "respiratory failure": [4229, 7816,8907,8691,7652,8250,935,1064],
            'acute respiratory distress':[7816,8907,935],
            'cardiogenic shock': [8454],
            'Delirium': [8412,871,7882,2832,2850,1645],
            'Atrial fibrillation': [9053,8807,3055,3142,3141,3136],
            'HEART FAILURE': [9047,8442,8398,8031,1024,1040,407,408,409],
            'Hypertension': [8482,9075,1044,421,436],
            'CHRONIC OBSTRUCTIVE PULMONARY DISEASE': [942,1065,1067,1082],
            'Acute pancreatitis': [8993],
            'Pulmonary Embolism ': [936,1084,1066],
            'Acute Cholecystitis': [8622],
            'Meningitis': [8640,976,1314,1315,1269,1268,1271],
            'Acute Leukemia': [851,855,522,508,509],
        }

disease_weights = {
    'PTSD': torch.tensor([125.0]).cuda(),  # 0.008
    'Type 2 diabetes': torch.tensor([25.0]).cuda(),  # 0.04
    'Hyperlipidemia': torch.tensor([11.0]).cuda(),  # 0.091
    'Depression': torch.tensor([30.0]).cuda(),  # 以前16,30效果更好
    'Obstructive airway disease': torch.tensor([50.0]).cuda(),  # 60的时候0.763,100涨了点。
    'Gastroesophageal reflux': torch.tensor([60.0]).cuda(),  # 以前25,45貌似涨了一点
    'Arteriosclerosis': torch.tensor([20.0]).cuda(),  # 以前30
    'acute kidney injury': torch.tensor([25.0]).cuda(),  # 30的时候是0.779
    'sepsis': torch.tensor([200.0]).cuda(),
    'liver dysfunction': torch.tensor([100.0]).cuda(),  # 11 (0.1%)
    'coagulation disorder': torch.tensor([100.0]).cuda(),  # 58 (0.5%)
    'respiratory failure': torch.tensor([100.0]).cuda(),  # 109 (1.0%)
    'acute respiratory distress': torch.tensor([200.0]).cuda(),  # 38 (0.7%)
    'cardiogenic shock': torch.tensor([200.0]).cuda(),
    'Delirium': torch.tensor([200.0]).cuda(),  # 23 (0.4%)
    'Atrial fibrillation': torch.tensor([30.0]).cuda(),  # 166（3.3%）
    'HEART FAILURE': torch.tensor([30.0]).cuda(),  # 203（3.9%）
    'Hypertension': torch.tensor([10.0]).cuda(),  # 382（10.8%）
    'CHRONIC OBSTRUCTIVE PULMONARY DISEASE': torch.tensor([200.0]).cuda(),  # 8（0.1%）
    'Acute pancreatitis': torch.tensor([200.0]).cuda(),  # 15 (0.3%)
    'Pulmonary Embolism ': torch.tensor([200.0]).cuda(),
    'Acute Cholecystitis': torch.tensor([200.0]).cuda(),
    'Meningitis': torch.tensor([200.0]).cuda(),
    'Acute Leukemia': torch.tensor([200.0]).cuda(),  # 2 (0.0%)
}