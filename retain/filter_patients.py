import numpy as np
import torch

# 至少有两次visit
def filter_valid_patients(data):
    valid_patients = {}
    for patient_id, patient_data in data.items():
        if (len(patient_data['events']) >= 1 and
               50 < (sum(len(event['codes']) for event in patient_data['events'])) <7000) :  #有两次以上visit，且所有code加起来需要小于7000，大于50(可根据显存调整)
            valid_patients[patient_id] = patient_data


    return valid_patients

