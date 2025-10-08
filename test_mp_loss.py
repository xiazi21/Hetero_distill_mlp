#!/usr/bin/env python3
"""Test script for the new meta-path loss function with total scaling."""

import torch
import torch.nn as nn
import sys
sys.path.append('.')
from two_teachjer_kd_update_han_direct_student_ts_v2 import meta_path_alignment_losses

def test_mp_loss():
    """Test the new meta-path loss function."""
    print("Testing meta-path loss function with total scaling...")
    
    device = torch.device('cpu')
    mp_teacher = {'mp1': torch.randn(10, 5), 'mp2': torch.randn(10, 5)}
    mp_student = {'mp1': torch.randn(10, 5), 'mp2': torch.randn(10, 5)}
    tail_teacher = torch.randn(10, 5)
    tail_student = torch.randn(10, 5)
    teacher_proj = nn.Linear(5, 5)
    student_proj = nn.Linear(5, 5)
    beta_teacher = torch.randn(10, 2)
    beta_student = torch.randn(10, 2)
    reliability = torch.randn(10)
    metapath_keys = ['mp1', 'mp2']

    # Test with total scaling
    result = meta_path_alignment_losses(
        mp_teacher=mp_teacher,
        mp_student=mp_student,
        tail_teacher=tail_teacher,
        tail_student=tail_student,
        teacher_proj=teacher_proj,
        student_proj=student_proj,
        beta_teacher=beta_teacher,
        beta_student=beta_student,
        reliability=reliability,
        metapath_keys=metapath_keys,
        component_weights={'feat': 1.0, 'relpos': 1.0, 'beta': 1.0},
        lambda_mp_total=2.0,
        balance_override=None
    )

    print('Meta-path loss test successful!')
    print(f'Total loss: {result["total"]:.4f}')
    print(f'Feature loss: {result["feature"]:.4f}')
    print(f'Relpos loss: {result["relpos"]:.4f}')
    print(f'Beta loss: {result["beta"]:.4f}')
    print(f'Scale: {result["weights"]["scale"]:.4f}')
    
    # Test with balance override
    result2 = meta_path_alignment_losses(
        mp_teacher=mp_teacher,
        mp_student=mp_student,
        tail_teacher=tail_teacher,
        tail_student=tail_student,
        teacher_proj=teacher_proj,
        student_proj=student_proj,
        beta_teacher=beta_teacher,
        beta_student=beta_student,
        reliability=reliability,
        metapath_keys=metapath_keys,
        component_weights={'feat': 1.0, 'relpos': 1.0, 'beta': 1.0},
        lambda_mp_total=2.0,
        balance_override=0.8  # 80% feature, 20% split between relpos and beta
    )
    
    print('\nMeta-path loss test with balance override successful!')
    print(f'Total loss: {result2["total"]:.4f}')
    print(f'Feature weight: {result2["weights"]["feat"]:.4f}')
    print(f'Relpos weight: {result2["weights"]["relpos"]:.4f}')
    print(f'Beta weight: {result2["weights"]["beta"]:.4f}')

if __name__ == '__main__':
    test_mp_loss()
