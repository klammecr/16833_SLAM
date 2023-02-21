import pytest
from motion_model import wrap_angle
import numpy as np

class TestWrapAngle:
    def test_wrap_angle_pos(self):
        assert np.pi/2 == wrap_angle(np.pi/2)
    
    def test_wrap_angle_neg(self):
        assert -np.pi/2 == wrap_angle(-np.pi/2)
    
    def test_wrap_angle_over_pi(self):
        assert -np.pi/2 == wrap_angle(3*np.pi/2)
    
    def test_wrap_angle_under_neg_pi(self):
        assert np.pi/2 == wrap_angle(-3*np.pi/2)
    
    def test_wrap_angle_pi(self):
        assert -np.pi == wrap_angle(np.pi)
    
    def test_wrap_angle_neg_pi(self):
        assert -np.pi == wrap_angle(-np.pi)