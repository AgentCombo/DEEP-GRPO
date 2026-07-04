# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for math_grader.py numeric_equal function."""

import pytest
from math import isclose

from recipe.deep_grpo.reward.math_grader import numeric_equal


class TestNumericEqual:
    """Test cases for the numeric_equal function."""
    
    def test_basic_integer_equality(self):
        """Test basic equality with integer values."""
        assert numeric_equal(5.0, 5.0)
        assert numeric_equal(10.0, 10.0)
        assert numeric_equal(-3.0, -3.0)
        assert numeric_equal(0.0, 0.0)
    
    def test_basic_float_equality(self):
        """Test basic equality with float values."""
        assert numeric_equal(3.14, 3.14)
        assert numeric_equal(-2.718, -2.718)
        assert numeric_equal(0.001, 0.001)
    
    def test_relative_tolerance_precision(self):
        """Test relative tolerance boundary conditions."""
        # Just within relative tolerance (1e-4 for 1.0)
        assert numeric_equal(1.0001, 1.0)
        assert numeric_equal(0.9999, 1.0)
        
        # Just outside relative tolerance
        assert not numeric_equal(1.00011, 1.0)
        assert not numeric_equal(0.99989, 1.0)
    
    def test_large_numbers_with_relative_tolerance(self):
        """Test relative tolerance with large numbers."""
        large_number = 1000000.0
        tolerance = large_number * 1e-4  # 100
        
        # Within tolerance
        assert numeric_equal(large_number + tolerance - 1, large_number)
        assert numeric_equal(large_number - tolerance + 1, large_number)
        
        # Outside tolerance
        assert not numeric_equal(large_number + tolerance + 1, large_number)
        assert not numeric_equal(large_number - tolerance - 1, large_number)
    
    def test_small_numbers_with_relative_tolerance(self):
        """Test relative tolerance with small numbers."""
        small_number = 0.0001
        tolerance = small_number * 1e-4  # 1e-8
        
        # Within tolerance
        assert numeric_equal(small_number + tolerance - 1e-9, small_number)
        assert numeric_equal(small_number - tolerance + 1e-9, small_number)
        
        # Outside tolerance
        assert not numeric_equal(small_number + tolerance + 1e-9, small_number)
        assert not numeric_equal(small_number - tolerance - 1e-9, small_number)
    
    def test_zero_handling(self):
        """Test edge cases with zero."""
        # Zero values
        assert numeric_equal(0.0, 0.0)
        assert numeric_equal(-0.0, 0.0)
        assert numeric_equal(-0.0, -0.0)
        
        # Very small values near zero
        assert numeric_equal(1e-10, 0.0)  # Within relative tolerance of 0
        assert numeric_equal(-1e-10, 0.0)
        
        # Larger values that should not be considered equal to zero
        assert not numeric_equal(1e-3, 0.0)
        assert not numeric_equal(-1e-3, 0.0)
    
    def test_negative_numbers(self):
        """Test negative number handling."""
        assert numeric_equal(-5.0, -5.0)
        assert numeric_equal(-3.14, -3.14)
        
        # Negative vs positive - should not be equal
        assert not numeric_equal(5.0, -5.0)
        assert not numeric_equal(-3.14, 3.14)
    
    def test_nan_and_infinity(self):
        """Test NaN and infinity handling."""
        import math
        
        # NaN should not equal anything, including itself
        assert not numeric_equal(math.nan, math.nan)
        assert not numeric_equal(math.nan, 0.0)
        assert not numeric_equal(0.0, math.nan)
        
        # Infinity tests
        assert numeric_equal(math.inf, math.inf)
        assert numeric_equal(-math.inf, -math.inf)
        
        # Positive and negative infinity should not be equal
        assert not numeric_equal(math.inf, -math.inf)
        assert not numeric_equal(-math.inf, math.inf)
        
        # Infinity vs finite numbers
        assert not numeric_equal(math.inf, 1e100)
        assert not numeric_equal(-math.inf, -1e100)
    
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        large_num = 1e100
        tolerance = large_num * 1e-4
        
        # Within tolerance
        assert numeric_equal(large_num + tolerance - 1, large_num)
        assert numeric_equal(large_num - tolerance + 1, large_num)
        
        # Outside tolerance
        assert not numeric_equal(large_num + tolerance + 1, large_num)
        assert not numeric_equal(large_num - tolerance - 1, large_num)
    
    def test_identical_reference_implementation(self):
        """Test that numeric_equal matches math.isclose with same parameters."""
        test_cases = [
            (1.0, 1.0),
            (1.0001, 1.0),
            (0.9999, 1.0),
            (1.00011, 1.0),
            (0.99989, 1.0),
            (1000000.0001, 1000000.0),
            (0.00010001, 0.0001),
            (0.0, 0.0),
            (1e-10, 0.0),
        ]
        
        for pred, ref in test_cases:
            assert numeric_equal(pred, ref) == isclose(ref, pred, rel_tol=1e-4)
    
    def test_edge_case_precision(self):
        """Test precision edge cases."""
        # Test numbers with many decimal places
        assert numeric_equal(3.141592653589793, 3.141592653589793)
        assert numeric_equal(2.718281828459045, 2.718281828459045)
        
        # Test that different representations of the same value are equal
        assert numeric_equal(0.1 + 0.2, 0.3)  # Floating point arithmetic
        
        # Very close but not exactly equal floating point numbers
        assert numeric_equal(1.000000000000001, 1.0)
        assert not numeric_equal(1.00000000000001, 1.0)
    
    def test_negative_relative_tolerance_scenarios(self):
        """Test scenarios that could fail with relative tolerance."""
        # When reference is very small, relative tolerance becomes very small
        small_ref = 1e-10
        assert numeric_equal(small_ref * 1.0001, small_ref)  # Within 1e-4 relative
        assert not numeric_equal(small_ref * 1.001, small_ref)  # Outside 1e-4 relative
        
        # When prediction is much larger than reference
        assert not numeric_equal(1000.0, 1.0)
        assert not numeric_equal(0.001, 1000.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])