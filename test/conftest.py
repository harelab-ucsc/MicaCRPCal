import sys
from unittest.mock import MagicMock

# Mock ROS2 packages if they are not installed in the current environment
mock_modules = [
    'rclpy', 'rclpy.node', 'rclpy.qos', 'rclpy.executors',
    'cv_bridge', 'sensor_msgs', 'sensor_msgs.msg',
    'std_msgs', 'std_msgs.msg',
    'rcl_interfaces', 'rcl_interfaces.msg', 'rcl_interfaces.srv',
    'custom_msgs', 'custom_msgs.msg',
    'as7265x_at_msgs', 'as7265x_at_msgs.msg',
    'qreader'
]

for name in mock_modules:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

if 'rcl_interfaces.msg' in sys.modules:
    class ParameterType:
        PARAMETER_BOOL = 1
        PARAMETER_INTEGER = 2
        PARAMETER_DOUBLE = 3
        PARAMETER_STRING = 4

    class ParameterValue:
        def __init__(self, type=0, bool_value=False, integer_value=0, double_value=0.0, string_value=""):
            self.type = type
            self.bool_value = bool_value
            self.integer_value = integer_value
            self.double_value = double_value
            self.string_value = string_value

    class Parameter:
        def __init__(self, name="", value=None):
            self.name = name
            self.value = value

    sys.modules['rcl_interfaces.msg'].ParameterType = ParameterType
    sys.modules['rcl_interfaces.msg'].ParameterValue = ParameterValue
    sys.modules['rcl_interfaces.msg'].Parameter = Parameter
