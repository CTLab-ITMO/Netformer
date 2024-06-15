import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)


from lib.netformer.dataset import Net, act
from converter import matrix_converter
net = Net(10, 1, act)
converted = matrix_converter(net)
print(converted)
