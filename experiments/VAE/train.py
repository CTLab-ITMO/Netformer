import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)


from lib.netformer.dataset import Net, act
net = Net(1, 10, act)
print(net)
