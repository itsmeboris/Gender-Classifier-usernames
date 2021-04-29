import os
import sys

os.system("nohup bash -c '" +
          sys.executable + " main.py --size 192 >result.txt" +
          "' &")
