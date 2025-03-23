#! /usr/bin/env python3

import rospy
import torch

def main():
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()