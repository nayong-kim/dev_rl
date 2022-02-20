# block.py

import numpy as np


class Block:
    @staticmethod
    def get_code(**args):
        pass
    
    @staticmethod
    def get_color(**args):
        pass

class EmptyBlock(Block):
    @staticmethod
    def get_code(**args):
        return 0
    
    @staticmethod
    def get_color():
        return 255, 255, 255

class CharacterBlock(Block):
    @staticmethod
    def get_code(**args):
        return 1
    
    @staticmethod
    def get_color():
        return 0, 255, 0

class EnemyBlock(Block):
    @staticmethod
    def get_code(**args):
        return 2
    
    @staticmethod
    def get_color():
        return 255, 0, 0


