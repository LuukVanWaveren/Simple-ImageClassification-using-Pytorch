
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 01:04:30 2019

@author: Luuk
"""

from IC_ImageClassification import ImageClassification
import IC_Input as I

if __name__ == '__main__':
    IC = ImageClassification()

    if not I.diag:
        input("enter any key to continue")
        IC.startLearning()
    else:
        print(f'\nDiagnostics mode is on, set diag to "False" to train and test cnn')