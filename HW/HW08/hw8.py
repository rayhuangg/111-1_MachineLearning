# %%
import math
# import numpy as np

total_n = 14
class1n = 6
class2n = 4
class3n = 4

class1y = 2
class2y = 0
class3y = 4

def cal_en(total_n, class1n, class2n, class3n, class1y, class2y, class3y):
    total_y = class1y + class2y + class3y
    en = -(((total_y/total_n) * math.log2(total_y/total_n + 1e-8)) + (((total_n-total_y)/total_n) * math.log2((total_n-total_y)/total_n + 1e-8)))
    en1 = -(((class1y/class1n) * math.log2(class1y/class1n + 1e-8)) + (((class1n-class1y)/class1n) * math.log2((class1n-class1y)/class1n + 1e-8)))
    en2 = -(((class2y/class2n) * math.log2(class2y/class2n + 1e-8)) + (((class2n-class2y)/class2n) * math.log2((class2n-class2y)/class2n + 1e-8)))

    if class3n == 0:
        en3 = 0
        gain = en - class1n/total_n * en1 - class2n/total_n * en2
    else:
        en3 = -(((class3y/class3n) * math.log2(class3y/class3n + 1e-8)) + (((class3n-class3y)/class3n) * math.log2((class3n-class3y)/class3n + 1e-8)))
        gain = en - class1n/total_n * en1 - class2n/total_n * en2 - class3n/total_n * en3

    print(f"en  = {en:0.4}")
    print(f"en1 = {en1:0.4}")
    print(f"en2 = {en2:0.4}")
    print(f"en3 = {en3:0.4f}")
    print(f"gain = {gain}")

cal_en(total_n, class1n, class2n, class3n, class1y, class2y, class3y)

# %%
