import numpy as np 
import cv2
from PIL import ImageGrab

mylist =[]
mylist.append(1) # listai, koki pirma idesi toks ir bus tipas

remainder = 11 % 3 #liekana

squared = 3 ** 2 #kelimas

cubed = 3 ** 3 #kubu ir tt

nesamone = "labas" * 10 #writes labas 10 times

indexOf = nesamone.index("a") #first occuring letter

countLetters = nesamone.count("a") #counts all a letters in text

substring - nesamone[3:7] #from to letter of string

substring - nesamone[3:7:2] #from to letter of string + step

invert = nesamone[::-1] #invert word

checkifStart = nesamone.startswith("la") #check if string starts with some sequence

checkifEnd = nesamone.endswith("la") #check if ends with sequence

splitstring = nesamone.split("s") # splits string on this symbol

for i in range(0, 5, 1): #start stop step
    print(i)
    if i == 4 :
        print("break")
        break

def myFunc(a):
    c = a + a
    return c

print(myFunc(5))

mylist2 = [2, 3, 5]