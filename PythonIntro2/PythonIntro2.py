#listas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#def poviena(dalykas):
#    for sk in dalykas:
#        print(sk)

#poviena(listas)

#class klase:
#    para1 = "aha"
#    para2 = "ane"

#    def classfun(self):
#        print(self.para1)

#dalykas = klase()
#dalykas.classfun()
import DviFunc as df
import numpy as np
dict = {}
dict["vienas"] = 1
dict["du"] = 2

dict.pop("vienas") #remove
df.func()
df.funkc(dict["du"], 5)

arr = np.array([[1, 2, 3],[4, 5, 6]], dtype = np.int16)
print(arr)
zero_mas = np.zeros((10, 10))
one_mas = np.ones((5,5))
rand_mas = np.random.random((3,3))
skaic1, skaic2, skaic3 = np.loadtxt("skaiciai.txt", skiprows=1)
print(skaic1)

