import pandas as  pd 
import numpy as np
import random
np.random.seed(0)

# Reshape.
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
arr.reshape(4,3)
newarr = arr.reshape(2, 3,2)


# Return copy or view . base
print(arr.reshape(4,3).base)

# Numpy Array Join
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2)) # default axis = 0
arr = np.concatenate((arr1, arr2), axis = 1) # default axis = 0

# Use Stack - join along axis
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1) # default axis is 0
print(arr)

# Use Hstack - join along rows ; vstack - along coumns
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.hstack((arr1, arr2))
print(arr)

# Use dstack - join along depth

# Splitting Numpy Arrays
# Use array_split() de tra ve so array tuong ung
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3) # tra ve 1 list co 3 array.
print(newarr)

newarr = np.array_split(arr, 3,axis = 1) # tra ve 1 list co 3 array, axis = 1 along row

# Use hsplit() tra ve 1 list voi so array tuong ung
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr, 3)
print(newarr)

# Tuong tu voi vsplit(), dsplit()

# Searching Array
# Su dung where de tra ve 1 list gia tri theo dieu kien
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)

# Use searchsorted() : return index, theo gia tri da duoc sorted trc. 
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7)
print(x)
x = np.searchsorted(arr, 7, side='right') # sorted tu lon den nho, roi index


# Numpy Sorting Arrays
# Use sort()


# Numpy filter
arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)

# Tao ra 
import numpy as np
arr = np.array([41, 42, 43, 44])
# Create an empty list
filter_arr = []
# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)



# Random Numbers in Numpy
# Generate a random integer from 0 to 100
from numpy import random
x = random.randint(100)
print(x)

# Generate a random float between 0 to 1
from numpy import random
x = random.rand()
print(x)

# Generate a 1-D array containing 5 random integer from 0 to 100
from numpy import random
x=random.randint(100, size=(5))
print(x)

x=random.randint(100, size=(3,5))

# tuong tu nhi TH so thuc float : rand()

# Generate random number from array : random ket qua ngẫu nhiên từ 1 dãy số cố định
from numpy import random
x = random.choice([3, 5, 7, 9], size = (2,2)) # default is size = 1, want to specify into matrix
print(x)



# Random Data Distribution, có giá trị xác xuất
from numpy import random
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(1,100))
print(x)


# Random Permutation of Elements : shuffle() and permutation() : hoans vi
#Shuffle means changing arrangement of elements in-place. i.e. in the array itself.
from numpy import random
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr)
print(arr)

# permutation() : Hoan vi
from numpy import random
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(random.permutation(arr))


# Normal ( Gaussian ) Distribution
# Use random.normal() with loc ( mean ), scale(std), size : shape return array
from numpy import random 
x = random.normal(loc=1, scale=2, size=(2, 3))
print(x)

# Binimial Distribution with n( n-trials) and p(pro3), size(shape returned)
from numpy import random
x = random.binomial(n=10, p=0.5, size=10)
print(x)

# Possion Distribution with lam(lambda_pro3),size.
from numpy import random
x = random.poisson(lam=2, size=10)
print(x)

# Uniform Distribution with a(lower bound), b(upper bound), size, default is 0 to 1
from numpy import random
x = random.uniform(a = 0, b= 10,size=(2, 3))
print(x)


# Logistics Distribution with loc(mean),scale(std),size : logistics regression
from numpy import random
x = random.logistic(loc=1, scale=2, size=(2, 3))
print(x)

# Exponential Distribution : is used for describing time til next event
# scale ( lambda ), size ( shaoe )
from numpy import random
x = random.exponential(scale=2, size=(2, 3))
print(x)


# Chi Square Distribution : is used as a basic to verify the hypothesis
# It has two parameters :
# df ( degree of freedom)
# size(shape)
from numpy import random
x = random.chisquare(df=2, size=(2, 3))
print(x)

# Rayleigh Distribution : is used in signal processing
# scale(std) and size(shape)
from numpy import random
x = random.rayleigh(scale=2, size=(2, 3))
print(x)



# NumPy ufuncs : Universal Functions
# Ham zip()
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []
for i, j in zip(x, y):
  z.append(i + j)
print(z)

# ham add() de tom gon ma do lai
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = np.add(x, y)
print(z)


# Create your own ufunc
# defind users func sau do khai bao frompyfunc(function, input, output)
import numpy as np
def myadd(x, y):
  return x+y
myadd = np.frompyfunc(myadd, 2, 1)
print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))


# Check whether func is ufunc
import numpy as np
if type(np.add) == np.ufunc:
  print('add is ufunc')
else:
  print('add is not ufunc')

# Simple Arithmetics
# dung + - * / de lam
# Su dung cac ham trong phan nay
# add(), substract(), multiply(), divide()
# Power(x, n) : ham mu

import numpy as np
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 5, 6, 8, 2, 33])
newarr = np.power(arr1, arr2)
print(newarr)

# Remainder : lay phan du. mod() and remainder()
# 10%3, 20%7,....
import numpy as np
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
newarr = np.mod(arr1, arr2)
print(newarr)

import numpy as np
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
newarr = np.remainder(arr1, arr2)
print(newarr)

# Quotient and Mod
# Ham divmod() return 2 array, 1 is quotient(phan nguyen), 2 is mod(so du)
import numpy as np
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
newarr = np.divmod(arr1, arr2)
print(newarr)
# Results : (array([ 3,  2,  3,  5, 25,  1]), array([ 1,  6,  3,  0,  0, 27]))


# Absolute Values and abs()
# nen su dung absolute() de tranh confuse with py's inbuilt math.abs()
import numpy as np
arr = np.array([-1, -2, 1, 2, 3, -4])
newarr = np.absolute(arr)
print(newarr)



# Rounding Decimals
# Truncation () and fix() : xóa đi phần thập phân và trả về số thực gần 0 nhất.
import numpy as np
arr = np.trunc([-3.1666, 3.6667])
arr = np.fix([-3.1666, 3.6667])
print(arr)

# Rounding
# around() func : làm tròn lên 1 nếu phần thập phân >=0.5 or reverse
arr = np.around(3.1666, 2)
print(arr)

# floor() round off to lower
import numpy as np
arr = np.floor([-3.1666, 3.6667])
print(arr)

# Ceil() round off to upper
import numpy as np
arr = np.ceil([-3.1666, 3.6667])
print(arr)

# Numpy Logs
#Log2(), log at base 2
import numpy as np
arr = np.arange(1, 10)
print(np.log2(arr))

# log10() log base 10
import numpy as np
arr = np.arange(1, 10)
print(np.log10(arr))

# Natural log or log base e
# log() represent base e
import numpy as np
arr = np.arange(1, 10)
print(np.log(arr))

# Log any base, np không cung cấp, khi mà có bài toán đó, thì mình define hàm, rồi frompyfunc()
from math import log
import numpy as np
nplog = np.frompyfunc(log, 2, 1)
print(nplog(100, 15))
# results = 1.7005 : log15 (100)



# Numpy Summations
# add() them giua 2 obj, sum() dung cho n-obj
# de axis is 1, sum se tinh tong moi array
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
newarr = np.sum([arr1, arr2], axis=1)
print(newarr) # [6,6]

# Cummulative sum : cumsum()
import numpy as np
arr = np.array([1, 2, 3])
newarr = np.cumsum(arr)
print(newarr)


# Products
# Use prod()
import numpy as np
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
x = np.prod([arr1, arr2])
print(x)
# results : 40320 because 1*2*3*4*5*6*7*8 = 40320
# specify axis ís 1
import numpy as np
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
newarr = np.prod([arr1, arr2], axis=1)
print(newarr)
# result: [24 1680]

# Cummulative production
# cumprod() he partial product of [1, 2, 3, 4] is [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]
import numpy as np
arr = np.array([5, 6, 7, 8])
newarr = np.cumprod(arr)
print(newarr)


# Numpy Different : A discrete difference means subtracting two successive elements.
import numpy as np
arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr)
print(newarr)
# results : [5 10 -20]
arr = np.array([10, 15, 25, 5], n=2) # specify n_loop


# Finding LCM ( TÌm Bội chung nhỏ nhất.)
import numpy as np
num1 = 4
num2 = 6
x = np.lcm(num1, num2)
print(x) # is 12

import numpy as np
arr = np.array([3, 6, 9])
x = np.lcm.reduce(arr)
print(x)

# Finding GCD(Ước chung nhỏ nhất)
import numpy as np
num1 = 6
num2 = 9
x = np.gcd(num1, num2)
print(x) # is 3
# For array
import numpy as np
arr = np.array([20, 8, 32, 36, 16])
x = np.gcd.reduce(arr)
print(x)


# Numpy Trigonometric Functions
# sinx, cosx, tanx theo radians

# Convert deg to rad : use deg2rad() or rad2deg()
import numpy as np
arr = np.array([90, 180, 270, 360])
x = np.deg2rad(arr)
print(x)


# Finding angles
# arcsin, arccos, arctan
# Dinh ly PItago with Hypotenues
import numpy as np
base = 3
perp = 4
x = np.hypot(base, perp)
print(x) # is 5


# Numpy Set Operation
# Create Set using unique(): filter những giá trị trung nhau.
arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])
x = np.unique(arr)
print(x)

# Finding Union : Tìm unique giữa 2 array
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])
newarr = np.union1d(arr1, arr2)
print(newarr)

# Finding Intersection: intersect1d()
import numpy as np
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])
newarr = np.intersect1d(arr1, arr2, assume_unique=True)
print(newarr)

# Finding Difference : setdiff1d()
import numpy as np
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])
newarr = np.setdiff1d(set1, set2, assume_unique=True)
print(newarr)

# Chi tim nhung gia tri k nam trong 2 set : setor1d()
import numpy as np
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])
newarr = np.setxor1d(set1, set2, assume_unique=True)
print(newarr)





























